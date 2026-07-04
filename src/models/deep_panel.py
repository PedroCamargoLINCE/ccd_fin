"""
Deep learning de painel via pytorch-forecasting:
- DeepAR  (RNN + NegativeBinomial — alvo contagem)
- N-HiTS  (hierárquico, covariáveis no encoder)
- Temporal Fusion Transformer (TFT)
- LSTM / GRU (RecurrentNetwork autorregressivo)  [NOVO — pedido da revisão]

Treinamento: Lightning Trainer, rolling origin.

============================================================================
CORREÇÕES DE VALIDAÇÃO (revisão 2026-07) — 3 problemas encontrados na v antiga
============================================================================
[FIX 1] JANELA DE TESTE ENVIESADA (avaliação inválida)
  Antes: `validation = TimeSeriesDataSet.from_dataset(training, df, predict=True)`
  passava o painel INTEIRO com predict=True -> para TODA origem, previa sempre
  os últimos `max_prediction_length` meses do painel (a janela final, 2023).
  O `train_end_time_idx` só afetava o treino, nunca o alvo. Resultado: os deep
  models eram avaliados 4x na MESMA janela final, enquanto os baselines faziam
  rolling-origin real -> comparação não era apples-to-apples.
  Agora: o dataset de teste é recortado até `train_end + H`, de modo que o
  decoder cai exatamente em [T+1, T+H] daquela origem.

[FIX 2] VAZAMENTO DE COVARIÁVEL FUTURA (viés a favor dos deep)
  Antes: variáveis de clima entravam como `time_varying_known_reals` -> o modelo
  via o clima REAL do futuro na janela de previsão (lookahead). No mundo real
  você não conhece o clima de 2023 ao prever 2023. Os baselines usam só clima
  DEFASADO (lag 1/3/12), então a comparação era duplamente injusta.
  Agora: clima entra como `time_varying_unknown_reals` (só encoder/passado).
  Apenas covariáveis deterministicamente conhecidas no futuro (calendário,
  período COVID, time_idx) permanecem como known.

[FIX 3] VAZAMENTO DO TESTE NO EARLY STOPPING (seleção de modelo enviesada)
  Antes: o MESMO dataset usado para early stopping/checkpoint (val_loss) era o
  dataset de teste -> o modelo "espiava" o teste para decidir quando parar e
  qual checkpoint guardar. Agora há uma janela de validação SEPARADA, recortada
  de dentro do treino ([T-H+1, T]), que nunca toca o teste ([T+1, T+H]).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    DeepAR,
    NHiTS,
    RecurrentNetwork,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import (
    MAE,
    NegativeBinomialDistributionLoss,
    QuantileLoss,
)

from src.eval.metrics import evaluate
from src.utils.paths import CHECKPOINTS, DISEASES, LIGHTNING_LOGS, PROCESSED, PROJECT_ROOT
from src.utils.splits import DEFAULT_HORIZONS, DEFAULT_ORIGINS, Split, rolling_origin

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")

CLIMATE_VARS = ["evapot", "precip", "temp_min", "temp_max", "umid"]
SOCIO_VARS = ["populacao", "dens_demog", "ppc", "urbanizacao"]
# Covariáveis genuinamente conhecidas no futuro (determinísticas / de calendário).
KNOWN_FUTURE_REALS = ["time_idx", "month_of_year", "covid_period"]


@dataclass
class DeepConfig:
    target_col: str
    max_encoder_length: int = 36
    max_prediction_length: int = 12
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-5
    lr_patience: int = 5
    learning_rate: float = 1e-3
    hidden_size: int = 32
    dropout: float = 0.2
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "32"
    seed: int = 42  # [NOVO] semente p/ reprodutibilidade e estudo de variância


def _prepare_panel(panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = panel.copy()
    df["cd_mun"] = df["cd_mun"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)
    df[target_col] = df[target_col].astype(float)
    return df


def make_datasets(
    df: pd.DataFrame,
    cfg: DeepConfig,
    train_end_time_idx: int,
    include_covariates: bool = True,
    for_nb_loss: bool = False,
    fixed_lengths: bool = False,
    recurrent: bool = False,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Retorna (training, val_es, test) com validação temporal honesta.

    training : time_idx <= T - H          (fit)
    val_es   : decoder em [T-H+1, T]       (early stopping — NÃO toca o teste)
    test     : decoder em [T+1,  T+H]      (avaliação da origem)
    onde T = train_end_time_idx e H = cfg.max_prediction_length.
    """
    T = train_end_time_idx
    H = cfg.max_prediction_length

    target = cfg.target_col
    static_cats = ["cd_mun"]
    time_known = list(KNOWN_FUTURE_REALS)
    # [FIX 2] clima é DESCONHECIDO no futuro -> só encoder.
    time_unknown = [target]
    static_reals: list[str] = []
    if include_covariates and not recurrent:
        time_unknown += CLIMATE_VARS
        static_reals = SOCIO_VARS
    # RecurrentNetwork (LSTM/GRU) é autorregressivo e não aceita reais
    # desconhecidos além do alvo -> roda com alvo + calendário conhecido.

    if for_nb_loss:
        normalizer = GroupNormalizer(groups=["cd_mun"], transformation="log1p", center=False)
    else:
        normalizer = GroupNormalizer(groups=["cd_mun"], transformation="softplus")

    min_enc = cfg.max_encoder_length if fixed_lengths else cfg.max_encoder_length // 2
    min_pred = cfg.max_prediction_length if fixed_lengths else 1

    common = dict(
        time_idx="time_idx",
        target=target,
        group_ids=["cd_mun"],
        min_encoder_length=min_enc,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=min_pred,
        max_prediction_length=H,
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_reals=time_known,
        time_varying_unknown_reals=time_unknown,
        target_normalizer=normalizer,
        add_relative_time_idx=not fixed_lengths,
        add_target_scales=True,
        add_encoder_length=not fixed_lengths,
        allow_missing_timesteps=False,
    )

    # [FIX 1 + FIX 3] training fit em <= T-H; validação de early stopping em
    # [T-H+1, T]; teste em [T+1, T+H]. As normalizações/encoders são fitados
    # apenas no `training` e reaproveitados via from_dataset (sem vazamento).
    training = TimeSeriesDataSet(df[df["time_idx"] <= T - H], **common)
    val_es = TimeSeriesDataSet.from_dataset(
        training, df[df["time_idx"] <= T], predict=True, stop_randomization=True
    )
    test = TimeSeriesDataSet.from_dataset(
        training, df[df["time_idx"] <= T + H], predict=True, stop_randomization=True
    )
    return training, val_es, test


def _trainer(cfg: DeepConfig, log_name: str) -> L.Trainer:
    ckpt = ModelCheckpoint(
        dirpath=CHECKPOINTS / log_name, filename="best",
        monitor="val_loss", save_top_k=1, mode="min",
    )
    es = EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min", min_delta=cfg.min_delta)
    return L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=0.1,
        callbacks=[ckpt, es],
        default_root_dir=LIGHTNING_LOGS,
        enable_progress_bar=False,
        log_every_n_steps=20,
        enable_model_summary=False,
        deterministic="warn",  # [NOVO] reprodutibilidade (com seed_everything)
    )


def _fit(model, cfg, training, val_es, log_name):
    train_loader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_loader = val_es.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0)
    trainer = _trainer(cfg, log_name)
    trainer.fit(model, train_loader, val_loader)
    return model


def train_deepar(df, cfg, train_end_time_idx, log_name):
    training, val_es, test = make_datasets(df, cfg, train_end_time_idx, include_covariates=True, for_nb_loss=True)
    model = DeepAR.from_dataset(
        training, learning_rate=cfg.learning_rate, hidden_size=cfg.hidden_size,
        rnn_layers=2, dropout=cfg.dropout, loss=NegativeBinomialDistributionLoss(),
        reduce_on_plateau_patience=cfg.lr_patience,
    )
    return _fit(model, cfg, training, val_es, log_name), test


def train_nhits(df, cfg, train_end_time_idx, log_name):
    training, val_es, test = make_datasets(df, cfg, train_end_time_idx, include_covariates=True, fixed_lengths=True)
    model = NHiTS.from_dataset(
        training, learning_rate=cfg.learning_rate, hidden_size=cfg.hidden_size * 2,
        dropout=cfg.dropout, loss=QuantileLoss(), reduce_on_plateau_patience=cfg.lr_patience,
    )
    return _fit(model, cfg, training, val_es, log_name), test


def train_tft(df, cfg, train_end_time_idx, log_name):
    training, val_es, test = make_datasets(df, cfg, train_end_time_idx, include_covariates=True)
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=cfg.learning_rate, hidden_size=cfg.hidden_size,
        attention_head_size=4, dropout=cfg.dropout, hidden_continuous_size=cfg.hidden_size // 2,
        loss=QuantileLoss(), log_interval=0, reduce_on_plateau_patience=cfg.lr_patience,
    )
    return _fit(model, cfg, training, val_es, log_name), test


def _train_recurrent(df, cfg, train_end_time_idx, log_name, cell_type):
    """[NOVO] LSTM / GRU autorregressivo via RecurrentNetwork."""
    training, val_es, test = make_datasets(
        df, cfg, train_end_time_idx, include_covariates=False, recurrent=True
    )
    model = RecurrentNetwork.from_dataset(
        training, cell_type=cell_type, hidden_size=cfg.hidden_size,
        rnn_layers=2, dropout=cfg.dropout, learning_rate=cfg.learning_rate,
        loss=MAE(), reduce_on_plateau_patience=cfg.lr_patience,
    )
    return _fit(model, cfg, training, val_es, log_name), test


def train_lstm(df, cfg, train_end_time_idx, log_name):
    return _train_recurrent(df, cfg, train_end_time_idx, log_name, cell_type="LSTM")


def train_gru(df, cfg, train_end_time_idx, log_name):
    return _train_recurrent(df, cfg, train_end_time_idx, log_name, cell_type="GRU")


TRAINERS = {
    "deepar": train_deepar,
    "nhits": train_nhits,
    "tft": train_tft,
    "lstm": train_lstm,
    "gru": train_gru,
}


def predict_and_score(model, test_dataset, df, cfg, split, model_name, disease):
    preds = model.predict(
        test_dataset, return_index=True, return_x=False,
        trainer_kwargs={"accelerator": cfg.accelerator, "devices": cfg.devices},
    )
    y_hat = preds.output if hasattr(preds, "output") else preds[0]
    index = preds.index if hasattr(preds, "index") else preds[1]
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
    if y_hat.ndim == 3:
        mid = y_hat.shape[-1] // 2
        y_hat_point = y_hat[..., mid]
    else:
        y_hat_point = y_hat

    records = []
    pred_len = y_hat_point.shape[1]
    for series_i, (_, row) in enumerate(index.iterrows()):
        cd_mun = row["cd_mun"]
        base_time = int(row["time_idx"])  # 1º passo do decoder = T+1 (por origem)
        gt = df[df["cd_mun"] == cd_mun].set_index("time_idx")[cfg.target_col]
        for step in range(pred_len):
            t = base_time + step
            if t not in gt.index:
                continue
            h = step + 1
            if h not in split.horizons:
                continue
            records.append({
                "model": model_name, "disease": disease, "horizon": h,
                "origin": split.name, "cd_mun": cd_mun, "time_idx": t,
                "y_true": float(gt.loc[t]), "y_pred": float(y_hat_point[series_i, step]),
            })
    return records


def run_deep_single(
    panel, disease, model_name="tft",
    origins=DEFAULT_ORIGINS, horizons=DEFAULT_HORIZONS,
    target_kind="count", cfg_overrides=None, seed: int = 42,
):
    target_col = f"n_{disease}" if target_kind == "count" else f"tx_{disease}"
    cfg = DeepConfig(target_col=target_col, max_prediction_length=max(horizons), seed=seed)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    df = _prepare_panel(panel, target_col)
    trainer_fn = TRAINERS[model_name]
    all_records = []
    for split in rolling_origin(origins, horizons):
        L.seed_everything(cfg.seed, workers=True)  # [NOVO] reprodutibilidade por origem
        train_end_time_idx = int(df[df["date"] == split.train_end]["time_idx"].iloc[0])
        log_name = f"{model_name}_{disease}_{split.name}_seed{cfg.seed}"
        model, test_ds = trainer_fn(df, cfg, train_end_time_idx, log_name)
        records = predict_and_score(model, test_ds, df, cfg, split, model_name, disease)
        all_records.extend(records)
        rec_df = pd.DataFrame(records)
        for h in horizons:
            sub = rec_df[rec_df["horizon"] == h]
            if len(sub) == 0:
                continue
            m = evaluate(sub["y_true"].values, sub["y_pred"].values, name=model_name, disease=disease, horizon=h)
            print(f"  [{split.name}] h={h}: mae={m['mae']:.3f} rmse={m['rmse']:.3f} n={m['n']}")
    out = pd.DataFrame(all_records)
    out["seed"] = cfg.seed
    return out


def run_deep_multiseed(panel, disease, model_name="tft", seeds=(42, 1, 7), **kw):
    """[NOVO] roda N sementes -> permite reportar média +/- desvio (item variância)."""
    frames = [run_deep_single(panel, disease, model_name, seed=s, **kw) for s in seeds]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys
    panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    disease = sys.argv[1] if len(sys.argv) > 1 else "tuberculose"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tft"
    seeds = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (42,)
    print(f"== Treinando {model_name} em {disease} | seeds={seeds} ==")
    df = run_deep_multiseed(panel, disease=disease, model_name=model_name, seeds=seeds)
    out = PROJECT_ROOT / "reports" / f"deep_{model_name}_{disease}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")
