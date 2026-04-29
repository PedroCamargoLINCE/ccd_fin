"""
Deep learning de painel via pytorch-forecasting:
- DeepAR (GRU+NB distribuição)
- N-BEATS (univariado, sem covariáveis)
- N-HiTS (hierárquico, suporta covariáveis)
- Temporal Fusion Transformer (TFT — covariáveis estáticas + temporais + future-known)

Treinamento: Lightning Trainer, mixed precision (bf16), rolling origin.
Loss padrão: NegativeBinomialDistributionLoss (alvo = contagem).
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
    NBeats,
    NHiTS,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
    QuantileLoss,
    SMAPE,
)

from src.eval.metrics import evaluate
from src.utils.paths import CHECKPOINTS, DISEASES, LIGHTNING_LOGS, PROCESSED, PROJECT_ROOT
from src.utils.splits import DEFAULT_HORIZONS, DEFAULT_ORIGINS, Split, rolling_origin

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")

CLIMATE_VARS = ["evapot", "precip", "temp_min", "temp_max", "umid"]
SOCIO_VARS = ["populacao", "dens_demog", "ppc", "urbanizacao"]


@dataclass
class DeepConfig:
    target_col: str  # "n_<disease>" (contagem) ou "tx_<disease>" (taxa)
    max_encoder_length: int = 36   # 3 anos de contexto
    max_prediction_length: int = 12
    batch_size: int = 128
    max_epochs: int = 200            # teto alto — early stopping decide
    patience: int = 15               # paciência para val_loss melhorar
    min_delta: float = 1e-5          # melhoria mínima considerada
    lr_patience: int = 5             # paciência do ReduceLROnPlateau
    learning_rate: float = 1e-3
    hidden_size: int = 32
    dropout: float = 0.2
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "32"


def _prepare_panel(panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = panel.copy()
    df["cd_mun"] = df["cd_mun"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)
    # N-BEATS/NHiTS não aceitam inteiros no target — cast para float
    df[target_col] = df[target_col].astype(float)
    return df


def make_dataset(
    df: pd.DataFrame,
    cfg: DeepConfig,
    train_end_time_idx: int,
    include_covariates: bool = True,
    allow_missing_timesteps: bool = False,
    for_nb_loss: bool = False,
    fixed_lengths: bool = False,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    target = cfg.target_col
    static_cats = ["cd_mun"]
    time_vary_known_reals = ["time_idx", "month_of_year", "covid_period"]
    time_vary_unknown_reals = [target]
    if include_covariates:
        time_vary_known_reals += CLIMATE_VARS  # tratamos como known (simplificação — ver nota no plano)
        static_reals = SOCIO_VARS
    else:
        static_reals = []

    # NB loss exige center=False; log1p é mais estável que softplus em séries com
    # muitos zeros (softplus degenera e produz NaN nas predições).
    if for_nb_loss:
        normalizer = GroupNormalizer(groups=["cd_mun"], transformation="log1p", center=False)
    else:
        normalizer = GroupNormalizer(groups=["cd_mun"], transformation="softplus")

    min_enc = cfg.max_encoder_length if fixed_lengths else cfg.max_encoder_length // 2
    min_pred = cfg.max_prediction_length if fixed_lengths else 1

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_end_time_idx],
        time_idx="time_idx",
        target=target,
        group_ids=["cd_mun"],
        min_encoder_length=min_enc,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=min_pred,
        max_prediction_length=cfg.max_prediction_length,
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_reals=time_vary_known_reals,
        time_varying_unknown_reals=time_vary_unknown_reals,
        target_normalizer=normalizer,
        add_relative_time_idx=not fixed_lengths,  # NHiTS exige False
        add_target_scales=True,
        add_encoder_length=not fixed_lengths,     # NHiTS exige False
        allow_missing_timesteps=allow_missing_timesteps,
        randomize_length=None if fixed_lengths else None,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    return training, validation


def _trainer(cfg: DeepConfig, log_name: str) -> L.Trainer:
    ckpt = ModelCheckpoint(
        dirpath=CHECKPOINTS / log_name,
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
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
    )


def train_deepar(
    df: pd.DataFrame, cfg: DeepConfig, train_end_time_idx: int, log_name: str,
) -> tuple[DeepAR, TimeSeriesDataSet]:
    tr, val = make_dataset(df, cfg, train_end_time_idx, include_covariates=True, for_nb_loss=True)
    train_loader = tr.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_loader = val.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0)

    model = DeepAR.from_dataset(
        tr,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        rnn_layers=2,
        dropout=cfg.dropout,
        loss=NegativeBinomialDistributionLoss(),
        reduce_on_plateau_patience=cfg.lr_patience,
    )
    trainer = _trainer(cfg, log_name)
    trainer.fit(model, train_loader, val_loader)
    return model, val


def train_nhits(
    df: pd.DataFrame, cfg: DeepConfig, train_end_time_idx: int, log_name: str,
) -> tuple[NHiTS, TimeSeriesDataSet]:
    tr, val = make_dataset(df, cfg, train_end_time_idx, include_covariates=True, fixed_lengths=True)
    train_loader = tr.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_loader = val.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0)

    model = NHiTS.from_dataset(
        tr,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size * 2,
        dropout=cfg.dropout,
        loss=QuantileLoss(),  # N-HiTS usa saída determinística; QuantileLoss dá distribucional barato
        reduce_on_plateau_patience=cfg.lr_patience,
    )
    trainer = _trainer(cfg, log_name)
    trainer.fit(model, train_loader, val_loader)
    return model, val


def train_tft(
    df: pd.DataFrame, cfg: DeepConfig, train_end_time_idx: int, log_name: str,
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    tr, val = make_dataset(df, cfg, train_end_time_idx, include_covariates=True)
    train_loader = tr.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
    val_loader = val.to_dataloader(train=False, batch_size=cfg.batch_size * 4, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        tr,
        learning_rate=cfg.learning_rate,
        hidden_size=cfg.hidden_size,
        attention_head_size=4,
        dropout=cfg.dropout,
        hidden_continuous_size=cfg.hidden_size // 2,
        loss=QuantileLoss(),
        log_interval=0,
        reduce_on_plateau_patience=cfg.lr_patience,
    )
    trainer = _trainer(cfg, log_name)
    trainer.fit(model, train_loader, val_loader)
    return model, val


def predict_and_score(
    model, val_dataset: TimeSeriesDataSet, df: pd.DataFrame, cfg: DeepConfig,
    split: Split, model_name: str, disease: str,
) -> list[dict]:
    preds = model.predict(val_dataset, return_index=True, return_x=False, trainer_kwargs={"accelerator": cfg.accelerator, "devices": cfg.devices})
    y_hat = preds.output if hasattr(preds, "output") else preds[0]
    index = preds.index if hasattr(preds, "index") else preds[1]
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()

    # y_hat shape: (n_series, pred_len) ou (n_series, pred_len, n_quantiles)
    if y_hat.ndim == 3:
        # pegar mediana (índice médio)
        mid = y_hat.shape[-1] // 2
        y_hat_point = y_hat[..., mid]
    else:
        y_hat_point = y_hat

    records = []
    pred_len = y_hat_point.shape[1]
    for series_i, (_, row) in enumerate(index.iterrows()):
        cd_mun = row["cd_mun"]
        base_time = int(row["time_idx"])
        # obter ground truth
        gt = df[df["cd_mun"] == cd_mun].set_index("time_idx")[cfg.target_col]
        for step in range(pred_len):
            t = base_time + step
            if t not in gt.index:
                continue
            h = step + 1
            if h not in split.horizons:
                continue
            y_true = float(gt.loc[t])
            y_pred = float(y_hat_point[series_i, step])
            records.append({
                "model": model_name, "disease": disease, "horizon": h,
                "origin": split.name, "cd_mun": cd_mun, "time_idx": t,
                "y_true": y_true, "y_pred": y_pred,
            })
    return records


def run_deep_single(
    panel: pd.DataFrame, disease: str, model_name: str = "tft",
    origins=DEFAULT_ORIGINS, horizons=DEFAULT_HORIZONS,
    target_kind: str = "count", cfg_overrides: dict | None = None,
) -> pd.DataFrame:
    target_col = f"n_{disease}" if target_kind == "count" else f"tx_{disease}"
    cfg = DeepConfig(target_col=target_col, max_prediction_length=max(horizons))
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)

    df = _prepare_panel(panel, target_col)

    all_records = []
    for split in rolling_origin(origins, horizons):
        train_end_time_idx = int(df[df["date"] == split.train_end]["time_idx"].iloc[0])
        log_name = f"{model_name}_{disease}_{split.name}"
        trainer_fn = {"deepar": train_deepar, "nhits": train_nhits, "tft": train_tft}[model_name]
        model, val_ds = trainer_fn(df, cfg, train_end_time_idx, log_name)
        records = predict_and_score(model, val_ds, df, cfg, split, model_name, disease)
        all_records.extend(records)
        # metric summary por horizonte
        rec_df = pd.DataFrame(records)
        for h in horizons:
            sub = rec_df[rec_df["horizon"] == h]
            if len(sub) == 0:
                continue
            m = evaluate(sub["y_true"].values, sub["y_pred"].values, name=model_name, disease=disease, horizon=h)
            print(f"  [{split.name}] h={h}: mae={m['mae']:.3f} rmse={m['rmse']:.3f} n={m['n']}")
    return pd.DataFrame(all_records)


if __name__ == "__main__":
    import sys
    panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    disease = sys.argv[1] if len(sys.argv) > 1 else "tuberculose"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tft"
    print(f"== Treinando {model_name} em {disease} ==")
    df = run_deep_single(panel, disease=disease, model_name=model_name,
                         cfg_overrides={"max_epochs": 40, "patience": 7})
    out = PROJECT_ROOT / "reports" / f"deep_{model_name}_{disease}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")
