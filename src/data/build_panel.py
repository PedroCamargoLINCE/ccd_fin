"""
Painel long unificado: (cd_mun × date mensal) × todas as features + alvos.
"""
from __future__ import annotations

import pandas as pd

from src.data.load import (
    load_disease, load_climate, load_population, load_ppc, load_urban, load_density,
)
from src.utils.paths import DISEASES, MUNICIPIOS_ALVO, PROCESSED

MONTHS_RANGE = pd.date_range("2000-01-01", "2023-12-01", freq="MS")
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-01")


def _yearly_to_monthly(df: pd.DataFrame, col_name: str, how: str = "linear") -> pd.DataFrame:
    """
    Expande painel anual para mensal.
    - linear: interpolação linear entre Janeiros (bom para pop).
    - step: propaga valor do ano (bom para categorias que mudam pouco).
    """
    out = []
    for cd, g in df.groupby("cd_mun"):
        g = g.set_index("date").sort_index()[["value"]]
        g.columns = [col_name]
        mg = g.reindex(MONTHS_RANGE)
        if how == "linear":
            mg[col_name] = mg[col_name].interpolate("linear").bfill().ffill()
        else:
            mg[col_name] = mg[col_name].ffill().bfill()
        mg["cd_mun"] = cd
        mg.index.name = "date"
        out.append(mg.reset_index())
    return pd.concat(out, ignore_index=True)


def build_panel(
    municipios: dict | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Retorna painel (cd_mun × date × features+alvos) restrito aos municípios
    dados (default: 23 municípios-alvo).
    """
    municipios = municipios if municipios is not None else MUNICIPIOS_ALVO
    cd_keep = set(municipios.keys())

    # Alvos: taxa + contagem por doença
    target_frames = []
    for d in DISEASES:
        tx = load_disease(d, "rate")[["cd_mun", "date", "value"]].rename(columns={"value": f"tx_{d}"})
        n = load_disease(d, "count")[["cd_mun", "date", "value"]].rename(columns={"value": f"n_{d}"})
        target_frames.append(tx)
        target_frames.append(n)

    panel = None
    for tf in target_frames:
        tf = tf[tf["cd_mun"].isin(cd_keep)]
        panel = tf if panel is None else panel.merge(tf, on=["cd_mun", "date"], how="outer")

    # Clima (mensal direto)
    for var in ("evapot", "precip", "temp_min", "temp_max", "umid"):
        c = load_climate(var)
        c = c[c["cd_mun"].isin(cd_keep)][["cd_mun", "date", "value"]].rename(columns={"value": var})
        panel = panel.merge(c, on=["cd_mun", "date"], how="left")

    # Socio (anual → mensal)
    pop = _yearly_to_monthly(load_population()[lambda d: d["cd_mun"].isin(cd_keep)], "populacao", how="linear")
    ppc = _yearly_to_monthly(load_ppc()[lambda d: d["cd_mun"].isin(cd_keep)], "ppc", how="linear")
    urb = _yearly_to_monthly(load_urban()[lambda d: d["cd_mun"].isin(cd_keep)], "urbanizacao", how="linear")
    dens = _yearly_to_monthly(load_density()[lambda d: d["cd_mun"].isin(cd_keep)], "dens_demog", how="linear")

    for socio in (pop, ppc, urb, dens):
        panel = panel.merge(socio, on=["cd_mun", "date"], how="left")

    # Restringir ao período onde temos alvos
    panel = panel[panel["date"].between(MONTHS_RANGE.min(), MONTHS_RANGE.max())].copy()

    # Preencher taxa faltante a partir de contagem + populacao (hansen./tuberc. não têm 2023)
    for d in DISEASES:
        tx_col, n_col = f"tx_{d}", f"n_{d}"
        if tx_col in panel.columns and n_col in panel.columns:
            panel[tx_col] = panel[tx_col].astype(float)
            mask = panel[tx_col].isna() & panel[n_col].notna() & panel["populacao"].notna()
            panel.loc[mask, tx_col] = panel.loc[mask, n_col] / panel.loc[mask, "populacao"] * 1e5

    # Features temporais
    panel["year"] = panel["date"].dt.year
    panel["month_of_year"] = panel["date"].dt.month
    panel["time_idx"] = ((panel["date"].dt.year - 2000) * 12 + panel["date"].dt.month - 1).astype(int)
    panel["covid_period"] = panel["date"].between(COVID_START, COVID_END).astype(int)

    panel["nm_mun"] = panel["cd_mun"].map(municipios)

    cols_order = (
        ["cd_mun", "nm_mun", "date", "year", "month_of_year", "time_idx", "covid_period"]
        + [f"tx_{d}" for d in DISEASES]
        + [f"n_{d}" for d in DISEASES]
        + ["populacao", "dens_demog", "ppc", "urbanizacao"]
        + ["evapot", "precip", "temp_min", "temp_max", "umid"]
    )
    panel = panel[[c for c in cols_order if c in panel.columns]]
    panel = panel.sort_values(["cd_mun", "date"]).reset_index(drop=True)

    if save:
        out = PROCESSED / "panel_23munis.parquet"
        panel.to_parquet(out, index=False)
        print(f"[build_panel] salvo em {out} — shape={panel.shape}")
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(p.head())
    print(f"dtypes:\n{p.dtypes}")
    print(f"munis únicos: {p['cd_mun'].nunique()}")
    print(f"período: {p['date'].min()} → {p['date'].max()}")
    print(f"missing por coluna:\n{p.isna().mean().round(3).sort_values(ascending=False).head(10)}")
