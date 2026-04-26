"""
Leitores dos arquivos brutos (Excel/ODS/CSV) em formato wide, retornando
painel long com colunas ['cd_mun', 'nm_mun', 'date', 'value'].

Cada família de arquivo tem um parser dedicado porque as convenções de
cabeçalho diferem (MM-YYYY., YYYY/MÊS, DS_POP_XX, ano bruto).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils.paths import RAW, TAXAS, PROCESSED, DISEASES

MONTH_PT_TO_NUM = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


def _normalize_cd_mun(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(7)
    )


def _read_excel(path: Path) -> pd.DataFrame:
    # calamine reads xlsx (incl. strict OOXML) e ods; robusto e rápido
    return pd.read_excel(path, engine="calamine")


def _parse_date_disease(col: str) -> pd.Timestamp | None:
    # "2000/Jan" ... "2023/Dez"
    m = re.match(r"(\d{4})[/_-](\w{3})", str(col))
    if not m:
        return None
    year = int(m.group(1))
    mon = MONTH_PT_TO_NUM.get(m.group(2).lower()[:3])
    if mon is None:
        return None
    return pd.Timestamp(year=year, month=mon, day=1)


def _parse_date_climate(col: str) -> pd.Timestamp | None:
    # "01-1999.", "12-2023."
    m = re.match(r"(\d{1,2})-(\d{4})", str(col))
    if not m:
        return None
    return pd.Timestamp(year=int(m.group(2)), month=int(m.group(1)), day=1)


def _parse_year_column(col: str) -> int | None:
    # "DS_POP_00" (dens_demog), "2000" (PPC/Pop/Urban yearly)
    s = str(col)
    m = re.search(r"_(\d{2})$", s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 70 else 1900 + yy
    m = re.fullmatch(r"(19|20)\d{2}(\.0)?", s)
    if m:
        return int(float(s))
    return None


def _melt_wide(
    df: pd.DataFrame,
    id_cols: list[str],
    date_parser,
) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in id_cols]
    parsed = {c: date_parser(c) for c in value_cols}
    keep = [c for c, d in parsed.items() if d is not None]
    if not keep:
        raise ValueError(
            f"Nenhuma coluna temporal reconhecida. Amostra: {value_cols[:5]}"
        )
    long = df.melt(id_vars=id_cols, value_vars=keep, var_name="_col", value_name="value")
    long["date"] = long["_col"].map(parsed)
    long = long.drop(columns=["_col"])
    return long


def load_disease(disease: str, kind: str = "rate") -> pd.DataFrame:
    """
    kind = 'rate' carrega Taxas/TX_<disease>_00_23.xlsx (taxa)
    kind = 'count' carrega <disease>_00_23.xlsx (contagem bruta)
    Retorna long: [cd_mun, nm_mun, date, value]
    """
    assert disease in DISEASES, f"doença desconhecida: {disease}"
    assert kind in ("rate", "count")
    path = (TAXAS / f"TX_{disease}_00_23.xlsx") if kind == "rate" else (RAW / f"{disease}_00_23.xlsx")
    df = _read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    id_cols = [c for c in ("CD_MUN", "COD_SUS", "MUN") if c in df.columns]
    long = _melt_wide(df, id_cols, _parse_date_disease)
    long = long.rename(columns={"CD_MUN": "cd_mun", "MUN": "nm_mun"})
    long["cd_mun"] = _normalize_cd_mun(long["cd_mun"])
    if "COD_SUS" in long.columns:
        long = long.drop(columns=["COD_SUS"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["cd_mun", "nm_mun", "date", "value"]].sort_values(["cd_mun", "date"]).reset_index(drop=True)


def load_climate(var: str) -> pd.DataFrame:
    """var ∈ {'evapot','precip','temp_min','temp_max','umid'}"""
    files = {
        "evapot": RAW / "Evapot_SP.xlsx",
        "precip": RAW / "Precip_SP.xlsx",
        "temp_min": RAW / "Temp_Min_SP.xlsx",
        "temp_max": RAW / "Temp_Max_SP.ods",
        "umid": RAW / "Umid_SP.xlsx",
    }
    path = files[var]
    df = _read_excel(path)
    df.columns = [str(c).strip().rstrip(".") for c in df.columns]
    id_cols = [c for c in ("CD_MUN", "MUN") if c in df.columns]
    long = _melt_wide(df, id_cols, _parse_date_climate)
    long = long.rename(columns={"CD_MUN": "cd_mun", "MUN": "nm_mun"})
    long["cd_mun"] = _normalize_cd_mun(long["cd_mun"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["cd_mun", "nm_mun", "date", "value"]].sort_values(["cd_mun", "date"]).reset_index(drop=True)


def load_yearly_wide(path: Path) -> pd.DataFrame:
    """Lê arquivos anuais (PPC, Pop, Urban, Dens) → long com date = 1º-jan do ano."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")
    else:
        df = _read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    id_cols = [c for c in ("CD_MUN", "MUN", "NM_MUN") if c in df.columns]
    value_cols = [c for c in df.columns if c not in id_cols]
    years = {c: _parse_year_column(c) for c in value_cols}
    keep = [c for c, y in years.items() if y is not None]
    if not keep:
        raise ValueError(f"Nenhum ano reconhecido em {path.name}. Cols: {value_cols[:5]}")
    long = df.melt(id_vars=id_cols, value_vars=keep, var_name="_col", value_name="value")
    long["year"] = long["_col"].map(years)
    long["date"] = pd.to_datetime(long["year"].astype("Int64").astype(str) + "-01-01")
    long = long.drop(columns=["_col", "year"])
    long = long.rename(columns={"CD_MUN": "cd_mun", "MUN": "nm_mun", "NM_MUN": "nm_mun"})
    long["cd_mun"] = _normalize_cd_mun(long["cd_mun"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[["cd_mun", "nm_mun", "date", "value"]].sort_values(["cd_mun", "date"]).reset_index(drop=True)


def load_population() -> pd.DataFrame:
    return load_yearly_wide(RAW / "Pop_Geral_SP.xlsx")


def load_ppc() -> pd.DataFrame:
    return load_yearly_wide(RAW / "Indice_PPC_SP.xlsx")


def load_urban() -> pd.DataFrame:
    return load_yearly_wide(RAW / "Urban_SP.xlsx")


def load_density() -> pd.DataFrame:
    return load_yearly_wide(RAW / "Dens_demog_SP.csv")


@lru_cache(maxsize=64)
def cached_parquet(key: str, loader_repr: str) -> pd.DataFrame:
    """Cache em parquet para acelerar iterações."""
    path = PROCESSED / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"{path} não existe — rode build_cache() primeiro.")


def build_cache() -> None:
    """Persiste cada fonte em parquet. Executar uma vez após mudar fontes."""
    jobs: list[tuple[str, callable]] = [
        *[(f"disease_rate_{d}", lambda d=d: load_disease(d, "rate")) for d in DISEASES],
        *[(f"disease_count_{d}", lambda d=d: load_disease(d, "count")) for d in DISEASES],
        *[(f"climate_{v}", lambda v=v: load_climate(v)) for v in ("evapot", "precip", "temp_min", "temp_max", "umid")],
        ("socio_pop", load_population),
        ("socio_ppc", load_ppc),
        ("socio_urban", load_urban),
        ("socio_density", load_density),
    ]
    for key, fn in jobs:
        out = PROCESSED / f"{key}.parquet"
        if out.exists():
            print(f"[skip] {key}")
            continue
        print(f"[load] {key} ...", end=" ", flush=True)
        df = fn()
        df.to_parquet(out, index=False)
        print(f"rows={len(df):,} -> {out.name}")


if __name__ == "__main__":
    build_cache()
