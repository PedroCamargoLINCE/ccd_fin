"""
Rolling origin / expanding window para avaliação temporal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class Split:
    name: str
    train_end: pd.Timestamp
    horizons: tuple[int, ...]  # meses à frente, ex: (1, 3, 6, 12)

    def test_dates(self) -> list[pd.Timestamp]:
        return [self.train_end + pd.DateOffset(months=h) for h in self.horizons]


DEFAULT_ORIGINS = (
    pd.Timestamp("2019-12-01"),
    pd.Timestamp("2020-12-01"),
    pd.Timestamp("2021-12-01"),
    pd.Timestamp("2022-12-01"),
)
DEFAULT_HORIZONS = (1, 3, 6, 12)


def rolling_origin(
    origins: tuple[pd.Timestamp, ...] = DEFAULT_ORIGINS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> Iterator[Split]:
    for o in origins:
        yield Split(name=f"origin_{o:%Y%m}", train_end=o, horizons=horizons)


def apply_split(panel: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (train_df, test_df) onde test cobre max(horizons) meses."""
    max_h = max(split.horizons)
    test_end = split.train_end + pd.DateOffset(months=max_h)
    train = panel[panel["date"] <= split.train_end].copy()
    test = panel[(panel["date"] > split.train_end) & (panel["date"] <= test_end)].copy()
    return train, test
