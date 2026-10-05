from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DatasetConfig, ExperimentConfig


FEATURE_NAMES = (
    "ret_1", "ret_5", "ret_10", "ret_20", "intraday", "range",
    "ma20_gap", "vol_5", "vol_20", "volume_change", "volume_z20",
    "dollar_volume_z20",
)


def _shift_ratio(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[lag:] = np.log(x[lag:] / x[:-lag])
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(filled, axis=0)], axis=0)
    cc = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(finite, axis=0)], axis=0)
    sums = cs[window:] - cs[:-window]
    counts = cc[window:] - cc[:-window]
    out = np.full_like(x, np.nan, dtype=np.float32)
    out[window - 1:] = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts >= max(3, window // 2))
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(x, window)
    mean2 = _rolling_mean(x * x, window)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)


def _cross_z(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
    return ((x - mean) / np.where(std > 1e-6, std, np.nan)).astype(np.float32)


@dataclass
class DateBatch:
    x: np.ndarray
    market: np.ndarray
    y: np.ndarray
    raw_return: np.ndarray
    assets: np.ndarray
    retrieval: np.ndarray
    complementarity: np.ndarray
    date_index: int
    date: str


class PanelData:
    """Past-only feature panel with date batches for cross-sectional learning."""

    def __init__(self, root: Path, spec: DatasetConfig, cfg: ExperimentConfig):
        self.spec, self.cfg = spec, cfg
        packed = np.load(root / spec.file, allow_pickle=False)
        self.dates = packed["dates"].astype("datetime64[D]")
        self.symbols = packed["symbols"].astype(str)
        values = packed["values"].astype(np.float32)
        self.features, self.targets_raw = self._engineer(values)
        self.targets = _cross_z(self.targets_raw)
        self.market = self._market_context()
        self.split_indices = self._make_splits()
        self._scale_from_training()
        self.complementarity = self._past_complementarity()
        self._retrieval_cache: dict[int, np.ndarray] = {}

    def _engineer(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        opn, high, low, close, adj, volume = [values[:, :, i] for i in range(6)]
        ret1, ret5, ret10, ret20 = [_shift_ratio(adj, lag) for lag in (1, 5, 10, 20)]
        with np.errstate(divide="ignore", invalid="ignore"):
            intraday = np.log(close / opn)
            price_range = np.log(high / low)
            log_volume = np.log1p(volume)
            dollar_volume = np.log1p(np.maximum(close * volume, 0.0))
        ma20 = _rolling_mean(adj, 20)
        ma20_gap = np.log(adj / ma20)
        vol5, vol20 = _rolling_std(ret1, 5), _rolling_std(ret1, 20)
        volume_change = _shift_ratio(np.maximum(volume, 1.0), 1)
        volume_z = (log_volume - _rolling_mean(log_volume, 20)) / (_rolling_std(log_volume, 20) + 1e-6)
        dollar_z = (dollar_volume - _rolling_mean(dollar_volume, 20)) / (_rolling_std(dollar_volume, 20) + 1e-6)
        features = np.stack([
            ret1, ret5, ret10, ret20, intraday, price_range, ma20_gap,
            vol5, vol20, volume_change, volume_z, dollar_z,
        ], axis=-1).astype(np.float32)
        target = np.full_like(adj, np.nan, dtype=np.float32)
        step = self.cfg.horizon + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            adjusted_open = opn * adj / close
            target[:-step] = np.log(adjusted_open[step:] / adjusted_open[1:-self.cfg.horizon])
        return features, target

    def _market_context(self) -> np.ndarray:
        f = self.features
        context = np.stack([
            np.nanmean(f[:, :, 0], axis=1), np.nanstd(f[:, :, 0], axis=1),
            np.nanmean(np.where(np.isfinite(f[:, :, 0]), (f[:, :, 0] > 0).astype(float), np.nan), axis=1), np.nanmean(f[:, :, 1], axis=1),
            np.nanmedian(f[:, :, 10], axis=1), np.nanstd(f[:, :, 1], axis=1),
        ], axis=1).astype(np.float32)
        context[~np.isfinite(context)] = 0.0
        return context

    def _scale_from_training(self) -> None:
        fit_end = self.split_indices["fit"][-1]
        sample = self.features[:fit_end + 1]
        mean = np.nanmean(sample, axis=(0, 1), keepdims=True)
        std = np.nanstd(sample, axis=(0, 1), keepdims=True)
        self.features = np.clip((self.features - mean) / np.where(std > 1e-6, std, 1.0),-8.0,8.0).astype(np.float32)
        market_sample = self.market[:fit_end + 1]
        mm, ms = np.nanmean(market_sample, axis=0), np.nanstd(market_sample, axis=0)
        self.market = np.clip((self.market - mm) / np.where(ms > 1e-6, ms, 1.0),-8.0,8.0).astype(np.float32)

    def _past_complementarity(self) -> np.ndarray:
        f = self.features
        role = np.stack([f[:, :, 0], f[:, :, 3], -f[:, :, 1], -f[:, :, 8] - 0.25 * f[:, :, 10]], axis=-1)
        role = np.stack([_cross_z(role[:, :, i]) for i in range(4)], axis=-1)
        residual = role - self.targets[:, :, None]
        out = np.zeros((len(self.dates), 4, 4), dtype=np.float32)
        maturity = self.cfg.horizon + 1
        for t in range(len(self.dates)):
            stop, start = t - maturity, max(0, t - maturity - 60)
            if stop <= start:
                continue
            block = residual[start:stop].reshape(-1, 4)
            block = block[np.isfinite(block).all(axis=1)]
            if len(block) < 24:
                continue
            corr = np.corrcoef(block, rowvar=False)
            comp = np.nan_to_num(np.clip(1.0 - corr, 0.0, 2.0), nan=0.0)
            np.fill_diagonal(comp, 0.0)
            out[t] = comp.astype(np.float32)
        return out

    def _make_splits(self) -> dict[str, np.ndarray]:
        eligible = np.arange(len(self.dates))
        observed = np.isfinite(self.features).all(axis=2).astype(np.int16)
        cumulative = np.concatenate([np.zeros((1, observed.shape[1]), dtype=np.int32),
                                     np.cumsum(observed, axis=0, dtype=np.int32)], axis=0)
        usable = np.zeros_like(observed, dtype=bool)
        usable[self.cfg.lookback - 1:] = (
            cumulative[self.cfg.lookback:] - cumulative[:-self.cfg.lookback]
        ) == self.cfg.lookback
        self.feature_eligible = usable
        feature_count = usable.sum(axis=1)
        eligible = eligible[(eligible >= self.cfg.lookback - 1) & (feature_count >= 8)]
        train_end, valid_end, test_end = map(np.datetime64, (self.spec.train_end, self.spec.valid_end, self.spec.test_end))
        eligible = eligible[eligible + self.cfg.horizon + 1 < len(self.dates)]
        target_end_index = eligible + self.cfg.horizon + 1
        target_end_dates = self.dates[target_end_index]
        splits = {
            "train": eligible[target_end_dates <= train_end],
            "valid": eligible[(self.dates[eligible] > train_end) & (target_end_dates <= valid_end)],
            "test": eligible[(self.dates[eligible] > valid_end) & (target_end_dates <= test_end)],
        }
        train = splits["train"]
        if len(train) < 100:
            raise ValueError(f"{self.spec.name}: insufficient origins for chronological router training")
        route_start = train[int(len(train) * (1.0 - self.cfg.router_fraction))]
        before_route = train[train + self.cfg.horizon + 1 < route_start]
        tune_start = before_route[int(len(before_route) * 0.9)]
        splits["fit"] = before_route[before_route + self.cfg.horizon + 1 < tune_start]
        splits["core_valid"] = before_route[before_route >= tune_start]
        splits["router"] = train[train >= route_start]
        return splits

    def _retrieval(self, t: int) -> np.ndarray:
        if t in self._retrieval_cache:
            return self._retrieval_cache[t]
        result = np.zeros((len(self.symbols), len(FEATURE_NAMES) + 10), dtype=np.float32)
        stop = t - self.cfg.horizon - 1
        if stop <= self.cfg.lookback:
            self._retrieval_cache[t] = result
            return result
        candidates = np.arange(max(self.cfg.lookback, stop - 252), stop)
        distances = np.square(self.market[candidates] - self.market[t]).mean(axis=1)
        neighbors = candidates[np.argsort(distances)[:8]]
        for asset in range(len(self.symbols)):
            labels = self.targets[neighbors, asset]
            good = np.isfinite(labels)
            if not good.any():
                continue
            neigh = neighbors[good]
            weights = np.exp(-distances[np.searchsorted(candidates, neigh)])
            weights /= weights.sum() + 1e-8
            last_features = self.features[neigh, asset]
            valid_rows = np.isfinite(last_features).all(axis=1)
            if valid_rows.any():
                w = weights[valid_rows]
                w /= w.sum() + 1e-8
                result[asset, :len(FEATURE_NAMES)] = (last_features[valid_rows] * w[:, None]).sum(axis=0)
            clipped = np.clip(labels[good], -2.5, 2.5)
            hist, _ = np.histogram(clipped, bins=np.linspace(-2.5, 2.5, 11), weights=weights)
            result[asset, len(FEATURE_NAMES):] = hist / (hist.sum() + 1e-8)
        self._retrieval_cache[t] = result
        return result

    def batch(self, t: int, rng: np.random.Generator | None = None, require_target: bool = True) -> DateBatch | None:
        start = t - self.cfg.lookback + 1
        x = self.features[start:t + 1].transpose(1, 0, 2)
        mask = np.isfinite(x).all(axis=(1, 2))
        if require_target:
            mask &= np.isfinite(self.targets[t]) & np.isfinite(self.targets_raw[t])
        assets = np.flatnonzero(mask)
        if len(assets) < 8:
            return None
        if rng is not None and len(assets) > self.cfg.max_assets_per_date:
            assets = np.sort(rng.choice(assets, self.cfg.max_assets_per_date, replace=False))
        return DateBatch(
            x=x[assets], market=np.repeat(self.market[t][None, :], len(assets), axis=0),
            y=self.targets[t, assets], raw_return=self.targets_raw[t, assets], assets=assets,
            retrieval=self._retrieval(t)[assets], complementarity=self.complementarity[t],
            date_index=int(t), date=str(self.dates[t]),
        )

    def iter_dates(self, split: str, limit: int | None, rng: np.random.Generator | None) -> Iterator[DateBatch]:
        dates = self.split_indices[split].copy()
        if limit and len(dates) > limit:
            if rng is None:
                dates = dates[np.linspace(0, len(dates) - 1, limit).round().astype(int)]
            else:
                dates = np.sort(rng.choice(dates, limit, replace=False))
        for t in dates:
            fitting = split in ("train", "fit")
            batch = self.batch(int(t), rng if fitting else None, require_target=fitting)
            if batch is not None:
                yield batch

    @property
    def input_size(self) -> int:
        return len(FEATURE_NAMES)

    @property
    def retrieval_size(self) -> int:
        return len(FEATURE_NAMES) + 10
