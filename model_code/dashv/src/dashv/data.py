from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config, DatasetConfig


# Variances are in (percent return)^2, i.e. daily squared log returns times 10,000.
EPS = 1e-6
FEATURE_NAMES = (
    "log_own_v1", "log_own_v5", "log_own_v20", "log_own_v60",
    "log_own_down5", "log_own_down20", "log_range5", "log_range20",
    "return1_pct", "return5_pct", "volume_z20", "market_beta60",
    "log_market_v1", "log_market_v5", "log_market_v20",
    "log_market_down5", "log_market_down20",
)


def _mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; every input must be observed, including today's."""
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0).astype(np.float64)
    total = np.concatenate((np.zeros_like(filled[:1]), np.cumsum(filled, axis=0)), axis=0)
    count = np.concatenate((np.zeros_like(finite[:1], dtype=np.int32),
                            np.cumsum(finite, axis=0, dtype=np.int32)), axis=0)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    vals = total[window:] - total[:-window]
    counts = count[window:] - count[:-window]
    out[window - 1:] = np.where(counts == window, vals / window, np.nan)
    return out


def _log_positive(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, EPS)).astype(np.float32)


@dataclass
class Batch:
    date: str
    date_index: int
    assets: np.ndarray
    x: np.ndarray
    seq: np.ndarray
    anchor: np.ndarray
    own: np.ndarray
    market: np.ndarray
    y: np.ndarray
    stress: np.ndarray


class Panel:
    """One public daily-bar panel; origin t uses information through close t only."""

    def __init__(self, root: Path, spec: DatasetConfig, cfg: Config):
        self.spec, self.cfg = spec, cfg
        with np.load(root / spec.file, allow_pickle=False) as pack:
            self.dates = pack["dates"].astype("datetime64[D]")
            self.symbols = pack["symbols"].astype(str)
            values = pack["values"].astype(np.float32)
        self._engineer(values)
        self.split_indices = self._split()
        self._scale_features()

    def _engineer(self, values: np.ndarray) -> None:
        opn, high, low, close, adj, volume = [values[:, :, i] for i in range(6)]
        del opn
        t_count, n_assets = adj.shape
        ret = np.full_like(adj, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret[1:] = np.log(adj[1:] / adj[:-1])
            day_range = np.log(high / low)
            log_volume = np.log1p(volume)
        ret[~np.isfinite(ret)] = np.nan
        day_range[~np.isfinite(day_range)] = np.nan
        rv = ret * ret * 10000.0
        downside = np.where(np.isfinite(ret), np.minimum(ret, 0.0) ** 2 * 10000.0, np.nan)
        range_proxy = day_range * day_range * 10000.0
        v1, v5, v20, v60 = [rv if w == 1 else _mean(rv, w) for w in (1, 5, 20, 60)]
        d5, d20 = _mean(downside, 5), _mean(downside, 20)
        q5, q20 = _mean(range_proxy, 5), _mean(range_proxy, 20)
        ret5 = _mean(ret, 5) * 5.0 * 100.0
        logvol_mean = _mean(log_volume, 20)
        logvol_sq = _mean(log_volume * log_volume, 20)
        volz = (log_volume - logvol_mean) / np.sqrt(np.maximum(logvol_sq - logvol_mean ** 2, EPS))
        # At each date the market proxy uses only same-date available panel returns.
        with np.errstate(invalid="ignore"):
            mret = np.nanmean(ret, axis=1)
        mrv = mret * mret * 10000.0
        mdown = np.minimum(mret, 0.0) ** 2 * 10000.0
        mv1, mv5, mv20 = mrv, _mean(mrv, 5), _mean(mrv, 20)
        md5, md20 = _mean(mdown, 5), _mean(mdown, 20)
        mret_matrix = np.broadcast_to(mret[:, None], ret.shape)
        cov = _mean(ret * mret_matrix, 60) - _mean(ret, 60) * _mean(mret_matrix, 60)
        market_var = _mean(mret_matrix * mret_matrix, 60) - _mean(mret_matrix, 60) ** 2
        beta = np.clip(cov / np.maximum(market_var, EPS / 10000.0), -2.0, 4.0)
        market = [np.broadcast_to(z[:, None], (t_count, n_assets)) for z in (mv1, mv5, mv20, md5, md20)]
        self.feature_raw = np.stack([
            _log_positive(v1), _log_positive(v5), _log_positive(v20), _log_positive(v60),
            _log_positive(d5), _log_positive(d20), _log_positive(q5), _log_positive(q20),
            ret * 100.0, ret5, volz, beta,
            *[_log_positive(z) for z in market],
        ], axis=-1).astype(np.float32)
        self.anchor = np.stack((v1, v5, v20, v60), axis=-1).astype(np.float32)
        self.own = np.stack((d5 / np.maximum(v20, EPS), d20 / np.maximum(v20, EPS),
                             q5 / np.maximum(v20, EPS)), axis=-1).astype(np.float32)
        self.market_state = np.stack((market[3] / np.maximum(market[2], EPS),
                                      market[0] / np.maximum(market[2], EPS),
                                      np.maximum(beta, 0.0),
                                      market[1] / np.maximum(v20, EPS)), axis=-1).astype(np.float32)
        self.stress = (md5 / np.maximum(mv20, EPS)).astype(np.float32)
        self.seq_raw = np.stack((ret * 100.0, np.abs(ret) * 100.0, downside,
                                 np.sqrt(np.maximum(range_proxy, 0.0))), axis=-1).astype(np.float32)
        self.target = np.full_like(adj, np.nan)
        h = self.cfg.horizon
        future = np.stack([rv[k:t_count - h + k] for k in range(1, h + 1)], axis=0)
        self.target[:t_count - h] = np.where(np.isfinite(future).all(axis=0),
                                             np.mean(future, axis=0), np.nan)
        # Feature eligibility is intentionally independent of future target availability.
        valid_seq = np.isfinite(self.seq_raw).all(axis=2).astype(np.int16)
        cs = np.concatenate((np.zeros((1, n_assets), dtype=np.int32),
                             np.cumsum(valid_seq, axis=0, dtype=np.int32)), axis=0)
        self.feature_eligible = np.zeros_like(valid_seq, dtype=bool)
        self.feature_eligible[self.cfg.lookback - 1:] = (
            cs[self.cfg.lookback:] - cs[:-self.cfg.lookback] == self.cfg.lookback)
        self.feature_eligible &= np.isfinite(self.feature_raw).all(axis=2)
        self.feature_eligible &= np.isfinite(self.anchor).all(axis=2)
        self.feature_eligible &= np.isfinite(self.own).all(axis=2)
        self.feature_eligible &= np.isfinite(self.market_state).all(axis=2)

    def _split(self) -> dict[str, np.ndarray]:
        cfg, spec = self.cfg, self.spec
        origins = np.arange(cfg.lookback - 1, len(self.dates) - cfg.horizon)
        origins = origins[self.feature_eligible[origins].sum(axis=1) >= 8]
        end = self.dates[origins + cfg.horizon]
        train_end, valid_end, test_end = map(np.datetime64, (spec.train_end, spec.valid_end, spec.test_end))
        train = origins[end <= train_end]
        valid = origins[(self.dates[origins] > train_end) & (end <= valid_end)]
        test = origins[(self.dates[origins] > valid_end) & (end <= test_end)]
        if len(train) < 100 or len(valid) < 20 or len(test) < 20:
            raise ValueError(f"{spec.name}: insufficient chronology for registered splits")
        tune_start = train[int(len(train) * 0.85)]
        fit = train[train + cfg.horizon < tune_start]
        core_valid = train[train >= tune_start]
        if len(fit) < 80 or len(core_valid) < 10:
            raise ValueError(f"{spec.name}: insufficient fit/checkpoint split")
        return {"fit": fit, "core_valid": core_valid, "valid": valid, "test": test}

    def _scale_features(self) -> None:
        train = self.feature_raw[self.split_indices["fit"]]
        mean = np.nanmean(train, axis=(0, 1))
        std = np.nanstd(train, axis=(0, 1))
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self.x = np.clip((self.feature_raw - mean) / self.feature_std, -8.0, 8.0).astype(np.float32)
        self.seq = np.clip(self.seq_raw, -10.0, 10.0).astype(np.float32)

    def batch(self, t: int, rng: np.random.Generator | None = None) -> Batch | None:
        assets = np.flatnonzero(self.feature_eligible[t])
        if len(assets) < 8:
            return None
        if rng is not None and len(assets) > self.cfg.max_assets_per_date:
            assets = np.sort(rng.choice(assets, self.cfg.max_assets_per_date, replace=False))
        start = t - self.cfg.lookback + 1
        return Batch(str(self.dates[t]), int(t), assets, self.x[t, assets],
                     self.seq[start:t + 1][:, assets, :].transpose(1, 0, 2),
                     self.anchor[t, assets], np.clip(self.own[t, assets], 0.0, 20.0),
                     np.clip(self.market_state[t, assets], 0.0, 20.0),
                     self.target[t, assets], np.repeat(self.stress[t], len(assets)))

    def iter_dates(self, split: str, limit: int | None = None,
                   rng: np.random.Generator | None = None) -> Iterator[Batch]:
        dates = self.split_indices[split]
        if limit is not None and len(dates) > limit:
            if rng is None:
                dates = dates[np.linspace(0, len(dates) - 1, limit).round().astype(int)]
            else:
                dates = np.sort(rng.choice(dates, limit, replace=False))
        for t in dates:
            batch = self.batch(int(t), rng if split == "fit" else None)
            if batch is not None:
                yield batch
