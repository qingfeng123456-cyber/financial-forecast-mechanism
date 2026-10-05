from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    file: str
    train_end: str
    valid_end: str
    test_end: str


@dataclass
class ExperimentConfig:
    protocol_id: str = "care-v2.0-conditional-analog-risk-experts"
    seed: int = 2026
    lookback: int = 20
    horizon: int = 5
    hidden_size: int = 64
    num_experts: int = 4
    epochs: int = 36
    patience: int = 8
    dates_per_epoch: int = 256
    valid_dates_per_epoch: int = 128
    max_assets_per_date: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    balance_weight: float = 0.002
    retrieval_window_dates: int = 252
    retrieval_neighbors: int = 16
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    bootstrap_block_length: int = 20
    bootstrap_draws: int = 5000
    progress_updates_per_epoch: int = 10
    datasets: list[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig("sp500_cc0", "data/sp500_cc0_panel.npz", "2018-12-31", "2020-12-31", "2024-12-20"),
        DatasetConfig("kdd17", "data/kdd17_panel.npz", "2014-12-31", "2015-12-31", "2016-12-30"),
        DatasetConfig("acl18", "data/acl18_panel.npz", "2015-12-31", "2016-06-30", "2017-08-25"),
    ])
    baseline_models: list[str] = field(default_factory=lambda: [
        "historical", "market_analog", "gru_quantile", "stockmixer_quantile",
        "master_quantile", "mera_quantile"
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["datasets"] = [DatasetConfig(**item) for item in raw["datasets"]]
        cfg = cls(**raw)
        if cfg.seed != 2026 or cfg.num_experts != 4:
            raise ValueError("The registered configuration requires four internal experts")
        if len(cfg.datasets) != 3 or len(cfg.baseline_models) != 6 or len(set(cfg.baseline_models)) != 6:
            raise ValueError("The registered matrix requires three panels and six forecasters")
        if cfg.quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("The registered forecast uses three fixed quantiles")
        if min(cfg.epochs, cfg.dates_per_epoch, cfg.valid_dates_per_epoch,
               cfg.retrieval_window_dates, cfg.retrieval_neighbors) <= 0:
            raise ValueError("Invalid positive fixed setting")
        return cfg
