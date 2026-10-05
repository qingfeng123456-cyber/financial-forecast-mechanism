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
    protocol_id: str = "darc-v1.0-delayed-advantage-rank-council"
    seed: int = 2026
    lookback: int = 20
    horizon: int = 5
    hidden_size: int = 64
    num_experts: int = 4
    epochs: int = 18
    patience: int = 5
    dates_per_epoch: int = 128
    valid_dates_per_epoch: int = 64
    max_assets_per_date: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    rank_weight: float = 0.35
    balance_weight: float = 0.002
    router_fraction: float = 0.20
    calibration_fraction: float = 0.25
    competence_history_dates: int = 126
    advantage_history_dates: int = 126
    evidence_min_dates: int = 20
    pair_temperature: float = 0.15
    competence_temperature: float = 0.05
    advantage_standard_error_floor: float = 0.02
    bootstrap_block_length: int = 20
    bootstrap_draws: int = 5000
    transaction_cost_bps: float = 10.0
    progress_updates_per_epoch: int = 10
    datasets: list[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig("sp500_cc0", "data/sp500_cc0_panel.npz", "2018-12-31", "2020-12-31", "2024-12-20"),
        DatasetConfig("kdd17", "data/kdd17_panel.npz", "2014-12-31", "2015-12-31", "2016-12-30"),
        DatasetConfig("acl18", "data/acl18_panel.npz", "2015-12-31", "2016-06-30", "2017-08-25"),
    ])
    baseline_models: list[str] = field(default_factory=lambda: [
        "gru", "stockmixer", "master", "miga", "mera", "mafs"
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
        if not 0 < cfg.router_fraction < 0.5 or not 0 < cfg.calibration_fraction < 0.5:
            raise ValueError("Invalid chronological calibration fractions")
        if min(cfg.epochs, cfg.dates_per_epoch, cfg.valid_dates_per_epoch, cfg.evidence_min_dates,
               cfg.competence_history_dates, cfg.advantage_history_dates, cfg.pair_temperature,
               cfg.competence_temperature, cfg.advantage_standard_error_floor) <= 0:
            raise ValueError("Invalid positive fixed setting")
        return cfg
