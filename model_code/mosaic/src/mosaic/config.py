from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    file: str
    train_end: str
    valid_end: str
    test_end: str


@dataclass
class Config:
    protocol_id: str = "mosaic-v1.0-daily-downside-interaction"
    seed: int = 2026
    horizon: int = 5
    lookback: int = 60
    epochs: int = 36
    patience: int = 8
    dates_per_epoch: int = 256
    valid_dates_per_epoch: int = 128
    max_assets_per_date: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    hidden_size: int = 32
    bootstrap_draws: int = 5000
    bootstrap_block_length: int = 20
    progress_updates_per_epoch: int = 10
    datasets: list[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig("sp500_cc0", "data/sp500_cc0_panel.npz", "2018-12-31", "2020-12-31", "2024-12-20"),
        DatasetConfig("kdd17", "data/kdd17_panel.npz", "2014-12-31", "2015-12-31", "2016-12-30"),
        DatasetConfig("acl18", "data/acl18_panel.npz", "2015-12-31", "2016-06-30", "2017-08-25"),
    ])
    deterministic_controls: list[str] = field(default_factory=lambda: [
        "persistence", "ewma", "har", "shar", "market_shar", "market_shar_interaction",
    ])
    neural_controls: list[str] = field(default_factory=lambda: ["flat_mlp", "gru", "rg_resmoe"])
    factorial_cells: list[str] = field(default_factory=lambda: [
        "mosaic_base", "mosaic_own", "mosaic_market", "mosaic_full",
    ])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["datasets"] = [DatasetConfig(**row) for row in raw["datasets"]]
        cfg = cls(**raw)
        if cfg.seed != 2026 or cfg.horizon != 5 or cfg.lookback != 60:
            raise ValueError("This package uses its registered 5-date/60-date protocol")
        if len(cfg.datasets) != 3 or len(cfg.factorial_cells) != 4:
            raise ValueError("The registered matrix requires three panels and four factorial cells")
        if min(cfg.epochs, cfg.dates_per_epoch, cfg.valid_dates_per_epoch,
               cfg.bootstrap_draws, cfg.bootstrap_block_length) <= 0:
            raise ValueError("Invalid nonpositive setting")
        return cfg
