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
    protocol_id: str = "dashv-v1.0-memory-coupled-shrinkage"
    seed: int = 2026
    horizon: int = 5
    lookback: int = 60
    epochs: int = 40
    patience: int = 9
    dates_per_epoch: int = 320
    valid_dates_per_epoch: int = 160
    max_assets_per_date: int = 320
    learning_rate: float = 0.0004
    weight_decay: float = 0.0001
    hidden_size: int = 48
    bootstrap_draws: int = 5000
    bootstrap_block_length: int = 20
    progress_updates_per_epoch: int = 12
    aggregation_temperature: float = 2.0
    meta_temperature: float = 2.0
    aggregation_loss_clip: float = 4.0
    memory_decays: list[float] = field(default_factory=lambda: [0.90, 0.97, 0.995])
    single_memory_decay: float = 0.97
    datasets: list[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig("sp500_cc0", "data/sp500_cc0_panel.npz", "2018-12-31", "2020-12-31", "2024-12-20"),
        DatasetConfig("kdd17", "data/kdd17_panel.npz", "2014-12-31", "2015-12-31", "2016-12-30"),
        DatasetConfig("acl18", "data/acl18_panel.npz", "2015-12-31", "2016-06-30", "2017-08-25"),
    ])
    deterministic_controls: list[str] = field(default_factory=lambda: [
        "persistence", "ewma", "har", "shar", "market_shar",
    ])
    neural_controls: list[str] = field(default_factory=lambda: [
        "flat_mlp", "gru", "lstm", "tcn", "patch_transformer",
    ])
    aggregation_controls: list[str] = field(default_factory=lambda: [
        "equal_geometric", "equal_arithmetic", "static_qlike", "asset_local_single",
    ])
    factorial_cells: list[str] = field(default_factory=lambda: [
        "dashv_base", "dashv_hier", "dashv_multi", "dashv_full",
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
            raise ValueError("The registered development matrix requires three panels and four factorial cells")
        if len(cfg.memory_decays) != 3 or not all(0.0 < x < 1.0 for x in cfg.memory_decays):
            raise ValueError("Exactly three valid memory decays are required")
        if not 0.0 < cfg.single_memory_decay < 1.0:
            raise ValueError("Invalid single-memory decay")
        if min(cfg.epochs, cfg.dates_per_epoch, cfg.valid_dates_per_epoch,
               cfg.bootstrap_draws, cfg.bootstrap_block_length) <= 0:
            raise ValueError("Invalid nonpositive setting")
        return cfg
