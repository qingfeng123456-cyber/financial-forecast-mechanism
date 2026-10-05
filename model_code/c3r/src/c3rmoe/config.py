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
    protocol_id: str = "c3r-moe-v2.0-innovation-coalitions"
    seed: int = 2026
    lookback: int = 20
    horizon: int = 5
    hidden_size: int = 64
    router_hidden_size: int = 16
    num_experts: int = 4
    graph_topk: int = 2
    epochs: int = 24
    patience: int = 7
    dates_per_epoch: int = 192
    valid_dates_per_epoch: int = 64
    max_assets_per_date: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    rank_weight: float = 0.35
    credit_weight: float = 0.50
    balance_weight: float = 0.002
    expert_weight: float = 0.25
    confidence_weight: float = 0.03
    message_weight: float = 0.002
    coalition_weight: float = 0.15
    router_shrinkage: float = 0.10
    router_fraction: float = 0.20
    router_epochs: int = 30
    router_learning_rate: float = 0.001
    router_cache_dates: int = 192
    router_temperature: float = 0.7
    risk_scale_floor: float = 0.02
    bootstrap_block_length: int = 20
    transaction_cost_bps: float = 10.0
    bootstrap_draws: int = 5000
    tsne_points: int = 1800
    progress_updates_per_epoch: int = 10
    datasets: list[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig("sp500_cc0", "data/sp500_cc0_panel.npz", "2018-12-31", "2020-12-31", "2024-12-20"),
        DatasetConfig("kdd17", "data/kdd17_panel.npz", "2014-12-31", "2015-12-31", "2016-12-30"),
        DatasetConfig("acl18", "data/acl18_panel.npz", "2015-12-31", "2016-06-30", "2017-08-25"),
    ])
    baseline_models: list[str] = field(default_factory=lambda: [
        "gru", "stockmixer", "master", "miga", "mera", "mafs"
    ])
    ablation_models: list[str] = field(default_factory=lambda: [
        "c3r_full", "c3r_graph_only", "c3r_credit_only", "c3r_plain_moe"
    ])
    mechanism_models: list[str] = field(default_factory=lambda: ["c3r_unprojected"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["datasets"] = [DatasetConfig(**item) for item in raw.get("datasets", [])]
        config=cls(**raw)
        if config.seed!=2026 or config.num_experts!=4:
            raise ValueError("This registered protocol requires its fixed random configuration and four experts")
        if len(config.datasets)!=3 or len(config.baseline_models)!=6 or len(config.ablation_models)!=4 or config.mechanism_models!=["c3r_unprojected"]:
            raise ValueError("The frozen protocol requires three panels, six baselines, four factorial cells and the unprojected control")
        models=[*config.baseline_models,*config.ablation_models,*config.mechanism_models]
        if len(set(models))!=len(models):raise ValueError("Duplicate model in the registered matrix")
        if not 0<config.router_fraction<0.5 or not 0<=config.router_shrinkage<1 or config.router_temperature<=0:
            raise ValueError("Invalid staged routing configuration")
        if min(config.lookback,config.horizon,config.epochs,config.router_epochs,config.dates_per_epoch,config.progress_updates_per_epoch)<=0:
            raise ValueError("Training lengths and progress frequency must be positive")
        return config
