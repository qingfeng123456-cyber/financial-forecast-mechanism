from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from c3rmoe.config import ExperimentConfig
from c3rmoe.pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed C3R-MoE AutoDL research pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    archive, completed = run(ROOT, cfg, args.device)
    print(f"Finished. Download this file: {archive}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
