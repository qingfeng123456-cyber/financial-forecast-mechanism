from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dashv.config import Config  # noqa: E402
from dashv.pipeline import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DASH-V AutoDL experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    archive, complete = run(ROOT, Config.from_json(args.config), args.device)
    print(f"Finished. Download this file: {archive}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
