from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextcell.experiment import run_experiment
from contextcell.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fresh", action="store_true", help="Ignore completed target checkpoints and rerun all work.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg.setdefault("runtime", {})["resume"] = not args.fresh
    out = run_experiment(cfg)
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
