from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextcell.experiment import run_experiment
from contextcell.utils import load_yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_suite(config_path: str, only: list[str], fresh: bool) -> None:
    suite = load_yaml(config_path)
    selected = set(only)
    for variant in suite["variants"]:
        name = str(variant["name"])
        if selected and name not in selected:
            continue
        cfg = deep_update(suite["base"], variant.get("overrides", {}))
        cfg["experiment_name"] = name
        cfg["run_label"] = name
        cfg["output_dir"] = str(Path(suite["output_root"]) / name)
        cfg.setdefault("runtime", {})["resume"] = not fresh
        print(f"\n=== Running publication ablation: {name} ===")
        run_experiment(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/publication_ablations.yaml")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_suite(args.config, args.only, args.fresh)


if __name__ == "__main__":
    main()
