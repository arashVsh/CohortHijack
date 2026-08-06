from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_seed(*parts: object, modulo: int = 2**32 - 1) -> int:
    """Create a deterministic seed independent of Python's randomized hash()."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % modulo


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return cfg


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(payload: Any, path: str | Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2, sort_keys=True), path)


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Atomically write a CSV checkpoint on Windows, macOS, and Linux.

    The file must be opened for writing while flush/fsync is performed. On
    Windows, calling os.fsync() on a separately reopened read-only descriptor
    can raise ``OSError: [Errno 9] Bad file descriptor``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # The atomic rename still prevents a partial checkpoint from
            # replacing the last valid file on platforms/filesystems where
            # fsync is unavailable for this descriptor.
            pass

    os.replace(tmp, path)


def save_environment(path: str | Path) -> None:
    import anndata
    import joblib
    import matplotlib
    import pandas
    import scanpy
    import scipy
    import sklearn

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scanpy": scanpy.__version__,
        "anndata": anndata.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
    atomic_write_json(payload, path)
