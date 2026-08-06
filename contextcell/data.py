from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    name: str
    adata: AnnData
    label_key: str


def _clean_labels(adata: AnnData, label_key: str) -> AnnData:
    labels = adata.obs[label_key].astype(str)
    valid = (~labels.isna()) & (labels != "nan") & (labels != "Unknown")
    out = adata[valid].copy()
    out.obs[label_key] = out.obs[label_key].astype(str).astype("category")
    out.var_names_make_unique()
    return out


def load_dataset(name: str) -> DatasetBundle:
    if name == "pbmc3k":
        adata = sc.datasets.pbmc3k_processed().copy()
        label_key = "louvain"
    elif name == "paul15":
        adata = sc.datasets.paul15().copy()
        label_key = "paul15_clusters"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return DatasetBundle(name=name, adata=_clean_labels(adata, label_key), label_key=label_key)


def filter_and_cap_classes(
    adata: AnnData,
    label_key: str,
    min_class_cells: int,
    max_cells_per_class: int,
    seed: int,
) -> AnnData:
    rng = np.random.default_rng(seed)
    labels = adata.obs[label_key].astype(str).to_numpy()
    keep: list[int] = []
    for cls in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == cls)
        if idx.size < min_class_cells:
            continue
        if idx.size > max_cells_per_class:
            idx = rng.choice(idx, size=max_cells_per_class, replace=False)
        keep.extend(idx.tolist())
    if not keep:
        raise ValueError("No classes remain after filtering.")
    return adata[np.sort(np.asarray(keep, dtype=int))].copy()


def preprocess_features(adata: AnnData, n_hvg: int, n_pcs: int, seed: int) -> AnnData:
    out = adata.copy()
    # Scanpy's PBMC3K processed object is already scaled and can contain negative values;
    # Paul15 is count-like. Apply HVG selection only to nonnegative expression data.
    x_max = float(out.X.max())
    x_min = float(out.X.min())
    if x_min >= 0:
        if x_max > 50:
            sc.pp.normalize_total(out, target_sum=1e4)
            sc.pp.log1p(out)
        n_hvg_eff = min(n_hvg, out.n_vars)
        if n_hvg_eff < out.n_vars:
            sc.pp.highly_variable_genes(out, n_top_genes=n_hvg_eff, flavor="seurat", subset=True)
        sc.pp.scale(out, max_value=10)
    else:
        # Already transformed/scaled: retain the supplied features and only recompute PCA
        # after class filtering/subsampling.
        if out.n_vars > n_hvg:
            variances = np.asarray(out.X.var(axis=0)).ravel()
            top = np.argsort(variances)[-n_hvg:]
            out = out[:, np.sort(top)].copy()
    n_pcs_eff = min(n_pcs, out.n_vars - 1, out.n_obs - 1)
    sc.tl.pca(out, n_comps=max(2, n_pcs_eff), svd_solver="arpack", random_state=seed)
    return out


def make_split(
    adata: AnnData,
    label_key: str,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    y = adata.obs[label_key].astype(str).to_numpy()
    indices = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], test_idx
