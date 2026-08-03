"""Source-agnostic core for building eosbench datasets.

Shared by the ``prepare_*.py`` scripts (MoleculeNet, tdcommons, Polaris). Given a
table of SMILES + binary labels it produces the full on-disk layout eosbench expects:

    data/{source}/{task}/{name}/
        data.csv          # smiles, activity
        folds.csv         # random_fold, scaffold_split
        morgan.npy        # (n, 2048) int64
        rdkit.npy         # (n, 217)  float64
        metadata.json
    src/eosbench/_data/{source}/{task}/{name}/metadata.json   # bundled copy

Requires the optional per-source ``prepare-*`` extras, e.g. ``pip install -e
".[prepare-moleculenet]"`` (or ``prepare-tdcommons`` / ``prepare-polaris``).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

RDLogger.DisableLog("rdApp.*")

# Repo roots, resolved relative to this file (scripts/_prepare_common.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
PKG_DATA_ROOT = REPO_ROOT / "src" / "eosbench" / "_data"

MORGAN_NBITS = 2048
MORGAN_RADIUS = 2
# RDKit descriptor list, truncated to 217 features to match the existing datasets'
# rdkit.npy dimensionality (see FEATURIZATIONS in src/eosbench/dataset.py).
RDKIT_DIM = 217
_RDKIT_DESC = [name for name, _ in Descriptors.descList][:RDKIT_DIM]


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def coerce_binary(series: pd.Series) -> pd.Series | None:
    """Coerce a label column to {0.0, 1.0} floats with NaN preserved.

    Returns None if non-missing values aren't binary, or fewer than two classes appear.
    Shared by the ``prepare_*.py`` scripts so every source validates labels identically.
    """
    num = pd.to_numeric(series, errors="coerce")
    present = num.dropna().round()
    if present.empty or set(present.unique()) - {0.0, 1.0} or present.nunique() < 2:
        return None
    return num.round()


def coerce_continuous(series: pd.Series) -> pd.Series | None:
    """Coerce a label column to floats with NaN preserved, for regression targets.

    Sibling of :func:`coerce_binary`. Returns None only if the column has no usable signal
    (all-missing, or a single constant value with zero variance).
    """
    num = pd.to_numeric(series, errors="coerce")
    present = num.dropna()
    if present.empty or present.nunique() < 2:
        return None
    return num


def parse_molecules(smiles: list[str]) -> tuple[list[str], list[Chem.Mol], np.ndarray]:
    """Canonicalize SMILES and drop unparseable rows.

    Returns ``(canonical_smiles, mols, keep_mask)`` where ``keep_mask`` is a boolean
    array over the input aligned to the originals, so labels/splits can be filtered the
    same way.
    """
    canon, mols, keep = [], [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            keep.append(False)
            continue
        canon.append(Chem.MolToSmiles(mol))
        mols.append(mol)
        keep.append(True)
    return canon, mols, np.asarray(keep, dtype=bool)


# --------------------------------------------------------------------------------------
# Featurization
# --------------------------------------------------------------------------------------
def featurize_morgan(mols: list[Chem.Mol]) -> np.ndarray:
    """Morgan count fingerprints, shape (n, 2048) int64."""
    out = np.zeros((len(mols), MORGAN_NBITS), dtype=np.int64)
    for i, mol in enumerate(mols):
        try:
            fp = AllChem.GetHashedMorganFingerprint(
                mol, MORGAN_RADIUS, nBits=MORGAN_NBITS
            )
            arr = np.zeros((MORGAN_NBITS,), dtype=np.int64)
            ConvertToNumpyArray(fp, arr)
            out[i] = arr
        except Exception:  # noqa: BLE001 - leave a zero vector for the odd unfeaturizable mol
            pass
    return out


def featurize_rdkit(mols: list[Chem.Mol]) -> np.ndarray:
    """RDKit physicochemical descriptors, shape (n, 217) float64."""
    out = np.zeros((len(mols), len(_RDKIT_DESC)), dtype=np.float64)
    funcs = {name: fn for name, fn in Descriptors.descList if name in set(_RDKIT_DESC)}
    for i, mol in enumerate(mols):
        for j, name in enumerate(_RDKIT_DESC):
            try:
                out[i, j] = funcs[name](mol)
            except Exception:
                out[i, j] = np.nan
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


FEATURIZERS = {"morgan": featurize_morgan, "rdkit": featurize_rdkit}


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def make_random_folds(
    n: int, k: int = 5, seed: int = 42, stratify: np.ndarray | None = None
) -> np.ndarray:
    """Deterministic random K-fold assignment, shape (n,) of ints in [0, k).

    When ``stratify`` (a label array) is given, uses ``StratifiedKFold`` so every fold
    contains both classes. This is only possible for a single-task family — a conserved
    multi-task split has no single label to stratify on, so it passes ``stratify=None``
    and falls back to a plain shuffled ``KFold``.
    """
    folds = np.empty(n, dtype=int)
    if stratify is not None:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(n), np.asarray(stratify))
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(n))
    for i, (_, test_idx) in enumerate(iterator):
        folds[test_idx] = i
    return folds


def _murcko_groups(mols: list[Chem.Mol]) -> list[list[int]]:
    """Group molecule indices by Bemis-Murcko scaffold (deterministic order)."""
    scaffolds: dict[str, list[int]] = {}
    for i, mol in enumerate(mols):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:  # noqa: BLE001 - RDKit can raise on odd structures; treat as own scaffold
            scaffold = f"__unscaffolded_{i}"
        scaffolds.setdefault(scaffold, []).append(i)
    # Size desc, then first-index asc — a stable, seed-free order.
    return sorted(scaffolds.values(), key=lambda idxs: (-len(idxs), idxs[0]))


def scaffold_split(
    mols: list[Chem.Mol], y: np.ndarray | None = None, frac_train: float = 0.9
) -> np.ndarray:
    """Deterministic Bemis-Murcko scaffold split.

    Molecules are grouped by Murcko scaffold (so a scaffold never spans train and test).
    ``valid`` is merged into ``train`` (frac_train=0.9 = canonical 0.8 train + 0.1 valid).
    Returns an array of ``"train"``/``"test"`` labels, shape (n,).

    With ``y`` given (single-task family), the split is **class-stratified**: each scaffold
    group is binned by its majority label and the size-descending train/test fill runs
    independently per bin, guaranteeing both classes in train and test — avoiding the
    degenerate single-class test set a naive split produces on skewed sets like BBBP.

    With ``y=None`` (conserved multi-task split, where no single label exists), it falls
    back to a plain size-descending fill.
    """
    groups = _murcko_groups(mols)

    if y is None:
        n = len(mols)
        n_train = int(frac_train * n)
        if len(groups) > 1:
            n_train = min(n_train, n - min(len(g) for g in groups))
        labels = np.empty(n, dtype=object)
        train_count = 0
        for group in groups:
            if train_count + len(group) <= n_train:
                labels[group] = "train"
                train_count += len(group)
            else:
                labels[group] = "test"
        return labels

    y = np.asarray(y)

    # Bin each scaffold group by its majority label (ties -> positive).
    bins: dict[int, list[list[int]]] = {0: [], 1: []}
    for group in groups:
        majority = int(round(float(y[group].mean()) + 1e-9))
        bins[majority].append(group)

    labels = np.empty(len(mols), dtype=object)
    for cls, cls_groups in bins.items():
        bin_total = sum(len(g) for g in cls_groups)
        if bin_total == 0:
            continue
        n_train = int(frac_train * bin_total)
        # Reserve at least the smallest group for test when the bin has >1 group, so
        # neither side of a class is ever empty.
        if len(cls_groups) > 1:
            smallest = min(len(g) for g in cls_groups)
            n_train = min(n_train, bin_total - smallest)
        train_count = 0
        for group in cls_groups:  # already size-desc within the original ordering
            if train_count + len(group) <= n_train:
                labels[group] = "train"
                train_count += len(group)
            else:
                labels[group] = "test"

    # Safety guard: if a class still has no test sample (e.g. it lived in a single
    # scaffold group), move that class's smallest group from train into test.
    _ensure_class_in_test(labels, y, bins)
    return labels


def _ensure_class_in_test(labels: np.ndarray, y: np.ndarray, bins: dict) -> None:
    test_mask = labels == "test"
    for cls, cls_groups in bins.items():
        in_test = test_mask & (y == cls)
        present_classes = {int(c) for c in y if (y == c).any()}
        if cls not in present_classes or in_test.any():
            continue
        movable = [g for g in cls_groups if all(labels[i] == "train" for i in g)]
        if len(movable) > 1:  # keep at least one group in train for this class
            smallest = min(movable, key=len)
            labels[smallest] = "test"
            test_mask = labels == "test"


# --------------------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------------------
def _rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


def rf_baseline_cv(
    X: np.ndarray, y: np.ndarray, random_fold: np.ndarray, seed: int = 42
):
    """RandomForest baseline over random K-fold CV. Returns AUROC/AUPR mean & std."""
    aurocs, auprs = [], []
    for k in sorted(set(random_fold.tolist())):
        tr, te = random_fold != k, random_fold == k
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        clf = _rf(seed).fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], proba))
        auprs.append(average_precision_score(y[te], proba))
    if not aurocs:
        return {
            "auroc_mean": None,
            "auroc_std": None,
            "aupr_mean": None,
            "aupr_std": None,
        }
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "aupr_mean": float(np.mean(auprs)),
        "aupr_std": float(np.std(auprs)),
    }


def rf_baseline_holdout(
    X: np.ndarray, y: np.ndarray, scaffold: np.ndarray, seed: int = 42
):
    """RandomForest baseline on the scaffold holdout (fit on train, score on test)."""
    tr, te = scaffold == "train", scaffold == "test"
    if (
        tr.sum() == 0
        or te.sum() == 0
        or len(set(y[tr].tolist())) < 2
        or len(set(y[te].tolist())) < 2
    ):
        return {"auroc": None, "aupr": None}
    clf = _rf(seed).fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])[:, 1]
    return {
        "auroc": float(roc_auc_score(y[te], proba)),
        "aupr": float(average_precision_score(y[te], proba)),
    }


def _rfr(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)


def rf_regression_cv(X: np.ndarray, y: np.ndarray, random_fold: np.ndarray, seed: int = 42):
    """RandomForest regression baseline over random K-fold CV. Returns RMSE/R2 mean & std."""
    rmses, r2s = [], []
    for k in sorted(set(random_fold.tolist())):
        tr, te = random_fold != k, random_fold == k
        if tr.sum() < 2 or te.sum() < 2:  # R2 is undefined for <2 test points
            continue
        reg = _rfr(seed).fit(X[tr], y[tr])
        pred = reg.predict(X[te])
        rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
        r2s.append(float(r2_score(y[te], pred)))
    if not rmses:
        return {"rmse_mean": None, "rmse_std": None, "r2_mean": None, "r2_std": None}
    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
    }


def rf_regression_holdout(X: np.ndarray, y: np.ndarray, scaffold: np.ndarray, seed: int = 42):
    """RandomForest regression baseline on the scaffold holdout (fit train, score test)."""
    tr, te = scaffold == "train", scaffold == "test"
    if tr.sum() < 2 or te.sum() < 2:
        return {"rmse": None, "r2": None}
    reg = _rfr(seed).fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    return {
        "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
        "r2": float(r2_score(y[te], pred)),
    }


# --------------------------------------------------------------------------------------
# Metadata + writing
# --------------------------------------------------------------------------------------
def today_iso() -> str:
    return _dt.date.today().isoformat()


_NO_METRICS_CV = {
    "auroc_mean": None,
    "auroc_std": None,
    "aupr_mean": None,
    "aupr_std": None,
}
_NO_METRICS_HOLD = {"auroc": None, "aupr": None}
_NO_METRICS_CV_REG = {
    "rmse_mean": None,
    "rmse_std": None,
    "r2_mean": None,
    "r2_std": None,
}
_NO_METRICS_HOLD_REG = {"rmse": None, "r2": None}


def _leaderboard_fields(leaderboard: dict | None) -> dict:
    lb = leaderboard or {}
    return {
        "leaderboard_metric": lb.get("metric"),
        "leaderboard_value": lb.get("value"),
        "leaderboard_split": lb.get("split"),
        "leaderboard_provider": lb.get("provider"),
        "leaderboard_source": lb.get("source"),
    }


def _skewness(y) -> float | None:
    """Fisher-Pearson skewness of the target values (regression label-shape cue).

    Signed: negative = left-tailed, positive = right-tailed, ~0 = symmetric.
    Returns None for fewer than 3 points or a constant target.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return None
    d = y - y.mean()
    m2 = float((d ** 2).mean())
    if m2 == 0.0:
        return 0.0
    return float((d ** 3).mean() / m2 ** 1.5)


def _build_task_block(
    Xb: np.ndarray,
    y: np.ndarray,
    random_fold: np.ndarray,
    scaffold: np.ndarray,
    scaffold_method: str,
    leaderboard: dict | None,
    seed: int,
    compute_baseline: bool = True,
    task: str = "classification",
) -> dict:
    """Compute the per-task metadata block (baseline metrics + counts).

    Classification records AUROC/AUPR (random CV and the scaffold holdout) plus class counts;
    regression records RMSE/R2 instead, with no class balance. With ``compute_baseline=False``
    the metrics are recorded as ``None`` (fast path for very large families like ToxCast/QM9).
    """
    if task == "regression":
        cv = (
            rf_regression_cv(Xb, y, random_fold, seed=seed)
            if compute_baseline
            else _NO_METRICS_CV_REG
        )
        hold = (
            rf_regression_holdout(Xb, y, scaffold, seed=seed)
            if compute_baseline
            else _NO_METRICS_HOLD_REG
        )
        return {
            "n_samples": int(len(y)),
            "skewness": _skewness(y),
            "random_rmse_mean": cv["rmse_mean"],
            "random_rmse_std": cv["rmse_std"],
            "random_r2_mean": cv["r2_mean"],
            "random_r2_std": cv["r2_std"],
            "scaffold_rmse": hold["rmse"],
            "scaffold_r2": hold["r2"],
            "scaffold_split_method": scaffold_method,
            **_leaderboard_fields(leaderboard),
        }

    cv = (
        rf_baseline_cv(Xb, y, random_fold, seed=seed)
        if compute_baseline
        else _NO_METRICS_CV
    )
    hold = (
        rf_baseline_holdout(Xb, y, scaffold, seed=seed)
        if compute_baseline
        else _NO_METRICS_HOLD
    )
    test_mask = scaffold == "test"
    n = int(len(y))
    n_pos = int((y == 1).sum())
    return {
        "n_samples": n,
        "n_positives": n_pos,
        "n_negatives": n - n_pos,
        "random_auroc_mean": cv["auroc_mean"],
        "random_auroc_std": cv["auroc_std"],
        "random_aupr_mean": cv["aupr_mean"],
        "random_aupr_std": cv["aupr_std"],
        "scaffold_auroc": hold["auroc"],
        "scaffold_aupr": hold["aupr"],
        "scaffold_split_method": scaffold_method,
        "scaffold_test_positives": int(((y == 1) & test_mask).sum()),
        "scaffold_test_negatives": int(((y == 0) & test_mask).sum()),
        **_leaderboard_fields(leaderboard),
    }


def _mean_ignore_none(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def prepare_family(
    *,
    source: str,
    family: str,
    smiles: list[str],
    label_df: pd.DataFrame,
    task: str = "classification",
    featurizers: tuple[str, ...] = ("morgan", "rdkit"),
    n_folds: int = 5,
    seed: int = 42,
    leaderboard: dict | None = None,
    compute_baseline: bool = True,
    holdout: np.ndarray | None = None,
    holdout_method: str = "scaffold",
) -> Path:
    """Build and write one dataset *family* (≥1 binary label columns over shared molecules).

    ``label_df`` has one column per endpoint (aligned to ``smiles``), with NaN where a
    column is unlabeled. The molecule set, features, and the train/test split are shared
    across all columns (the split is **conserved**); each column is the split restricted to
    its labeled rows. The split is class-stratified only when the family has a single column.
    ``task`` is the ML task type ("classification"/"regression").

    By default the holdout is a Bemis-Murcko scaffold split computed here. When ``holdout`` is
    given (an array of ``"train"``/``"test"`` aligned to the original ``smiles``, *before* the
    unparseable-SMILES filter), it is used verbatim instead — this lets a source ship its own
    split (e.g. Polaris's official train/test split). ``holdout_method`` is the label recorded
    as ``scaffold_split_method`` in the metadata (e.g. ``"polaris"``).

    Returns the path to the written ``data/`` dataset directory.
    """
    canon, mols, keep = parse_molecules(smiles)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  [{family}] dropped {dropped} unparseable SMILES")
    label_df = label_df.iloc[np.where(keep)[0]].reset_index(drop=True)
    n_mol = len(mols)
    if n_mol == 0:
        raise ValueError(f"[{family}] no parseable molecules")

    columns = list(label_df.columns)
    single = len(columns) == 1

    # Conserved split over all family molecules; class-stratify only for a single-column,
    # fully-labeled *classification* family (regression has no classes to balance).
    stratify = None
    if task == "classification" and single and label_df[columns[0]].notna().all():
        stratify = label_df[columns[0]].to_numpy().astype(int)
    random_fold = make_random_folds(n_mol, k=n_folds, seed=seed, stratify=stratify)
    if holdout is not None:
        # Source-provided split (e.g. Polaris): filter to parseable rows, use verbatim.
        scaffold = np.asarray(holdout, dtype=object)[keep]
        scaffold_method = holdout_method
    else:
        scaffold = scaffold_split(
            mols, stratify
        )  # stratify (or None) drives class-balancing
        scaffold_method = "stratified-murcko" if stratify is not None else "murcko"

    features = {feat: FEATURIZERS[feat](mols) for feat in featurizers}
    base_feat = "morgan" if "morgan" in features else featurizers[0]
    Xb_all = features[base_feat]

    column_blocks: dict[str, dict] = {}
    for col in columns:
        mask = label_df[col].notna().to_numpy()
        y_c = label_df.loc[mask, col].to_numpy()
        y_c = y_c.astype(int) if task == "classification" else y_c.astype(float)
        column_blocks[col] = _build_task_block(
            Xb_all[mask],
            y_c,
            random_fold[mask],
            scaffold[mask],
            scaffold_method,
            leaderboard,
            seed,
            compute_baseline,
            task=task,
        )

    # Family aggregates (mean over columns) consumed by the collapsed catalog row; the keys
    # are task-specific (rmse_mean/r2_mean for regression, auroc_mean/aupr_mean otherwise).
    if task == "regression":
        aggregates = {
            "rmse_mean": _mean_ignore_none(
                b["random_rmse_mean"] for b in column_blocks.values()
            ),
            "r2_mean": _mean_ignore_none(
                b["random_r2_mean"] for b in column_blocks.values()
            ),
        }
    else:
        aggregates = {
            "auroc_mean": _mean_ignore_none(
                b["random_auroc_mean"] for b in column_blocks.values()
            ),
            "aupr_mean": _mean_ignore_none(
                b["random_aupr_mean"] for b in column_blocks.values()
            ),
        }

    metadata = {
        "source": source,
        "dataset": family,
        "task": task,
        "last_updated": today_iso(),
        "n_molecules": n_mol,
        "n_columns": len(columns),
        "featurizers": list(featurizers),
        "split": {
            "random_n_folds": n_folds,
            "scaffold_split_method": scaffold_method,
            "conserved": True,
            "stratified": stratify is not None,
        },
        "columns": column_blocks,
        **aggregates,
        **_leaderboard_fields(leaderboard),
    }

    out_dir = DATA_ROOT / source / task / family
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({"smiles": canon})
    for col in columns:
        data[col] = label_df[col].to_numpy()  # keep NaN; do not cast
    data.to_csv(out_dir / "data.csv", index=False)
    pd.DataFrame({"random_fold": random_fold, "scaffold_split": scaffold}).to_csv(
        out_dir / "folds.csv", index=False
    )
    for feat, arr in features.items():
        np.save(out_dir / f"{feat}.npy", arr)

    # Full on-disk footprint of the dataset (data + folds + all feature matrices) — what a
    # user downloads. The .npy fingerprint matrices dominate, so this is "size with features".
    payload = [out_dir / "data.csv", out_dir / "folds.csv"]
    payload += [out_dir / f"{feat}.npy" for feat in features]
    metadata["size_bytes"] = int(sum(p.stat().st_size for p in payload if p.exists()))

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Bundle a copy of metadata.json so the family appears in the catalog.
    pkg_dir = PKG_DATA_ROOT / source / task / family
    pkg_dir.mkdir(parents=True, exist_ok=True)
    with open(pkg_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if task == "regression":
        print(
            f"  [{family}] mols={n_mol} columns={len(columns)} "
            f"rmse_mean={_fmt(metadata['rmse_mean'])} -> {out_dir}"
        )
        for col, b in column_blocks.items():
            print(
                f"      - {col}: n={b['n_samples']} "
                f"cv_rmse={_fmt(b['random_rmse_mean'])} "
                f"cv_r2={_fmt(b['random_r2_mean'])} "
                f"scaffold_rmse={_fmt(b['scaffold_rmse'])}"
            )
    else:
        print(
            f"  [{family}] mols={n_mol} columns={len(columns)} "
            f"auroc_mean={_fmt(metadata['auroc_mean'])} -> {out_dir}"
        )
        for col, b in column_blocks.items():
            print(
                f"      - {col}: n={b['n_samples']} pos={b['n_positives']} "
                f"cv_auroc={_fmt(b['random_auroc_mean'])} "
                f"scaffold_auroc={_fmt(b['scaffold_auroc'])}"
            )
    return out_dir


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "n/a"


def print_upload_hint() -> None:
    """Print the command to publish newly prepared datasets via eosvc."""
    print(
        "\nTo publish, run from the repo root, scoped to what you just prepared, e.g.:\n"
        "  eosvc upload --path data/<source>/<task>/<name>\n"
        "(avoid a blanket `--path data/`: raw prep caches such as data/_raw/ don't belong "
        "on the public bucket.)\n"
        "The bundled src/eosbench/_data/**/metadata.json is committed with the package."
    )
