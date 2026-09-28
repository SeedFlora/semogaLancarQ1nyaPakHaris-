#!/usr/bin/env python3
"""Stage 1: export per-sample validation/test predictions for the uncertainty analysis.

Runs next to the thesis artifacts (the RunPod ``/workspace``; embedding cache + CatBoost
checkpoints) and writes, for every image x text pair, checkpoint ``cb``/``pgs`` and
inference mode ``argmax``/``pgs``, one gzip CSV of per-sample records plus a
``manifest.json`` (see CONTRACT.md sections 1 and 2, which are binding).

* ``argmax`` mode: ``model.predict_proba(X)`` (full model; ONNX-served path for ``pgs``).
* ``pgs`` mode: ``virtual_ensembles_predict(..., 'VirtEnsembles', M)`` -> raw logits
  (N, M, 9); per-member softmax; linear pooling ``p = mean_m softmax(z_m)`` (what
  notebooks 07/08 used); log-linear pooling ``q = softmax(mean_m z_m)`` for audit only;
  entropy decomposition (nats) and the legacy ``prob_std`` score.

Features are rebuilt exactly like the notebooks: ``split['row_id']`` indexes the
``.npy`` rows, each modality is L2-normalised row-wise with ``x / clip(norm, 1e-9)``,
then ``concatenate([img, txt], axis=1).astype(float32)``.

Example::

    python export_per_sample_predictions.py --artifacts smartCityReport/artifacts \\
        --out per_sample_predictions
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Constants fixed by CONTRACT.md
# ----------------------------------------------------------------------------------------

#: ``crm.TARGET_CLASSES`` (index = label_id). Cross-checked against crm when importable.
TARGET_CLASSES: tuple[str, ...] = (
    "Dinas Bina Marga",
    "Satuan Polisi Pamong Praja",
    "Dinas Perhubungan",
    "Kelurahan",
    "Dinas Pertamanan dan Hutan",
    "Dinas Sumber Daya Air",
    "Dinas Cipta Karya, Tata Ruang, dan Pertanahan",
    "Badan Pembinaan Badan Usaha Milik Daerah",
    "Instansi lain",
)
N_CLASSES = len(TARGET_CLASSES)
LABELS = list(range(N_CLASSES))

DEFAULT_PAIRS: tuple[str, ...] = (
    "dinov3_large__mE5_large",
    "eva02_large__mE5_large",
    "dinov2_large__mE5_large",
)
DEFAULT_DEPLOYED = "dinov3_large__mE5_large"
CHECKPOINTS: tuple[str, ...] = ("cb", "pgs")
MODES: tuple[str, ...] = ("argmax", "pgs")
SPLIT_ORDER: tuple[str, ...] = ("val", "test")

L2_EPS = 1e-9
ENTROPY_EPS = 1e-12
FLOAT_FORMAT = "%.10g"
POOLING = "linear (mean of per-member softmax)"
SUM_TOL = 1e-6
CRM_CHECK_ROWS = 100

P_COLS = [f"p{k}" for k in range(N_CLASSES)]
Q_COLS = [f"q{k}" for k in range(N_CLASSES)]
BASE_COLUMNS = ["split", "row_id", "label", "pred", *P_COLS]
PGS_EXTRA_COLUMNS = ["mi", "pred_entropy", "exp_entropy", "prob_std", *Q_COLS, "pred_loglin"]

#: 4-decimal test values printed in the manuscript (a list = any of them is acceptable).
EXPECTED_TEST: dict[tuple[str, str, str], dict[str, list[float]]] = {
    ("dinov3_large__mE5_large", "cb", "argmax"): {"accuracy": [0.7996], "macro_f1": [0.7684]},
    ("dinov3_large__mE5_large", "cb", "pgs"): {"accuracy": [0.7914], "macro_f1": [0.7600]},
    ("dinov3_large__mE5_large", "pgs", "argmax"): {"accuracy": [0.8116], "macro_f1": [0.7793]},
    ("dinov3_large__mE5_large", "pgs", "pgs"): {
        "accuracy": [0.8073, 0.8074],
        "macro_f1": [0.7747],
    },
    ("eva02_large__mE5_large", "pgs", "pgs"): {"macro_f1": [0.7747]},
    ("dinov2_large__mE5_large", "pgs", "pgs"): {"macro_f1": [0.7736]},
}
#: Reported but never counted towards ``reproduces_manuscript``.
EXPECTED_INFORMATIONAL: dict[tuple[str, str, str], dict[str, float]] = {
    ("dinov3_large__mE5_large", "pgs", "pgs"): {"mean_prob_std": 0.00339},
}
EXPECTED_TOP1_DISAGREEMENTS = {"dinov3_large__mE5_large": 130}
EXPECTED_SPLIT_SIZES = {"val": 9266, "test": 9266}


class ExportError(RuntimeError):
    """Fatal problem with the global inputs (splits, output directory, CLI)."""


# ----------------------------------------------------------------------------------------
# Pure numerics (unit-tested)
# ----------------------------------------------------------------------------------------


def l2_normalize(x: np.ndarray, eps: float = L2_EPS) -> np.ndarray:
    """Row-wise L2 normalisation, identical to ``crm.fusion._l2`` (dtype preserved)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)


def build_features(img_emb: np.ndarray, txt_emb: np.ndarray, row_ids: np.ndarray) -> np.ndarray:
    """Early-fusion features exactly as notebooks 06/07/08 build them.

    ``img_emb[row_ids]`` / ``txt_emb[row_ids]`` (row i of the cache = metadata row i),
    per-modality L2 normalisation, concatenation, cast to float32.
    """
    img = np.asarray(img_emb[row_ids])
    txt = np.asarray(txt_emb[row_ids])
    if img.shape[0] != txt.shape[0]:
        raise ValueError(f"row mismatch: img {img.shape[0]} vs txt {txt.shape[0]}")
    return np.concatenate([l2_normalize(img), l2_normalize(txt)], axis=1).astype(np.float32)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (same formula as the notebooks' ``pgs_predict_proba``)."""
    shifted = z - z.max(axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=axis, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats over the last axis: ``-sum p log clip(p, 1e-12, 1)``."""
    return -(p * np.log(np.clip(p, ENTROPY_EPS, 1.0))).sum(axis=-1)


@dataclass
class PooledPrediction:
    """Per-sample outputs of the ``pgs`` inference mode."""

    p: np.ndarray  # (N, C) linear pooling: mean_m softmax(z_m)
    q: np.ndarray  # (N, C) log-linear pooling: softmax(mean_m z_m)
    mi: np.ndarray
    pred_entropy: np.ndarray
    exp_entropy: np.ndarray
    prob_std: np.ndarray


def pool_virtual_ensembles(raw: np.ndarray) -> PooledPrediction:
    """Pool CatBoost ``VirtEnsembles`` raw logits of shape (N, M, C) (CONTRACT section 1)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError(f"VirtEnsembles output must be (N, M, C); got {raw.shape}")
    member_p = softmax(raw, axis=-1)
    p = member_p.mean(axis=1)
    pred_entropy = entropy(p)
    exp_entropy = entropy(member_p).mean(axis=1)
    return PooledPrediction(
        p=p,
        q=softmax(raw.mean(axis=1), axis=-1),
        mi=np.maximum(pred_entropy - exp_entropy, 0.0),
        pred_entropy=pred_entropy,
        exp_entropy=exp_entropy,
        prob_std=member_p.std(axis=1).mean(axis=1),
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 over the fixed 9 labels (zero_division=0)."""
    from sklearn.metrics import accuracy_score, f1_score

    if len(y_true) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
    }


def matches_4dp(value: float, candidates: Sequence[float]) -> bool:
    """True iff ``value`` printed with 4 decimals equals one of the candidates."""
    if value is None or not np.isfinite(value):
        return False
    return any(f"{value:.4f}" == f"{c:.4f}" for c in candidates)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------
# Inputs: artifacts, splits, embeddings, checkpoints
# ----------------------------------------------------------------------------------------


@dataclass
class SplitData:
    """One split file (val/test): row ids into the embedding cache and labels."""

    name: str
    row_id: np.ndarray
    label: np.ndarray
    label_name: np.ndarray | None = None  # optional ``label`` text column (class names)

    def __len__(self) -> int:
        return int(self.row_id.shape[0])


@dataclass
class RunState:
    """Mutable bookkeeping shared by the export steps."""

    problems: list[str] = field(default_factory=list)  # skipped / invalid inputs
    mismatches: list[str] = field(default_factory=list)  # manuscript values not reproduced
    notes: list[str] = field(default_factory=list)  # informational deviations

    def problem(self, msg: str) -> None:
        self.problems.append(msg)
        log(f"WARNING: {msg}")


def log(msg: str) -> None:
    """Timestamped progress line (ASCII only: Windows consoles may be cp1252)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def detect_artifacts(explicit: str | None) -> Path:
    """Return the artifacts directory (explicit, or first plausible candidate)."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise ExportError(f"--artifacts {path} is not a directory")
        return path
    bases = [Path.cwd(), _script_dir()]
    rels = ["../artifacts", "artifacts", "smartCityReport/artifacts",
            "../smartCityReport/artifacts"]
    tried: list[str] = []
    for base in bases:
        for rel in rels:
            cand = (base / rel).resolve()
            tried.append(str(cand))
            if (cand / "splits").is_dir() and (cand / "embeddings").is_dir():
                return cand
    raise ExportError(
        "could not auto-detect the artifacts directory (needs splits/ and embeddings/); "
        "pass --artifacts. Tried:\n  " + "\n  ".join(dict.fromkeys(tried))
    )


def parse_pairs(values: Sequence[str]) -> list[str]:
    """Accept ``a__b c__d`` and/or ``a__b,c__d``; keep order, drop duplicates."""
    out: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if len(item.split("__")) != 2 or not all(item.split("__")):
                raise ExportError(f"pair must look like IMAGE__TEXT, got {item!r}")
            if item not in out:
                out.append(item)
    return out


_CKPT_RE = re.compile(r"^(?P<img>.+?)__(?P<txt>.+?)__(?P<ckpt>cb|pgs)\.cbm$")


def pairs_in_checkpoint_dir(ckpt_dir: Path) -> list[str]:
    """All ``IMAGE__TEXT`` tags that have at least one ``__cb``/``__pgs`` checkpoint."""
    tags = set()
    for f in ckpt_dir.glob("*.cbm"):
        m = _CKPT_RE.match(f.name)
        if m:
            tags.add(f"{m['img']}__{m['txt']}")
    return sorted(tags)


def parse_splits(value: str) -> list[str]:
    """``val,test`` -> canonical order (val first, then test)."""
    requested = {s.strip() for s in value.split(",") if s.strip()}
    unknown = requested - set(SPLIT_ORDER)
    if unknown or not requested:
        raise ExportError(f"--splits must be a non-empty subset of {SPLIT_ORDER}, got {value!r}")
    return [s for s in SPLIT_ORDER if s in requested]


def _int_column(df: pd.DataFrame, col: str, where: str) -> np.ndarray:
    if col not in df.columns:
        raise ExportError(f"{where}: missing column {col!r} (has {list(df.columns)})")
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().any():
        raise ExportError(f"{where}: column {col!r} has missing/non-numeric values")
    arr = values.to_numpy()
    if not np.all(np.equal(arr, np.floor(arr))):
        raise ExportError(f"{where}: column {col!r} must be integer-valued")
    return arr.astype(np.int64)


def load_split(splits_dir: Path, name: str) -> SplitData:
    """Read ``splits/{name}.csv`` and validate ``row_id`` / ``label_id``."""
    path = splits_dir / f"{name}.csv"
    if not path.is_file():
        raise ExportError(f"split file not found: {path}")
    df = pd.read_csv(path)
    where = str(path)
    row_id = _int_column(df, "row_id", where)
    label = _int_column(df, "label_id", where)
    if len(df) == 0:
        raise ExportError(f"{where}: split is empty")
    if (row_id < 0).any():
        raise ExportError(f"{where}: negative row_id")
    if len(np.unique(row_id)) != len(row_id):
        raise ExportError(f"{where}: duplicated row_id values")
    bad = (label < 0) | (label >= N_CLASSES)
    if bad.any():
        raise ExportError(
            f"{where}: label_id outside 0..{N_CLASSES - 1} "
            f"(e.g. {sorted(set(label[bad].tolist()))[:5]})"
        )
    names = df["label"].astype(str).to_numpy() if "label" in df.columns else None
    return SplitData(name=name, row_id=row_id, label=label, label_name=names)


def validate_split_set(splits: dict[str, SplitData], splits_dir: Path, state: RunState) -> None:
    """Disjointness (also against train.csv when present) and label coverage."""
    ids = {name: set(sp.row_id.tolist()) for name, sp in splits.items()}
    train_path = splits_dir / "train.csv"
    if train_path.is_file():
        train = pd.read_csv(train_path, usecols=lambda c: c == "row_id")
        if "row_id" in train.columns:
            ids["train"] = set(train["row_id"].astype(np.int64).tolist())
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = len(ids[a] & ids[b])
            if overlap:
                state.problem(f"splits {a} and {b} share {overlap} row_id values")
    for name, sp in splits.items():
        missing = sorted(set(LABELS) - set(sp.label.tolist()))
        if missing:
            state.notes.append(f"split {name} has no samples of class(es) {missing}")
        # notebook 03: label_id = LABEL2ID[label], so the text column pins the class order
        if sp.label_name is not None:
            expected_names = np.asarray(TARGET_CLASSES, dtype=object)[sp.label]
            wrong = int((sp.label_name != expected_names).sum())
            if wrong:
                state.problem(
                    f"split {name}: {wrong} rows whose 'label' text is not "
                    "TARGET_CLASSES[label_id]; the class order assumed by CONTRACT.md does "
                    "not match this split file"
                )


class EmbeddingCache:
    """Memory-mapped ``.npy`` embeddings, loaded once per encoder name."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def path(self, modality: str, name: str) -> Path:
        return self.root / modality / f"{name}.npy"

    def get(self, modality: str, name: str) -> np.ndarray:
        key = (modality, name)
        if key not in self._cache:
            path = self.path(modality, name)
            if not path.is_file():
                raise FileNotFoundError(f"embedding not found: {path}")
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"{path}: expected a 2-D array, got shape {arr.shape}")
            self._cache[key] = arr
        return self._cache[key]


def load_crm(crm_src: str | None, artifacts: Path) -> ModuleType | None:
    """Import ``crm.fusion`` if available (``--crm-src``, ``<artifacts>/../src``, ``../src``...)."""
    candidates: list[Path] = []
    if crm_src:
        candidates.append(Path(crm_src).expanduser())
    candidates += [
        artifacts.parent / "src",
        Path.cwd() / ".." / "src",
        Path.cwd() / "src",
        _script_dir() / ".." / "src",
        _script_dir() / ".." / "smartCityReport" / "src",
    ]
    for cand in candidates:
        if (cand / "crm" / "__init__.py").is_file():
            resolved = str(cand.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            break
    try:
        fusion = importlib.import_module("crm.fusion")
        crm = importlib.import_module("crm")
    except Exception:  # noqa: BLE001 - optional dependency, any import failure = absent
        return None
    classes = tuple(getattr(crm, "TARGET_CLASSES", ()))
    if classes and classes != TARGET_CLASSES:
        raise ExportError(
            f"crm.TARGET_CLASSES differs from the contract class order: {classes}"
        )
    return fusion


def load_model(path: Path) -> Any:
    """Load a CatBoost classifier checkpoint."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def validate_model(model: Any, n_features: int) -> str | None:
    """Return an error message if the checkpoint does not fit the contract, else None.

    Besides class order and feature count this requires ``loss_function == 'MultiClass'``:
    both ``predict_proba`` and the per-member softmax of ``VirtEnsembles`` assume softmax
    logits. A ``MultiClassOneVsAll`` model also returns (N, M, 9) raw values, so without
    this guard it would be pooled silently with the wrong link function.
    """
    loss = str(model.get_all_params().get("loss_function", ""))
    if loss.split(":")[0] != "MultiClass":
        return f"loss_function is {loss!r}; softmax pooling requires 'MultiClass'"
    try:
        classes = [int(c) for c in model.classes_]
    except (TypeError, ValueError):
        return f"non-integer classes_ {list(model.classes_)}"
    if classes != LABELS:
        return f"model has {len(classes)} classes {classes}; expected {LABELS}"
    names = model.feature_names_ or []
    if len(names) != n_features:
        return f"model expects {len(names)} features but the fused vector has {n_features}"
    if int(model.tree_count_) < 1:
        return "model has no trees"
    return None


# ----------------------------------------------------------------------------------------
# Prediction (batched)
# ----------------------------------------------------------------------------------------


def predict_argmax(model: Any, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Full-model probabilities ``predict_proba`` (N, 9) in float64."""
    out = np.empty((X.shape[0], N_CLASSES), dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        out[start:stop] = np.asarray(model.predict_proba(X[start:stop]), dtype=np.float64)
    return out


def predict_pgs(model: Any, X: np.ndarray, n_members: int, batch_size: int) -> PooledPrediction:
    """Virtual-ensemble inference, pooled batch by batch (memory ~ batch x M x 9)."""
    n = X.shape[0]
    p = np.empty((n, N_CLASSES))
    q = np.empty((n, N_CLASSES))
    mi, pe, ee, ps = (np.empty(n) for _ in range(4))
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        raw = np.asarray(
            model.virtual_ensembles_predict(
                X[start:stop],
                prediction_type="VirtEnsembles",
                virtual_ensembles_count=n_members,
            ),
            dtype=np.float64,
        )
        if raw.shape != (stop - start, n_members, N_CLASSES):
            raise ValueError(
                f"VirtEnsembles returned shape {raw.shape}; "
                f"expected ({stop - start}, {n_members}, {N_CLASSES})"
            )
        if not np.isfinite(raw).all():
            raise ValueError("VirtEnsembles returned NaN/inf logits")
        pooled = pool_virtual_ensembles(raw)
        p[start:stop], q[start:stop] = pooled.p, pooled.q
        mi[start:stop], pe[start:stop] = pooled.mi, pooled.pred_entropy
        ee[start:stop], ps[start:stop] = pooled.exp_entropy, pooled.prob_std
    return PooledPrediction(p=p, q=q, mi=mi, pred_entropy=pe, exp_entropy=ee, prob_std=ps)


# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------


def output_filename(pair: str, ckpt: str, mode: str) -> str:
    return f"preds__{pair}__{ckpt}__{mode}.csv.gz"


def build_frame(
    split_col: np.ndarray,
    row_id: np.ndarray,
    label: np.ndarray,
    probs: np.ndarray,
    pooled: PooledPrediction | None,
) -> pd.DataFrame:
    """Assemble the per-sample table in the exact CONTRACT column order."""
    data: dict[str, Any] = {
        "split": split_col,
        "row_id": row_id.astype(np.int64),
        "label": label.astype(np.int64),
        "pred": probs.argmax(axis=1).astype(np.int64),
    }
    for k in range(N_CLASSES):
        data[f"p{k}"] = probs[:, k]
    columns = list(BASE_COLUMNS)
    if pooled is not None:
        data["mi"] = pooled.mi
        data["pred_entropy"] = pooled.pred_entropy
        data["exp_entropy"] = pooled.exp_entropy
        data["prob_std"] = pooled.prob_std
        for k in range(N_CLASSES):
            data[f"q{k}"] = pooled.q[:, k]
        data["pred_loglin"] = pooled.q.argmax(axis=1).astype(np.int64)
        columns += PGS_EXTRA_COLUMNS
    return pd.DataFrame(data, columns=columns)


def write_frame(df: pd.DataFrame, path: Path) -> None:
    """gzip CSV, ``%.10g`` floats, LF line endings, reproducible gzip header (mtime 0)."""
    df.to_csv(
        path,
        index=False,
        float_format=FLOAT_FORMAT,
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )


def verify_written(path: Path, expected: pd.DataFrame) -> dict[str, Any]:
    """Re-read a written file and check order, row count, sums and the argmax."""
    back = pd.read_csv(path)
    issues: list[str] = []
    if list(back.columns) != list(expected.columns):
        issues.append(f"column order {list(back.columns)}")
    if len(back) != len(expected):
        issues.append(f"row count {len(back)} != {len(expected)}")
    p = back[P_COLS].to_numpy(dtype=np.float64)
    max_dev = float(np.abs(p.sum(axis=1) - 1.0).max()) if len(p) else 0.0
    if not max_dev <= SUM_TOL:
        issues.append(f"probabilities deviate from 1 by {max_dev:.3g}")
    rounded_mismatch = int((p.argmax(axis=1) != back["pred"].to_numpy()).sum())
    if rounded_mismatch:
        issues.append(f"{rounded_mismatch} rows where argmax of the rounded p differs from pred")
    return {"max_sum_deviation": max_dev, "pred_mismatch_after_rounding": rounded_mismatch,
            "issues": issues}


def _finite_or_none(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _finite_or_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite_or_none(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = out_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(_finite_or_none(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


# ----------------------------------------------------------------------------------------
# Export driver
# ----------------------------------------------------------------------------------------


@dataclass
class PairInputs:
    """Fused features and per-row metadata of one pair (rows: val block, then test block)."""

    X: np.ndarray
    row_id: np.ndarray
    label: np.ndarray
    split_col: np.ndarray


def prepare_pair(
    pair: str, emb: EmbeddingCache, splits: dict[str, SplitData], state: RunState
) -> PairInputs | None:
    """Validate the embedding caches of a pair and build its features (None = skipped)."""
    img_name, txt_name = pair.split("__")
    try:
        img_emb = emb.get("image", img_name)
        txt_emb = emb.get("text", txt_name)
    except (FileNotFoundError, ValueError) as exc:
        state.problem(f"{pair}: skipped ({exc})")
        return None
    max_row = max(int(sp.row_id.max()) for sp in splits.values())
    for what, arr in ((f"image/{img_name}", img_emb), (f"text/{txt_name}", txt_emb)):
        if arr.shape[0] <= max_row:
            state.problem(
                f"{pair}: skipped (embedding {what} has {arr.shape[0]} rows but the splits "
                f"reference row_id {max_row})"
            )
            return None
    if img_emb.shape[0] != txt_emb.shape[0]:
        state.problem(
            f"{pair}: image cache has {img_emb.shape[0]} rows, text cache {txt_emb.shape[0]} "
            "(notebooks expected equal lengths); continuing"
        )
    names = list(splits)
    X = np.concatenate([build_features(img_emb, txt_emb, splits[s].row_id) for s in names])
    if not np.isfinite(X).all():
        state.problem(f"{pair}: skipped (non-finite values in the fused features)")
        return None
    log(f"{pair}: features {X.shape} {X.dtype} (image dim {img_emb.shape[1]}, "
        f"text dim {txt_emb.shape[1]})")
    return PairInputs(
        X=X,
        row_id=np.concatenate([splits[s].row_id for s in names]),
        label=np.concatenate([splits[s].label for s in names]),
        split_col=np.concatenate([np.full(len(splits[s]), s, dtype=object) for s in names]),
    )


def crosscheck_crm(
    pair: str, emb: EmbeddingCache, first_split: SplitData, X: np.ndarray,
    crm_fusion: ModuleType | None,
) -> str:
    """Assert the self-contained features equal ``crm.fusion.early_fusion`` (first rows)."""
    if crm_fusion is None:
        return "crm not importable; skipped"
    img_name, txt_name = pair.split("__")
    first = first_split.row_id[:CRM_CHECK_ROWS]
    ref = crm_fusion.early_fusion(
        emb.get("image", img_name)[first], emb.get("text", txt_name)[first], l2_per_modality=True
    )
    mine = X[: len(first)]
    if ref.shape != mine.shape or ref.dtype != mine.dtype or not np.array_equal(ref, mine):
        diff = (f"max abs diff {float(np.abs(ref.astype(np.float64) - mine).max()):.3g}"
                if ref.shape == mine.shape else f"shape {ref.shape} vs {mine.shape}")
        # systematic (same code for every pair): abort instead of exporting wrong features
        raise ExportError(
            f"{pair}: self-contained features differ from crm.fusion.early_fusion on the first "
            f"{len(first)} rows ({diff}, dtype {ref.dtype} vs {mine.dtype})"
        )
    msg = f"identical to crm.fusion.early_fusion on first {len(first)} rows"
    log(f"{pair}: {msg}")
    return msg


def compute_metrics(
    inputs: PairInputs, frame: pd.DataFrame, pooled: PooledPrediction | None,
    split_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-split accuracy / macro-F1 (+ log-linear and uncertainty means on test, pgs mode)."""
    pred = frame["pred"].to_numpy()
    metrics = {
        s: classification_metrics(inputs.label[inputs.split_col == s],
                                  pred[inputs.split_col == s])
        for s in split_names
    }
    if pooled is not None and "test" in split_names:
        is_test = inputs.split_col == "test"
        loglin = classification_metrics(inputs.label[is_test],
                                        frame["pred_loglin"].to_numpy()[is_test])
        metrics["test"].update(
            accuracy_loglin=loglin["accuracy"],
            macro_f1_loglin=loglin["macro_f1"],
            mean_prob_std=float(pooled.prob_std[is_test].mean()),
            mean_mi=float(pooled.mi[is_test].mean()),
        )
    return metrics


def check_expectations(
    key: tuple[str, str, str], metrics: dict[str, dict[str, float]], expect: bool,
    state: RunState,
) -> tuple[dict[str, list[float]] | None, dict[str, float] | None, bool | None]:
    """Return (expected_test, informational, reproduces_manuscript) for one exported file."""
    if not expect:
        return None, None, None
    pair, ckpt, mode = key
    expected = EXPECTED_TEST.get(key)
    informational = EXPECTED_INFORMATIONAL.get(key)
    test = metrics.get("test")
    reproduces: bool | None = None
    if expected is not None and test is not None:
        reproduces = True
        for metric, cands in expected.items():
            value = test[metric]
            if not matches_4dp(value, cands):
                reproduces = False
                state.mismatches.append(
                    f"{pair} {ckpt}/{mode}: test {metric} = {value:.4f}, manuscript "
                    + " or ".join(f"{c:.4f}" for c in cands)
                )
    if informational and test is not None:
        for metric, ref in informational.items():
            value = test.get(metric)
            if value is not None and f"{value:.5f}" != f"{ref:.5f}":
                state.notes.append(
                    f"{pair} {ckpt}/{mode}: test {metric} = {value:.5f} "
                    f"(manuscript ~{ref:.5f}; informational only)"
                )
    return expected, informational, reproduces


def export_pair(
    pair: str,
    *,
    args: argparse.Namespace,
    ckpt_dir: Path,
    emb: EmbeddingCache,
    splits: dict[str, SplitData],
    crm_fusion: ModuleType | None,
    out_dir: Path,
    state: RunState,
    manifest: dict[str, Any],
) -> dict[tuple[str, str], np.ndarray]:
    """Export every (checkpoint, mode) of one pair; return {(ckpt, mode): pred} arrays."""
    img_name, txt_name = pair.split("__")
    split_names = list(splits)
    preds: dict[tuple[str, str], np.ndarray] = {}
    inputs = prepare_pair(pair, emb, splits, state)
    if inputs is None:
        return preds
    manifest.setdefault("feature_crosscheck", {})[pair] = crosscheck_crm(
        pair, emb, splits[split_names[0]], inputs.X, crm_fusion
    )

    for ckpt in CHECKPOINTS:
        cbm = ckpt_dir / f"{pair}__{ckpt}.cbm"
        if not cbm.is_file():
            state.problem(f"{pair}: checkpoint {cbm.name} not found; skipped")
            continue
        try:
            model = load_model(cbm)
        except Exception as exc:  # noqa: BLE001 - report and continue
            state.problem(f"{pair}: cannot load {cbm.name} ({exc}); skipped")
            continue
        invalid = validate_model(model, inputs.X.shape[1])
        if invalid:
            state.problem(f"{pair}: {cbm.name} invalid ({invalid}); skipped")
            continue
        info = {
            "cbm_path": cbm.resolve().as_posix(),
            "cbm_sha256": sha256_file(cbm),
            "tree_count": int(model.tree_count_),
            "posterior_sampling": bool(model.get_all_params().get("posterior_sampling", False)),
        }
        # __cb must be the plain model and __pgs the SGLB one (notebooks 06/07/08); swapped
        # or overwritten files would otherwise only surface as a manuscript mismatch.
        if info["posterior_sampling"] != (ckpt == "pgs"):
            state.problem(
                f"{pair}: {cbm.name} has posterior_sampling={info['posterior_sampling']}, "
                f"expected {ckpt == 'pgs'} for a '{ckpt}' checkpoint (exported anyway)"
            )
        for mode in MODES:
            t0 = time.time()
            try:
                if mode == "argmax":
                    probs, pooled = predict_argmax(model, inputs.X, args.batch_size), None
                else:
                    pooled = predict_pgs(model, inputs.X, args.virtual_ensembles,
                                         args.batch_size)
                    probs = pooled.p
            except Exception as exc:  # noqa: BLE001 - e.g. too few trees for M members
                state.problem(f"{pair} {ckpt}/{mode}: prediction failed ({exc}); skipped")
                continue
            dev = float(np.abs(probs.sum(axis=1) - 1.0).max())
            if not np.isfinite(probs).all() or dev > SUM_TOL:
                state.problem(f"{pair} {ckpt}/{mode}: invalid probabilities (sum dev {dev:.3g})")
            frame = build_frame(inputs.split_col, inputs.row_id, inputs.label, probs, pooled)
            fname = output_filename(pair, ckpt, mode)
            write_frame(frame, out_dir / fname)
            check = verify_written(out_dir / fname, frame)
            for issue in check["issues"]:
                state.problem(f"{fname}: {issue}")

            preds[(ckpt, mode)] = frame["pred"].to_numpy()
            metrics = compute_metrics(inputs, frame, pooled, split_names)
            expected, informational, reproduces = check_expectations(
                (pair, ckpt, mode), metrics, args.expect, state
            )
            manifest["files"].append({
                "file": fname,
                "image": img_name,
                "text": txt_name,
                "checkpoint": ckpt,
                "mode": mode,
                **info,
                "virtual_ensembles": args.virtual_ensembles if mode == "pgs" else None,
                "n_rows": int(len(frame)),
                "sha256": sha256_file(out_dir / fname),
                "metrics": metrics,
                "expected_test": expected,
                "expected_test_informational": informational,
                "reproduces_manuscript": reproduces,
                "readback": {k: check[k]
                             for k in ("max_sum_deviation", "pred_mismatch_after_rounding")},
            })
            write_manifest(out_dir, manifest)
            test_m = metrics.get("test")
            log(f"{fname}: {len(frame)} rows, {time.time() - t0:.1f}s"
                + (f", test acc {test_m['accuracy']:.4f} F1 {test_m['macro_f1']:.4f}"
                   if test_m else ""))
        del model
    return preds


def print_summary(manifest: dict[str, Any], state: RunState, expect: bool) -> None:
    """Human-readable summary table plus the WARNING/problem blocks."""
    rows = []
    for f in manifest["files"]:
        m = f["metrics"]
        exp = f["expected_test"] or {}
        exp_txt = " / ".join(
            "|".join(f"{c:.4f}" for c in exp[k]) if k in exp else "-"
            for k in ("accuracy", "macro_f1")
        ) if exp else "-"
        status = {True: "OK", False: "MISMATCH", None: "-"}[f["reproduces_manuscript"]]
        rows.append([
            f"{f['image']}__{f['text']}", f["checkpoint"], f["mode"], str(f["tree_count"]),
            *(f"{m[s][k]:.4f}" if s in m and m[s][k] is not None else "-"
              for s in ("val", "test") for k in ("accuracy", "macro_f1")),
            exp_txt, status,
        ])
    header = ["pair", "ckpt", "mode", "trees", "val_acc", "val_F1", "test_acc", "test_F1",
              "manuscript acc / F1", "status"]
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = "  ".join
    print("\n" + "=" * 100)
    print("SUMMARY  (test metrics vs. manuscript Table 4; accuracy / macro-F1 at 4 dp)")
    print("=" * 100)
    print(line(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print(line("-" * w for w in widths))
    for r in rows:
        print(line(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))
    for f in manifest["files"]:
        if f["mode"] == "pgs" and "test" in f["metrics"]:
            t = f["metrics"]["test"]
            print(f"  {f['image']}__{f['text']} {f['checkpoint']}/pgs: loglin acc "
                  f"{t['accuracy_loglin']:.4f} F1 {t['macro_f1_loglin']:.4f}, mean prob_std "
                  f"{t['mean_prob_std']:.5f}, mean MI {t['mean_mi']:.5f} nats")
    top1 = manifest.get("top1_disagreements_pgs_vs_argmax")
    print(f"  top-1 disagreements pgs/argmax vs pgs/pgs ({manifest['deployed_pair']}, test): "
          f"{top1 if top1 is not None else 'n/a'}"
          + (f" (manuscript {manifest['top1_disagreements_expected']})"
             if manifest.get("top1_disagreements_expected") is not None else ""))
    print(f"  n_val={manifest['n_val']}  n_test={manifest['n_test']}  "
          f"files={len(manifest['files'])}  out={manifest['out_dir']}")
    if state.notes:
        print("\nNOTES:")
        for n in state.notes:
            print(f"  - {n}")
    if state.problems:
        print("\n" + "!" * 100)
        print(f"PROBLEMS ({len(state.problems)}): some inputs were skipped or look wrong")
        for p in state.problems:
            print(f"  - {p}")
        print("!" * 100)
    if expect and state.mismatches:
        print("\n" + "#" * 100)
        print("#" * 100)
        print(f"WARNING: {len(state.mismatches)} MANUSCRIPT VALUE(S) NOT REPRODUCED")
        for mm in state.mismatches:
            print(f"   * {mm}")
        print("The files were written anyway. Check that the checkpoints, embedding caches and")
        print("split files are the ones used for the manuscript before running stage 2.")
        print("#" * 100)
        print("#" * 100)
    elif expect:
        print("\nAll manuscript expectations checked here were reproduced.")
    else:
        print("\n(--no-expect: manuscript expectation checks skipped)")


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    """Run the whole export; returns the manifest dict (also written to disk)."""
    import catboost
    import sklearn

    if args.virtual_ensembles < 2:
        raise ExportError("--virtual-ensembles must be >= 2")
    if args.batch_size < 1:
        raise ExportError("--batch-size must be >= 1")
    artifacts = detect_artifacts(args.artifacts)
    ckpt_dir = Path(args.checkpoints) if args.checkpoints else artifacts / "models" / "checkpoints"
    splits_dir = artifacts / "splits"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_names = parse_splits(args.splits)
    state = RunState()
    log(f"artifacts: {artifacts.resolve()}")
    log(f"checkpoints: {ckpt_dir.resolve()}")
    if not ckpt_dir.is_dir():
        raise ExportError(f"checkpoint directory not found: {ckpt_dir}")

    if args.all_pairs:
        pairs = pairs_in_checkpoint_dir(ckpt_dir)
        if not pairs:
            raise ExportError(f"--all-pairs: no *__cb.cbm / *__pgs.cbm in {ckpt_dir}")
    else:
        pairs = parse_pairs(args.pairs or list(DEFAULT_PAIRS))
    deployed = parse_pairs([args.deployed])[0]
    log(f"pairs: {pairs}")

    splits = {name: load_split(splits_dir, name) for name in split_names}
    validate_split_set(splits, splits_dir, state)
    for name, sp in splits.items():
        log(f"split {name}: {len(sp)} rows, max row_id {int(sp.row_id.max())}, "
            f"class counts {np.bincount(sp.label, minlength=N_CLASSES).tolist()}")
        exp_n = EXPECTED_SPLIT_SIZES.get(name)
        if args.expect and exp_n is not None and len(sp) != exp_n:
            state.mismatches.append(f"split {name} has {len(sp)} rows, manuscript n = {exp_n}")

    crm_fusion = load_crm(args.crm_src, artifacts)
    log("crm.fusion: " + ("found, cross-check enabled" if crm_fusion else "not importable"))

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"),
        "catboost_version": catboost.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "virtual_ensembles": args.virtual_ensembles,
        "pooling": POOLING,
        # rows exported per split (0 = split not exported); CONTRACT types these as ints and
        # stage 2 does int(manifest["n_val"]), so never write null here
        "n_val": len(splits["val"]) if "val" in splits else 0,
        "n_test": len(splits["test"]) if "test" in splits else 0,
        "splits": split_names,
        "class_order": list(TARGET_CLASSES),
        "float_format": FLOAT_FORMAT,
        "artifacts_dir": artifacts.resolve().as_posix(),
        "out_dir": out_dir.resolve().as_posix(),
        "expectations_checked": bool(args.expect),
        "deployed_pair": deployed,
        "top1_disagreements_pgs_vs_argmax": None,
        "top1_disagreements_pgs_vs_argmax_val": None,
        "top1_disagreements_expected": (
            EXPECTED_TOP1_DISAGREEMENTS.get(deployed) if args.expect else None),
        "files": [],
    }
    # Replace any manifest left by an earlier run right away, so a crash below can never
    # leave an old manifest pointing at a mix of old and new CSVs.
    write_manifest(out_dir, manifest)

    emb = EmbeddingCache(artifacts / "embeddings")
    for pair in pairs:
        preds = export_pair(
            pair, args=args, ckpt_dir=ckpt_dir, emb=emb, splits=splits,
            crm_fusion=crm_fusion, out_dir=out_dir, state=state, manifest=manifest,
        )
        if pair == deployed and ("pgs", "argmax") in preds and ("pgs", "pgs") in preds:
            split_col = np.concatenate([np.full(len(splits[s]), s) for s in split_names])
            differ = preds[("pgs", "argmax")] != preds[("pgs", "pgs")]
            for split, key in (("test", "top1_disagreements_pgs_vs_argmax"),
                               ("val", "top1_disagreements_pgs_vs_argmax_val")):
                if split in splits:
                    manifest[key] = int(differ[split_col == split].sum())

    if deployed not in pairs:
        state.notes.append(f"deployed pair {deployed} was not exported")
    exp_top1 = manifest["top1_disagreements_expected"]
    got_top1 = manifest["top1_disagreements_pgs_vs_argmax"]
    if exp_top1 is not None and got_top1 != exp_top1:
        state.mismatches.append(
            f"top-1 disagreements pgs/argmax vs pgs/pgs ({deployed}, test) = {got_top1}, "
            f"manuscript {exp_top1}"
        )
    listed = {f["file"] for f in manifest["files"]}
    stale = sorted(p.name for p in out_dir.glob("preds__*.csv.gz") if p.name not in listed)
    if stale:
        state.problem(
            f"{len(stale)} prediction file(s) in {out_dir} are from an earlier run and NOT "
            f"listed in manifest.json (delete them before copying the folder): {stale}"
        )
    manifest["problems"] = state.problems
    manifest["manuscript_mismatches"] = state.mismatches if args.expect else []
    manifest["notes"] = state.notes
    write_manifest(out_dir, manifest)
    print_summary(manifest, state, args.expect)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    # Defaults are spelled out in the help strings: ArgumentDefaultsHelpFormatter printed
    # "(default: True)" for --no-expect and duplicated the hand-written defaults.
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--artifacts", default=None,
                   help="artifacts dir (default: auto-detect ../artifacts, ./artifacts, "
                        "smartCityReport/artifacts)")
    p.add_argument("--out", default="per_sample_predictions",
                   help="output directory (default: per_sample_predictions)")
    p.add_argument("--pairs", nargs="+", default=None, metavar="IMAGE__TEXT",
                   help="pairs to export, space- or comma-separated "
                        f"(default: {' '.join(DEFAULT_PAIRS)})")
    p.add_argument("--all-pairs", action="store_true",
                   help="export every pair that has a checkpoint in models/checkpoints/")
    p.add_argument("--virtual-ensembles", type=int, default=30,
                   help="M for VirtEnsembles (default: 30)")
    p.add_argument("--splits", default="val,test",
                   help="comma-separated subset of val,test (default: val,test)")
    p.add_argument("--no-expect", dest="expect", action="store_false",
                   help="skip the manuscript expectation checks (they run by default)")
    p.add_argument("--deployed", default=DEFAULT_DEPLOYED,
                   help=f"pair used for the top-1 disagreement count (default: {DEFAULT_DEPLOYED})")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="rows per prediction call (default: 2048)")
    p.add_argument("--checkpoints", default=None,
                   help="checkpoint dir (default <artifacts>/models/checkpoints)")
    p.add_argument("--crm-src", default=None,
                   help="directory containing the crm package (for the feature cross-check)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_export(args)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if manifest["files"] else 1


if __name__ == "__main__":
    sys.exit(main())
