#!/usr/bin/env python3
"""Synthetic ``per_sample_predictions/`` directory in the exact CONTRACT.md section 2 schema.

Independent of the stage-1 export script: probabilities are simulated directly.  The data are
built to look like the real problem (9 imbalanced classes, ~80 % accuracy, over-confident
served probabilities, PGS members that disagree more on hard samples, correlated encoders), so
the stage-2 figures and prose can be inspected on realistic numbers.

Usage::

    python tests/synthetic_per_sample.py --out _work/synthetic_preds [--n-val 9266 --n-test 9266]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

N_CLASSES = 9
PRIOR = np.array([0.24, 0.17, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04])
PROB_COLS = [f"p{k}" for k in range(N_CLASSES)]
Q_COLS = [f"q{k}" for k in range(N_CLASSES)]


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=axis)


def sample_categorical(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(p.shape[0])[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def _frame(split: np.ndarray, row_id: np.ndarray, label: np.ndarray, probs: np.ndarray,
           members: np.ndarray | None) -> pd.DataFrame:
    df = pd.DataFrame({"split": split, "row_id": row_id, "label": label, "pred": probs.argmax(axis=1)})
    for k in range(N_CLASSES):
        df[f"p{k}"] = probs[:, k]
    if members is not None:
        pm = softmax(members, axis=2)                      # (N, M, 9) per-member probabilities
        mean_p = pm.mean(axis=1)
        pe = entropy(mean_p)
        ee = entropy(pm).mean(axis=1)
        df["mi"] = np.maximum(pe - ee, 0.0)
        df["pred_entropy"] = pe
        df["exp_entropy"] = ee
        df["prob_std"] = pm.std(axis=1).mean(axis=1)
        q = softmax(members.mean(axis=1), axis=1)
        for k in range(N_CLASSES):
            df[f"q{k}"] = q[:, k]
        df["pred_loglin"] = q.argmax(axis=1)
    return df


def make_synthetic_preds(out: Path, n_val: int = 9266, n_test: int = 9266, seed: int = 7, m: int = 30,
                         images: Sequence[str] = ("dinov3_large", "eva02_large", "dinov2_large"),
                         text: str = "mE5_large", zero_mi: int = 12,
                         expected: str = "self") -> dict[str, Any]:
    """Write CSVs + manifest.json into ``out``; return the manifest.

    ``expected="self"`` fills ``expected_test`` with the rounded synthetic values (so every row
    reproduces); ``expected="none"`` leaves them null.
    """
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n = n_val + n_test
    split = np.array(["val"] * n_val + ["test"] * n_test)
    row_id = np.sort(rng.choice(np.arange(3 * n), size=n, replace=False))
    row_id = np.concatenate([rng.permutation(row_id[:n_val]), rng.permutation(row_id[n_val:])])
    latent = rng.choice(N_CLASSES, size=n, p=PRIOR)
    margin = rng.gamma(shape=3.0, scale=2.4, size=n)
    base = rng.normal(0.0, 1.0, size=(n, N_CLASSES))
    base[np.arange(n), latent] += margin
    # labels come from a softer posterior than the one the model reports -> over-confidence
    label = sample_categorical(softmax(base / 1.35), rng)
    hardness = np.exp(-margin / 2.5)
    manifest_files = []
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for ii, img in enumerate(images):
        enc_noise = [0.0, 0.55, 0.60][ii % 3]
        u_pgs = base + rng.normal(0.0, enc_noise, size=base.shape) if enc_noise else base.copy()
        u_cb = 0.93 * u_pgs + rng.normal(0.0, 0.40, size=base.shape)
        for ckpt, u in (("cb", u_cb), ("pgs", u_pgs)):
            s = 0.02 + 0.3 * hardness
            s[:zero_mi] = 0.0                               # identical members -> MI exactly 0
            # virtual-ensemble centre differs slightly from the full model (prefix truncation)
            centre = 0.97 * u + rng.normal(0.0, 0.25, size=u.shape)
            members = centre[:, None, :] + rng.normal(0.0, 1.0, size=(n, m, N_CLASSES)) * s[:, None, None]
            for mode in ("argmax", "pgs"):
                if mode == "argmax":
                    df = _frame(split, row_id, label, softmax(u, axis=1), None)
                else:
                    df = _frame(split, row_id, label, softmax(members, axis=2).mean(axis=1), members)
                frames[(img, ckpt, mode)] = df
                fname = f"preds__{img}__{text}__{ckpt}__{mode}.csv.gz"
                df.to_csv(out / fname, index=False, float_format="%.10g", compression="gzip")
                test = df[df["split"] == "test"]
                acc = float((test["pred"] == test["label"]).mean())
                f1 = float(f1_score(test["label"], test["pred"], average="macro", labels=list(range(N_CLASSES)),
                                    zero_division=0))
                metrics: dict[str, Any] = {"accuracy": acc, "macro_f1": f1}
                if mode == "pgs":
                    metrics.update({
                        "accuracy_loglin": float((test["pred_loglin"] == test["label"]).mean()),
                        "macro_f1_loglin": float(f1_score(test["label"], test["pred_loglin"], average="macro",
                                                          labels=list(range(N_CLASSES)), zero_division=0)),
                        "mean_prob_std": float(test["prob_std"].mean()), "mean_mi": float(test["mi"].mean())})
                exp = None
                if expected == "self":
                    if img == images[0]:
                        exp = {"accuracy": [round(acc, 4)], "macro_f1": [round(f1, 4)]}
                    elif ckpt == "pgs" and mode == "pgs":
                        exp = {"macro_f1": [round(f1, 4)]}
                manifest_files.append({
                    "file": fname, "image": img, "text": text, "checkpoint": ckpt, "mode": mode,
                    "cbm_path": f"synthetic/{img}__{text}__{ckpt}.cbm", "cbm_sha256": "0" * 64,
                    "tree_count": 2989 if ckpt == "pgs" else 1500,
                    "metrics": {"val": {}, "test": metrics}, "expected_test": exp,
                    "reproduces_manuscript": None if exp is None else True,
                })
    served = frames[(images[0], "pgs", "argmax")]
    pgs = frames[(images[0], "pgs", "pgs")]
    t = served["split"] == "test"
    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "catboost_version": "synthetic", "numpy_version": np.__version__, "virtual_ensembles": m,
        "pooling": "linear (mean of per-member softmax)", "n_val": n_val, "n_test": n_test,
        "files": manifest_files,
        "top1_disagreements_pgs_vs_argmax": int((served.loc[t, "pred"].to_numpy() != pgs.loc[t, "pred"].to_numpy()).sum()),
        "synthetic": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-val", type=int, default=9266)
    p.add_argument("--n-test", type=int, default=9266)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    man = make_synthetic_preds(a.out, a.n_val, a.n_test, a.seed)
    print(f"wrote {len(man['files'])} files to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
