#!/usr/bin/env python3
"""Build a small, realistic, fully synthetic ``artifacts/`` tree for testing stage 1.

Layout mirrors the RunPod thesis artifacts (CONTRACT.md section 1)::

    <out>/embeddings/image/{dinov3_large,eva02_large,dinov2_large}.npy   (N_all, dim) float32
    <out>/embeddings/text/mE5_large.npy                                  (N_all, dim) float32
    <out>/splits/{train,val,test}.csv          row_id,gambar,laporan,label,label_id  (70/15/15)
    <out>/models/checkpoints/{img}__{txt}__{cb|pgs}.cbm
    <out>/fixture_info.json                     generation parameters + fixture accuracies

Data: 9 imbalanced classes; every modality carries class signal plus noise. A per-sample
difficulty factor is shared by both modalities (so uncertainty tracks errors), each
modality is sometimes "confused" towards another class (so fusion helps), and a few
samples are mislabelled (irreducible error). The three image encoders share one latent
image representation with encoder-specific rotation/noise, so their errors correlate the
way real encoders do. Rows are rescaled randomly so the per-modality L2 step matters, and
~1% of the image rows of every split are all-zero (failed image loads, as in the real
caches).

Models use the notebooks' ``_fit_catboost`` configuration (class weights
``N / (C * N_l)``, MultiClass, seed 42, early stopping 50 / 200 with the val split as
eval set, PGS = ``posterior_sampling=True`` with twice the iterations) but few iterations
and a small depth so the whole build runs in seconds on a CPU. One deliberate deviation:
the learning rate defaults to 0.25 instead of 0.05, because 80/160 iterations at 0.05
leave the model far from converged and its probabilities unrealistically flat (mean
confidence ~0.4 at ~0.76 accuracy); ``--learning-rate 0.05`` restores the exact notebook
value. Everything is seeded and deterministic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
#: Imbalanced class prior (roughly the shape of the real CRM label distribution).
CLASS_PRIOR = np.array([0.26, 0.18, 0.14, 0.12, 0.09, 0.08, 0.06, 0.04, 0.03])

#: image encoder -> (encoder-specific noise sd, probability the image view is confused)
IMAGE_ENCODERS: dict[str, tuple[float, float]] = {
    "dinov3_large": (0.55, 0.20),
    "eva02_large": (0.60, 0.21),
    "dinov2_large": (0.75, 0.23),
}
TEXT_ENCODERS: dict[str, tuple[float, float]] = {"mE5_large": (0.50, 0.14)}
DEFAULT_PAIRS = tuple(f"{img}__mE5_large" for img in IMAGE_ENCODERS)

IMAGE_SIGNAL = 0.70  # class-centroid scale of the shared image latent (noise sd = 1)
TEXT_SIGNAL = 1.00
LABEL_NOISE = 0.05  # fraction of samples whose label disagrees with both views
DEFAULT_LEARNING_RATE = 0.25  # notebooks: 0.05 with 1500 / 3000 iterations
DEFAULT_OUT = Path(__file__).resolve().parent / "_work" / "fixture" / "artifacts"


def _random_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q * np.sign(np.diag(r))


def _confuse(rng: np.random.Generator, y: np.ndarray, prob: float) -> np.ndarray:
    """Return the class each sample 'looks like': y, or with ``prob`` another class."""
    shift = rng.integers(1, N_CLASSES, size=y.shape[0])
    return np.where(rng.random(y.shape[0]) < prob, (y + shift) % N_CLASSES, y)


def make_embeddings(
    n_samples: int, dim: int, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Labels plus image/text embedding caches (row i = sample i)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(N_CLASSES, size=n_samples, p=CLASS_PRIOR / CLASS_PRIOR.sum())
    # true "content" class; a few labels disagree with the content (annotation noise)
    content = _confuse(rng, y, LABEL_NOISE)
    difficulty = rng.uniform(0.25, 1.25, size=n_samples)[:, None]  # shared by modalities
    scale = np.exp(rng.normal(0.0, 0.35, size=(n_samples, 1)))  # nuisance row norm

    img_centroids = rng.normal(size=(N_CLASSES, dim))
    image: dict[str, np.ndarray] = {}
    img_latent_noise = rng.normal(size=(n_samples, dim))
    for name, (noise_sd, confuse_p) in IMAGE_ENCODERS.items():
        # encoder-specific confusions on top of a shared latent image view
        seen = _confuse(rng, content, confuse_p)
        latent = IMAGE_SIGNAL * difficulty * img_centroids[seen] + img_latent_noise
        emb = latent @ _random_rotation(rng, dim) + noise_sd * rng.normal(size=(n_samples, dim))
        emb += rng.normal(0.0, 0.5, size=(1, dim))  # encoder-specific offset
        image[name] = (scale * emb).astype(np.float32)

    txt_centroids = rng.normal(size=(N_CLASSES, dim))
    text: dict[str, np.ndarray] = {}
    for name, (noise_sd, confuse_p) in TEXT_ENCODERS.items():
        seen = _confuse(rng, content, confuse_p)
        latent = TEXT_SIGNAL * difficulty * txt_centroids[seen] + rng.normal(size=(n_samples, dim))
        emb = latent @ _random_rotation(rng, dim) + noise_sd * rng.normal(size=(n_samples, dim))
        emb += rng.normal(0.0, 0.5, size=(1, dim))
        text[name] = (np.exp(rng.normal(0.0, 0.35, size=(n_samples, 1))) * emb).astype(np.float32)
    return y, image, text


def make_splits(y: np.ndarray, seed: int) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified split (sklearn, ``random_state=seed``), file order = shuffled.

    Columns follow notebook 03 (``row_id, gambar, laporan, label, label_id``). ``laporan``
    is free text with commas, double quotes and line breaks, like the real reports, so the
    readers are exercised on quoted multi-line CSV fields and must select columns by name.
    """
    from sklearn.model_selection import train_test_split

    row_id = np.arange(y.shape[0])
    tr, rest = train_test_split(row_id, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return {
        name: pd.DataFrame({
            "row_id": ids.astype(np.int64),
            "gambar": [f"images/{i:06d}.jpg" for i in ids],
            "laporan": [f'Laporan {i}: jalan rusak, "berlubang"\nmohon segera ditindak'
                        for i in ids],
            "label": [TARGET_CLASSES[k] for k in y[ids]],
            "label_id": y[ids].astype(np.int64),
        })
        for name, ids in (("train", tr), ("val", va), ("test", te))
    }


def zero_failed_images(
    image: dict[str, np.ndarray], splits: dict[str, pd.DataFrame], seed: int,
    fraction: float = 0.01,
) -> list[int]:
    """Zero a few image rows in every split, for every image encoder.

    Notebooks 04/07/08 leave the embedding of an image that fails to load at zeros, so the
    real caches contain all-zero rows; the per-modality L2 step must map them to zeros
    (``x / clip(0, 1e-9)``). Uses its own RNG so the embeddings themselves do not change.
    """
    rng = np.random.default_rng(seed + 1)
    rows: list[int] = []
    for df in splits.values():
        ids = df["row_id"].to_numpy()
        k = max(1, int(round(fraction * len(ids))))
        rows += sorted(int(i) for i in rng.choice(ids, size=k, replace=False))
    for arr in image.values():
        arr[rows] = 0.0
    return rows


def fuse(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """Notebook early fusion: per-modality L2 (eps 1e-9) + concat, float32."""
    def l2(x: np.ndarray) -> np.ndarray:
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)

    return np.concatenate([l2(img), l2(txt)], axis=1).astype(np.float32)


def class_weight_list(y: np.ndarray) -> list[float]:
    """Paper formula ``w_l = N / (C * N_l)`` (``crm.fusion.class_weights``)."""
    counts = np.bincount(y, minlength=N_CLASSES)
    return [len(y) / (N_CLASSES * max(int(c), 1)) for c in counts]


def fit_catboost(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    w_list: list[float], iterations: int, depth: int, posterior_sampling: bool,
    min_trees: int, learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tuple[Any, dict[str, Any]]:
    """Notebook ``_fit_catboost`` config; refit without shrinking if too few trees remain.

    ``virtual_ensembles_predict`` with M members needs >= 2M+1 trees. Early stopping +
    use_best_model may shrink a tiny synthetic model below that; in that case the model is
    refit keeping all iterations (recorded in ``fixture_info.json``).
    """
    from catboost import CatBoostClassifier

    params = dict(
        depth=depth, learning_rate=learning_rate, task_type="CPU", thread_count=-1,
        class_weights=w_list, loss_function="MultiClass", verbose=False, random_seed=42,
        posterior_sampling=posterior_sampling,
        iterations=iterations * 2 if posterior_sampling else iterations,
        allow_writing_files=False,
    )
    es = 200 if posterior_sampling else 50
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=es)
    info: dict[str, Any] = {"iterations": params["iterations"], "early_stopping_rounds": es,
                            "best_iteration": model.get_best_iteration(),
                            "refit_without_shrinking": False}
    if model.tree_count_ < min_trees:
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=False)
        info["refit_without_shrinking"] = True
    info["tree_count"] = int(model.tree_count_)
    return model, info


def build_fixture(
    out_dir: Path | str = DEFAULT_OUT,
    *,
    n_samples: int = 1800,
    dim: int = 64,
    iterations: int = 80,
    depth: int = 4,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 42,
    pairs: Sequence[str] = DEFAULT_PAIRS,
    virtual_ensembles: int = 30,
    clean: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Create the fixture tree under ``out_dir``; return the ``fixture_info`` dict."""
    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.time()
    out = Path(out_dir)
    if clean:
        for sub in ("embeddings", "splits", "models"):
            shutil.rmtree(out / sub, ignore_errors=True)
    for sub in ("embeddings/image", "embeddings/text", "splits", "models/checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    y, image, text = make_embeddings(n_samples, dim, seed)
    splits = make_splits(y, seed)
    zero_rows = zero_failed_images(image, splits, seed)
    for name, arr in image.items():
        np.save(out / "embeddings" / "image" / f"{name}.npy", arr)
    for name, arr in text.items():
        np.save(out / "embeddings" / "text" / f"{name}.npy", arr)
    for name, df in splits.items():
        df.to_csv(out / "splits" / f"{name}.csv", index=False, lineterminator="\n")

    ids = {k: v["row_id"].to_numpy() for k, v in splits.items()}
    labels = {k: v["label_id"].to_numpy() for k, v in splits.items()}
    w_list = class_weight_list(labels["train"])
    min_trees = 2 * virtual_ensembles + 1
    info: dict[str, Any] = {
        "n_samples": n_samples, "dim": dim, "seed": seed, "iterations_cb": iterations,
        "iterations_pgs": 2 * iterations, "depth": depth, "learning_rate": learning_rate,
        "min_trees": min_trees,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "class_counts": np.bincount(y, minlength=N_CLASSES).tolist(),
        "zero_image_rows": zero_rows,
        "models": {},
    }
    for pair in pairs:
        img_name, txt_name = pair.split("__")
        X = {k: fuse(image[img_name][ids[k]], text[txt_name][ids[k]]) for k in ids}
        for ckpt, ps in (("cb", False), ("pgs", True)):
            model, fit_info = fit_catboost(
                X["train"], labels["train"], X["val"], labels["val"], w_list,
                iterations, depth, ps, min_trees, learning_rate,
            )
            model.save_model(str(out / "models" / "checkpoints" / f"{pair}__{ckpt}.cbm"))
            pred = np.asarray(model.predict_proba(X["test"])).argmax(axis=1)
            fit_info["test_accuracy_argmax"] = float(accuracy_score(labels["test"], pred))
            fit_info["test_macro_f1_argmax"] = float(
                f1_score(labels["test"], pred, labels=list(range(N_CLASSES)),
                         average="macro", zero_division=0))
            info["models"][f"{pair}__{ckpt}"] = fit_info
            if verbose:
                print(f"  {pair}__{ckpt}: trees={fit_info['tree_count']} "
                      f"test acc={fit_info['test_accuracy_argmax']:.4f} "
                      f"F1={fit_info['test_macro_f1_argmax']:.4f}"
                      + (" (refit, no shrink)" if fit_info["refit_without_shrinking"] else ""),
                      flush=True)
    info["build_seconds"] = round(time.time() - t0, 2)
    (out / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"fixture written to {out.resolve()} in {info['build_seconds']}s", flush=True)
    return info


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="artifacts directory to create")
    p.add_argument("--n-samples", type=int, default=1800)
    p.add_argument("--dim", type=int, default=64, help="embedding dim of every encoder")
    p.add_argument("--iterations", type=int, default=80,
                   help="cb iterations (pgs uses twice as many, as in the notebooks)")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                   help="notebooks use 0.05 with 1500/3000 iterations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), metavar="IMAGE__TEXT")
    p.add_argument("--virtual-ensembles", type=int, default=30,
                   help="M the checkpoints must support (>= 2M+1 trees)")
    args = p.parse_args(argv)
    for pair in args.pairs:
        img, _, txt = pair.partition("__")
        if img not in IMAGE_ENCODERS or txt not in TEXT_ENCODERS:
            p.error(f"unknown pair {pair!r}; image in {list(IMAGE_ENCODERS)}, "
                    f"text in {list(TEXT_ENCODERS)}")
    build_fixture(args.out, n_samples=args.n_samples, dim=args.dim, iterations=args.iterations,
                  depth=args.depth, learning_rate=args.learning_rate, seed=args.seed,
                  pairs=args.pairs, virtual_ensembles=args.virtual_ensembles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
