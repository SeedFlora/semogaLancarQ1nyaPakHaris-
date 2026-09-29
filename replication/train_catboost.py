"""Train the baseline and PGS CatBoost heads exactly as notebook 08 (`train_and_eval`).

Per pair: early fusion (per-modality L2 + concat, crm.fusion.early_fusion), class weights
w_l = N / (C * N_l) (crm.fusion.class_weights), CatBoost MultiClass, depth 6, lr 0.05, seed 42,
CPU; baseline 1,500 iterations with 50-round early stopping on the validation loss; PGS
(posterior_sampling=True) 3,000 iterations with 200-round early stopping.  Checkpoints are
saved as models/checkpoints/{img}__{txt}__{cb,pgs}.cbm, the layout the export script reads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import DEFAULT_ARTIFACTS, import_crm

DEFAULT_PAIRS = ["dinov3_large__mE5_large", "eva02_large__mE5_large", "dinov2_large__mE5_large"]


def pgs_predict_proba(model, X, n_virtual_ensembles: int = 30):
    """Notebook 08 implementation: per-member softmax, linear pooling, mean across-member SD."""
    preds = np.asarray(model.virtual_ensembles_predict(
        X, prediction_type="VirtEnsembles", virtual_ensembles_count=n_virtual_ensembles))
    preds_exp = np.exp(preds - preds.max(axis=-1, keepdims=True))
    probs_per_ens = preds_exp / preds_exp.sum(axis=-1, keepdims=True)
    return probs_per_ens.mean(axis=1), probs_per_ens.std(axis=1).mean(axis=1)


def fit(X_tr, y_tr, X_va, y_va, w_list, iterations, depth, posterior_sampling, threads, snapshot=None):
    params = dict(depth=depth, learning_rate=0.05, task_type="CPU", thread_count=threads,
                  class_weights=w_list, loss_function="MultiClass", verbose=False,
                  random_seed=42, posterior_sampling=posterior_sampling)
    params["iterations"] = iterations * 2 if posterior_sampling else iterations
    model = CatBoostClassifier(**params)
    t0 = time.time()
    # CatBoost snapshots: an interrupted fit resumes from the last snapshot (every 5 min) with the
    # same result as an uninterrupted run; the snapshot is removed once the model is saved.
    extra = {} if snapshot is None else dict(save_snapshot=True, snapshot_file=str(snapshot), snapshot_interval=300)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=200 if posterior_sampling else 50, **extra)
    return model, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--virtual-ensembles", type=int, default=30)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--deployed", default=DEFAULT_PAIRS[0])
    ap.add_argument("--cb-only-deployed", action="store_true",
                    help="train the baseline (non-PGS) head only for the deployed pair")
    args = ap.parse_args()
    crm = import_crm()
    from crm.fusion import class_weights, early_fusion

    art = Path(args.artifacts)
    splits = {s: pd.read_csv(art / "splits" / f"{s}.csv") for s in ("train", "val", "test")}
    ckpt_dir = art / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_path = art / "models" / "matrix_results.csv"
    results = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []

    for pair in args.pairs:
        img, txt = pair.split("__")
        # The kurangan note only needs the ablation-matrix mode (CatBoost+PGS) for the comparison
        # encoders; the baseline head is needed for the deployed pair only (Tables 5-6).
        heads = ("cb", "pgs") if (pair == args.deployed or not args.cb_only_deployed) else ("pgs",)
        if all((ckpt_dir / f"{pair}__{k}.cbm").exists() for k in heads):
            print(f"[skip] {pair} checkpoints exist for {heads}", flush=True)
            continue
        img_emb = np.load(art / "embeddings" / "image" / f"{img}.npy", mmap_mode="r")
        txt_emb = np.load(art / "embeddings" / "text" / f"{txt}.npy", mmap_mode="r")

        def feats(split):
            idx = splits[split]["row_id"].values
            return early_fusion(img_emb[idx], txt_emb[idx], l2_per_modality=True)

        X_tr, X_va, X_te = feats("train"), feats("val"), feats("test")
        y_tr, y_va, y_te = (splits[s]["label_id"].values for s in ("train", "val", "test"))
        weights = class_weights(y_tr, num_classes=len(crm.TARGET_CLASSES))
        w_list = [weights[i] for i in range(len(crm.TARGET_CLASSES))]

        print(f"=== {pair}: X_tr {X_tr.shape}, heads {heads}", flush=True)

        def get(kind: str, posterior: bool):
            path = ckpt_dir / f"{pair}__{kind}.cbm"
            if path.exists():  # resume: a finished head is loaded, not retrained
                m = CatBoostClassifier()
                m.load_model(str(path))
                return m, float("nan")
            snap = ckpt_dir.parent / "snapshots" / f"{pair}__{kind}.cbsnapshot"
            snap.parent.mkdir(parents=True, exist_ok=True)
            if snap.exists():
                print(f"  resuming {kind} from snapshot {snap.name}", flush=True)
            m, t = fit(X_tr, y_tr, X_va, y_va, w_list, args.iterations, args.depth, posterior, args.threads, snap)
            m.save_model(str(path))
            snap.unlink(missing_ok=True)
            return m, t

        cb, t_cb, y_cb = None, float("nan"), None
        if "cb" in heads:
            cb, t_cb = get("cb", False)
            y_cb = cb.predict(X_te).flatten().astype(int)
            print(f"  cb : {cb.tree_count_} trees, {t_cb:.0f}s, acc={accuracy_score(y_te, y_cb):.4f}", flush=True)
        pgs, t_pgs = get("pgs", True)
        probs, unc = pgs_predict_proba(pgs, X_te, args.virtual_ensembles)
        y_pgs = probs.argmax(axis=1)
        row = {
            "image": img, "text": txt,
            "acc_catboost": accuracy_score(y_te, y_cb) if cb is not None else float("nan"),
            "f1_catboost": f1_score(y_te, y_cb, average="macro") if cb is not None else float("nan"),
            "acc_pgs": accuracy_score(y_te, y_pgs), "f1_pgs": f1_score(y_te, y_pgs, average="macro"),
            "mean_uncertainty": float(np.mean(unc)),
            "trees_cb": int(cb.tree_count_) if cb is not None else None, "trees_pgs": int(pgs.tree_count_),
            "train_sec_cb": round(t_cb, 1), "train_sec_pgs": round(t_pgs, 1),
        }
        row["delta_acc"], row["delta_f1"] = row["acc_pgs"] - row["acc_catboost"], row["f1_pgs"] - row["f1_catboost"]
        print(f"  pgs: {pgs.tree_count_} trees, {t_pgs:.0f}s, acc={row['acc_pgs']:.4f} f1={row['f1_pgs']:.4f}",
              flush=True)
        results = [r for r in results if not (r["image"] == img and r["text"] == txt)] + [row]
        pd.DataFrame(results).to_csv(results_path, index=False)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
