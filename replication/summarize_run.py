"""Collect the facts of the replication run into one JSON used to write Section 3.4.

Reads data_summary.json, download_report.json, embeddings/extraction_report.json,
models/matrix_results.csv and the stage-1 manifest; recomputes top-1 metrics per
(pair, checkpoint, mode) from the exported per-sample CSVs so every number in the manuscript
comes from the same files that the analysis used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common import DEFAULT_ARTIFACTS, REVISI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--preds", default=str(REVISI / "_replication" / "per_sample_predictions"))
    ap.add_argument("--out", default=str(REVISI / "_replication" / "replication_summary.json"))
    args = ap.parse_args()
    art, preds = Path(args.artifacts), Path(args.preds)

    data = json.loads((art / "data_summary.json").read_text(encoding="utf-8"))
    download = json.loads((art / "download_report.json").read_text(encoding="utf-8"))
    extraction = json.loads((art / "embeddings" / "extraction_report.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(art / "models" / "matrix_results.csv").to_dict("records")
    manifest = json.loads((preds / "manifest.json").read_text(encoding="utf-8"))

    top1 = {}
    for f in manifest["files"]:
        df = pd.read_csv(preds / f["file"])
        test = df[df["split"] == "test"]
        key = f"{f['image']}__{f['text']}__{f['checkpoint']}__{f['mode']}"
        top1[key] = {
            "accuracy": accuracy_score(test["label"], test["pred"]),
            "macro_f1": f1_score(test["label"], test["pred"], average="macro", labels=range(9), zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(test["label"], test["pred"]),
            "n_test": int(len(test)),
            "tree_count": f.get("tree_count"),
        }
        if f["mode"] == "pgs":
            top1[key]["mean_prob_std"] = float(test["prob_std"].mean())
            top1[key]["mean_mi"] = float(test["mi"].mean())
    dep = "dinov3_large__mE5_large"
    a, b = f"{dep}__pgs__argmax", f"{dep}__pgs__pgs"
    disagree = None
    if a in top1 and b in top1:
        da = pd.read_csv(preds / f"preds__{a}.csv.gz")
        db = pd.read_csv(preds / f"preds__{b}.csv.gz")
        m = da[da.split == "test"].merge(db[db.split == "test"], on="row_id", suffixes=("_a", "_b"))
        disagree = int((m["pred_a"] != m["pred_b"]).sum())

    summary = {
        "data": data,
        "download": {k: v for k, v in download.items() if k != "failures"},
        "extraction": extraction,
        "training": matrix,
        "top1_test": top1,
        "top1_disagreements_deployed_pgs_vs_argmax": disagree,
        "catboost_version": manifest.get("catboost_version"),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in top1.items()}, indent=2, default=float))
    print("disagreements:", disagree)


if __name__ == "__main__":
    main()
