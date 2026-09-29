"""Replicate notebooks 02 (cleaning) and 03 (stratified split) on the available snapshot.

The thesis snapshot (88,302 raw rows -> 61,773 pairs) is no longer retrievable; the local
`metadata.csv` is the later Drive mirror (86,888 raw rows, 2023-01 .. 2026-04).  The logic
below is copied from the notebooks unchanged, so only the input snapshot differs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from common import DEFAULT_ARTIFACTS, DEFAULT_METADATA, import_crm

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
# download_images.py statuses that mean the file is gone for good (Drive 404/410, no Drive id,
# served bytes are not an image); anything else may succeed on a retry
PERMANENT_FAILURES = {"http_404", "http_410", "missing_id", "not_an_image"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--out", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--require-images", action="store_true",
                    help="drop rows whose image failed to download (run after download_images.py)")
    ap.add_argument("--allow-transient-failures", action="store_true",
                    help="with --require-images: also drop rows whose download failed for a possibly "
                         "transient reason (throttling, network, 5xx, 403) instead of aborting")
    args = ap.parse_args()
    crm = import_crm()
    from pathlib import Path

    out = Path(args.out)
    (out / "crm_jakarta").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    # ---- notebook 02: normalise SKPD -> 9 classes, drop empties, exact dedup on laporan
    raw = pd.read_csv(args.metadata, low_memory=False)
    raw["label"] = raw["label_skpd"].apply(crm.normalize_skpd)
    keep = ["gambar", "laporan", "label", "drive_file_id", "code", "createdAt"]
    clean = raw[keep].copy()
    clean = clean.dropna(subset=["gambar", "laporan", "label"])
    clean["laporan"] = clean["laporan"].astype(str).str.strip()
    clean = clean[clean["laporan"].str.len() > 0].reset_index(drop=True)
    before = len(clean)
    clean = clean.drop_duplicates(subset="laporan").reset_index(drop=True)
    after_dedup = len(clean)
    unavailable = 0
    if args.require_images:
        # Section 2.1 of the manuscript: rows whose image cannot be read are discarded before splitting.
        report = json.loads((out / "download_report.json").read_text(encoding="utf-8"))
        # Only permanent failures define "image unavailable"; a throttled/network failure would make
        # the dataset (and the split) depend on download luck -> re-run download_images.py first.
        transient = Counter(f["status"] for f in report["failures"] if f["status"] not in PERMANENT_FAILURES)
        if transient and not args.allow_transient_failures:
            raise SystemExit(f"download_report.json has possibly transient failures {dict(transient)}; "
                             "re-run download_images.py (it resumes) or pass --allow-transient-failures")
        bad = {f["path"].replace("\\", "/") for f in report["failures"]}
        ok = ~clean["gambar"].isin(bad) & clean["gambar"].map(lambda g: (out / "crm_jakarta" / g).is_file())
        unavailable = int((~ok).sum())
        clean = clean[ok].reset_index(drop=True)
    clean.to_csv(out / "crm_jakarta" / "metadata_clean.csv", index=False)

    # ---- notebook 03: stratified 70/15/15, seed 42
    df = clean
    df["label_id"] = df["label"].map(crm.LABEL2ID)
    df["row_id"] = np.arange(len(df))
    assert df["label_id"].notna().all()
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    trainval_idx, test_idx = next(sss1.split(df, df["label_id"]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC),
                                  random_state=SEED)
    train_idx, val_idx = next(sss2.split(trainval_df, trainval_df["label_id"]))
    splits = {
        "train": trainval_df.iloc[train_idx].reset_index(drop=True),
        "val": trainval_df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }
    cols = ["row_id", "gambar", "laporan", "label", "label_id"]
    for name, part in splits.items():
        part[cols].to_csv(out / "splits" / f"{name}.csv", index=False)

    dist = pd.DataFrame({k: v["label"].value_counts(normalize=True).reindex(crm.TARGET_CLASSES)
                         for k, v in {"full": df, **splits}.items()})
    summary = {
        "snapshot": str(args.metadata),
        "raw_rows": int(len(raw)),
        "raw_skpd_labels": int(raw["label_skpd"].nunique()),
        "rows_before_dedup": int(before),
        "rows_after_dedup": int(after_dedup),
        "rows_dropped_image_unavailable": unavailable,
        "clean_rows": int(len(df)),
        "created_at_range": [str(df["createdAt"].min()), str(df["createdAt"].max())],
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "class_counts": {k: v["label"].value_counts().reindex(crm.TARGET_CLASSES).astype(int).tolist()
                         for k, v in {"full": df, **splits}.items()},
        "max_cross_split_proportion_diff": float((dist.max(axis=1) - dist.min(axis=1)).max()),
        "classes": crm.TARGET_CLASSES,
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
