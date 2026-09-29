"""Download the images referenced by metadata_clean.csv from the project's Drive mirror.

Resumable (existing non-empty, decodable files are skipped), retries with backoff on
throttling, verifies every file with Pillow and never leaves partial files behind.
Fallback Drive IDs come from the full index shipped in crm_jakarta_multimodal_compact_1845.zip.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from common import COMPACT_ZIP, DEFAULT_ARTIFACTS

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fallback_ids() -> dict[str, list[str]]:
    if not COMPACT_ZIP.exists():
        return {}
    with zipfile.ZipFile(COMPACT_ZIP) as z:
        idx = pd.read_csv(io.BytesIO(z.read("source/metadata_full_index.csv")),
                          usecols=["gambar", "drive_file_ids"])
    out: dict[str, list[str]] = {}
    for g, ids in zip(idx["gambar"], idx["drive_file_ids"].fillna("")):
        out.setdefault(g, []).extend(i for i in str(ids).split("|") if i)
    return out


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return path.stat().st_size > 0
    except Exception:
        return False


def fetch(target: Path, ids: list[str], retries: int = 5) -> str:
    """Never raises: one bad file (e.g. WinError 1450 under resource pressure) must not stop the run."""
    for attempt in range(3):
        try:
            return _fetch(target, ids, retries)
        except OSError as e:
            tmp = target.with_name(target.name + ".part")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(5 * (attempt + 1))
            last = f"os_error_{getattr(e, 'winerror', None) or e.errno}"
    return last


def _fetch(target: Path, ids: list[str], retries: int = 5) -> str:
    if target.is_file() and target.stat().st_size > 0 and valid_image(target):
        return "skipped"
    if not ids:
        return "missing_id"
    last = "unavailable"
    for fid in dict.fromkeys(ids):  # unique, order kept
        for attempt in range(retries):
            try:
                r = session().get(URL.format(fid), timeout=(15, 120))
            except requests.RequestException:
                time.sleep(2 ** attempt + random.random())
                last = "network_error"
                continue
            if r.status_code in (404, 410):  # file no longer on Drive (all 2026 uploads) - final
                last = f"http_{r.status_code}"
                break
            if r.status_code == 200 and not r.headers.get("content-type", "").startswith("text/html"):
                try:
                    with Image.open(io.BytesIO(r.content)) as im:
                        im.verify()
                except Exception:
                    last = "not_an_image"
                    break
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(target)
                return "downloaded"
            if r.status_code in (429, 500, 502, 503, 504) or r.headers.get("content-type", "").startswith("text/html"):
                time.sleep(min(60, 2 ** attempt * 2) + random.random())  # throttled / quota page
                last = f"throttled_{r.status_code}"
                continue
            last = f"http_{r.status_code}"
            break
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    root = Path(args.artifacts) / "crm_jakarta"
    df = pd.read_csv(root / "metadata_clean.csv", low_memory=False)
    fb = fallback_ids()
    jobs = []
    for g, fid in zip(df["gambar"], df["drive_file_id"].fillna("")):
        ids = [i for i in str(fid).split("|") if i] + fb.get(g, [])
        jobs.append((root / g, ids))
    # identical image paths appear in several rows: download each path once
    uniq = list({str(t): (t, ids) for t, ids in jobs}.values())
    print(f"{len(df):,} rows, {len(uniq):,} unique image paths, workers={args.workers}", flush=True)
    counts: dict[str, int] = {}
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, ((target, ids), status) in enumerate(zip(uniq, ex.map(lambda j: fetch(*j), uniq)), 1):
            counts[status] = counts.get(status, 0) + 1
            if status not in ("skipped", "downloaded"):
                failures.append({"path": str(target.relative_to(root)), "status": status})
            if n % 500 == 0 or n == len(uniq):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"{n:,}/{len(uniq):,} {counts} {rate:.1f}/s", flush=True)
    report = {"rows": len(df), "unique_paths": len(uniq), "counts": counts, "failures": failures,
              "seconds": round(time.time() - t0, 1)}
    (Path(args.artifacts) / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
