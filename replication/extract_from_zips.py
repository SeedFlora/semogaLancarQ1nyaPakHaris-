"""Extract the images listed in metadata_clean.csv from the Google Drive export zips (no network).

Usage: python extract_from_zips.py "/path/to/Thesis Data-*.zip"
Existing decodable files are kept; every extracted file is verified with Pillow and written
atomically. A zip that cannot be opened (e.g. still downloading) is reported and skipped.
"""

from __future__ import annotations

import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from common import DEFAULT_ARTIFACTS

PREFIX = "Thesis Data/"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pattern = sys.argv[1]
    root = Path(DEFAULT_ARTIFACTS) / "crm_jakarta"
    need = set(pd.read_csv(root / "metadata_clean.csv", low_memory=False)["gambar"])
    report = {"zips": {}, "unreadable": []}
    for zp in sorted(glob.glob(pattern)):
        try:
            z = zipfile.ZipFile(zp)
        except (zipfile.BadZipFile, OSError) as e:
            report["zips"][Path(zp).name] = f"cannot open: {e}"
            print(f"SKIP {Path(zp).name}: {e}", flush=True)
            continue
        added = present = 0
        with z:
            for info in z.infolist():
                g = info.filename.removeprefix(PREFIX)
                if g not in need:
                    continue
                target = root / g
                if target.is_file() and target.stat().st_size > 0:
                    present += 1
                    continue
                data = z.read(info)
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        im.verify()
                except Exception:  # noqa: BLE001
                    report["unreadable"].append(g)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(target)
                added += 1
        report["zips"][Path(zp).name] = {"added": added, "already_present": present}
        print(f"{Path(zp).name}: added {added:,}, already present {present:,}", flush=True)
    have = sum((root / g).is_file() for g in need)
    report.update({"needed": len(need), "available": have, "missing": sorted(g for g in need if not (root / g).is_file())})
    (Path(DEFAULT_ARTIFACTS) / "zip_extract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"available {have:,}/{len(need):,} ({have / len(need):.1%}); missing {len(need) - have:,}", flush=True)


if __name__ == "__main__":
    main()
