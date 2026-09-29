"""Run the whole replication end to end as one detached, resumable process.

Stages (each writes _replication/stages/<name>.done; re-running skips finished stages):
  models   download the four encoder checkpoints (safetensors) into HF_HUB_CACHE
  images   download all images; repeat while failures look transient (max 4 passes)
  prepare  notebook 02/03 logic + drop rows whose image is permanently unavailable
  embed    frozen embeddings (mE5-L, DINOv3-L, DINOv2-L, EVA-02-L) on the GPU
  train    CatBoost baseline + PGS heads for the three image encoders x mE5-L
  export   stage 1 of the analysis: per-sample predictions (argmax + PGS M=30)
  analyze  stage 2: calibration, paired tests, selective prediction, figures, text
  summary  replication_summary.json used to write Section 3.4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import CRM_SRC, DEFAULT_ARTIFACTS

HERE = Path(__file__).resolve().parent
REVISI = HERE.parent
REP = REVISI / "_replication"
ART = DEFAULT_ARTIFACTS
STAGES = REP / "stages"
LOG = REP / "logs" / "pipeline.log"
PY = sys.executable
PERMANENT = {"http_404", "http_410", "missing_id", "not_an_image"}

ENV = dict(os.environ)
ENV.update({
    # HF_HUB_CACHE: inherited from the environment if set (a disk with room for ~6 GB of weights)
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    # one plain HTTP connection, one file at a time: the parallel xet/image downloads exhausted
    # system resources (WinError 1450, xet MemoryError) and restarted the PC
    "HF_HUB_DISABLE_XET": "1",
    "PYTHONIOENCODING": "utf-8",
})


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], name: str, offline: bool = False) -> None:
    env = dict(ENV)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    out = REP / "logs" / f"{name}.log"
    log(f"RUN {name}: {' '.join(args)}")
    with out.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run(args, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}; see {out}")


def stage(name: str):
    def deco(fn):
        def wrapper():
            marker = STAGES / f"{name}.done"
            if marker.exists():
                log(f"SKIP {name} (done {marker.read_text(encoding='utf-8').strip()})")
                return
            t0 = time.time()
            fn()
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            log(f"DONE {name} in {(time.time() - t0) / 60:.1f} min")
        return wrapper
    return deco


@stage("models")
def models():
    run([PY, "-u", "download_models.py"], "models")
    last_run = (REP / "logs" / "models.log").read_text(encoding="utf-8", errors="replace").split("\n===== ")[-1]
    ok = [ln for ln in last_run.splitlines() if ln.startswith("OK ")]
    if len(ok) != 4 or "ALL_DONE" not in last_run:
        raise RuntimeError(f"not all encoder checkpoints downloaded ({len(ok)}/4 OK)")


@stage("images")
def images():
    for attempt in range(4):
        run([PY, "-u", "download_images.py", "--workers", "2"], "images")
        report = json.loads((ART / "download_report.json").read_text(encoding="utf-8"))
        transient = [f for f in report["failures"] if f["status"] not in PERMANENT]
        log(f"images pass {attempt + 1}: {report['counts']}; transient failures: {len(transient)}")
        if not transient:
            return
        time.sleep(60)


@stage("prepare")
def prepare():
    report = json.loads((ART / "download_report.json").read_text(encoding="utf-8"))
    transient = any(f["status"] not in PERMANENT for f in report["failures"])
    run([PY, "-u", "prepare_data.py", "--require-images"] + (["--allow-transient-failures"] if transient else []),
        "prepare")


@stage("embed")
def embed():
    run([PY, "-u", "extract_embeddings.py", "--workers", "2"], "embed", offline=True)


@stage("train")
def train():
    run([PY, "-u", "train_catboost.py", "--cb-only-deployed"], "train")


@stage("export")
def export():
    run([PY, "-u", str(REVISI / "export_per_sample_predictions.py"), "--artifacts", str(ART),
         "--out", str(REP / "per_sample_predictions"), "--no-expect",
         "--crm-src", str(CRM_SRC)], "export")


@stage("analyze")
def analyze():
    run([PY, "-u", str(REVISI / "analyze_uncertainty.py"), "--preds", str(REP / "per_sample_predictions"),
         "--out", str(REP / "results"), *(["--manuscript", os.environ["MANUSCRIPT_DOCX"]] if os.environ.get("MANUSCRIPT_DOCX") else [])],
        "analyze")


@stage("summary")
def summary():
    run([PY, "-u", "summarize_run.py"], "summary")


def main() -> None:
    STAGES.mkdir(parents=True, exist_ok=True)
    (REP / "logs").mkdir(parents=True, exist_ok=True)
    log(f"pipeline start (pid {os.getpid()})")
    try:
        for fn in (models, images, prepare, embed, train, export, analyze, summary):
            fn()
    except Exception as e:  # noqa: BLE001
        log(f"PIPELINE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
