"""Shared paths for the replication run. Every path can be overridden with an environment variable.

CRM_SRC          smartCityReport/src (the thesis `crm` package: classes, SKPD mapping, fusion, encoders)
REPL_METADATA    raw metadata.csv of the CRM Drive mirror
REPL_ARTIFACTS   working tree for images, splits, embeddings, checkpoints (default: ./_replication/artifacts)
REPL_EXPORT_ZIP  optional crm_jakarta_multimodal_compact_1845.zip (fallback Drive ids)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISI = HERE.parent  # repository root (export/analyze/fill scripts live here)
THESIS = Path(os.environ.get("THESIS_DIR", REVISI.parent))
SMARTCITY = THESIS / "smartCityReport"
CRM_SRC = Path(os.environ.get("CRM_SRC", SMARTCITY / "src"))
DEFAULT_METADATA = Path(os.environ.get("REPL_METADATA", SMARTCITY / "artifacts" / "crm_jakarta" / "metadata.csv"))
DEFAULT_ARTIFACTS = Path(os.environ.get("REPL_ARTIFACTS", REVISI / "_replication" / "artifacts"))
COMPACT_ZIP = Path(os.environ.get("REPL_EXPORT_ZIP", THESIS / "crm_jakarta_multimodal_compact_1845.zip"))


def import_crm():
    """Make the thesis `crm` package importable (class list, SKPD mapping, fusion, encoders)."""
    if not (CRM_SRC / "crm" / "__init__.py").exists():
        raise SystemExit(f"crm package not found under {CRM_SRC}; clone smartCityReport and set CRM_SRC")
    if str(CRM_SRC) not in sys.path:
        sys.path.insert(0, str(CRM_SRC))
    import crm  # noqa: F401

    return crm
