#!/usr/bin/env python3
"""Stage 3 of the uncertainty-analysis pipeline: fill the manuscript DOCX.

Reads ``fill_values.json``, ``fig_reliability.png`` and ``fig_selective.png``
from the stage-2 results directory (see CONTRACT.md §3–§4) and writes a new
DOCX in which

* every placeholder token ``⟦KEY⟧`` / ``⟦KEY: description⟧`` in the main
  document, headers, footers, footnotes, endnotes and comments is replaced by
  ``fill_values[KEY]``; the yellow ``<w:highlight>`` is removed from a run once
  it no longer contains token material (unless ``--keep-highlight``);
* the images whose ``<wp:docPr name=...>`` is ``PLACEHOLDER_FIG5_RELIABILITY``
  / ``PLACEHOLDER_FIG6_SELECTIVE`` get the new PNG bytes (height re-derived
  from the width when the aspect ratio differs by more than 1 %) and are
  renamed to ``Figure 5`` / ``Figure 6``.

The package is edited at the XML level (zipfile + lxml).  Only the parts that
actually change are re-serialised; every other entry is copied with the same
bytes, in the same order and with the same compression method.  Tracked-change
wrappers (``<w:ins>`` ...) are left untouched because only ``<w:t>`` text and
run properties are edited; deleted text (``<w:delText>``, and ``<w:t>`` inside
``<w:del>`` / ``<w:moveFrom>``) is never modified.

Tokens are expected to sit inside a single ``<w:t>``; as a safeguard, tokens
split across several runs of one paragraph are detected on the concatenated
paragraph text and replaced as well (the value goes into the first run).
Token-like fragments that cannot be resolved (and tokens in equations or
DrawingML text) are reported as errors; stray brackets that do not look like a
token, such as the notation "⟦...⟧" in an author comment, only as warnings.

Exit codes: 0 = complete; 1 = incomplete (tokens/figures left unfilled without
``--allow-missing``, or errors such as malformed tokens); 2 = usage error
(bad paths, output would overwrite input, unreadable inputs).

Example::

    python fill_manuscript.py --docx SmartCitty_IJOST_Rev_1.docx \\
        --results results/ --out SmartCitty_IJOST_Rev_1_filled.docx
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import struct
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

REL_TYPE_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_TYPE_OFFICE_DOC = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
# Relationship types (suffixes) of parts that carry document text.
STORY_REL_SUFFIXES = ("/header", "/footer", "/footnotes", "/endnotes", "/comments")
SVG_BLIP_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


W_T, W_R, W_P, W_RPR, W_PPR, W_HIGHLIGHT = (
    _w("t"), _w("r"), _w("p"), _w("rPr"), _w("pPr"), _w("highlight"))
W_DRAWING = _w("drawing")
W_DEL, W_MOVEFROM = _w("del"), _w("moveFrom")
W_DELTEXT, W_DELINSTRTEXT = _w("delText"), _w("delInstrText")
MC_FALLBACK = f"{{{MC_NS}}}Fallback"
# Text-bearing elements outside <w:t> that Word renders as visible text; a token
# there cannot be filled by this script and must be reported, not skipped.
_VISIBLE_UNSUPPORTED_TEXT = frozenset([
    "{http://schemas.openxmlformats.org/officeDocument/2006/math}t",   # m:t (equations)
    f"{{{A_NS}}}t",                                                    # a:t (DrawingML text)
])
WP_DOCPR, WP_EXTENT = f"{{{WP_NS}}}docPr", f"{{{WP_NS}}}extent"
A_BLIP, A_EXT, A_SRCRECT = f"{{{A_NS}}}blip", f"{{{A_NS}}}ext", f"{{{A_NS}}}srcRect"
PIC_CNVPR = f"{{{PIC_NS}}}cNvPr"
R_EMBED = f"{{{R_NS}}}embed"

# Elements that break the visible text flow of a paragraph (or hide text):
# a token can never legitimately span them.  They are replaced by SENTINEL in
# the concatenated paragraph text and their subtrees are not visited.
_BREAKING_TAGS = frozenset(
    [_w(t) for t in (
        "tab", "ptab", "br", "cr", "sym", "noBreakHyphen", "softHyphen",
        "delText", "instrText", "delInstrText", "drawing", "pict", "object",
        "footnoteReference", "endnoteReference", "commentReference",
        "footnoteRef", "endnoteRef", "annotationRef", "separator", "continuationSeparator",
        "pgNum", "fldChar", "contentPart",
        "del", "moveFrom",
    )]
    + [f"{{{MC_NS}}}AlternateContent"]
)
SENTINEL = "\x00"

# CONTRACT.md §4 regex is ⟦([A-Z0-9_]+)(?::[^⟧]*)?⟧.  The description here also
# excludes "⟦", so that a malformed/unclosed token can never swallow a following
# well-formed one ("⟦A: see ⟦B⟧" -> ⟦B⟧ is filled and "⟦A: see " is an error);
# for every well-formed token both regexes match identically.
TOKEN_RE = re.compile(r"⟦([A-Z0-9_]+)(?::[^⟦⟧]*)?⟧")
# A bracket that looks like part of a token (key-like character next to it).
# Other stray brackets, e.g. the notation "⟦...⟧" in an explanatory comment,
# are only warned about.
TOKEN_LIKE_FRAGMENT_RE = re.compile(r"⟦\s*[A-Za-z0-9_]|[A-Za-z0-9_]\s*⟧")
KEY_RE = re.compile(r"[A-Z0-9_]+")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_XML_DECL = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
ASPECT_TOLERANCE = 0.01


@dataclass(frozen=True)
class FigureSpec:
    """A figure placeholder and what replaces it."""

    placeholder: str      # <wp:docPr name=...> in the manuscript
    filename: str         # PNG in the results directory
    final_name: str       # docPr name after replacement
    media_stem: str       # stem for a new media part, if one has to be created
    expected_px: tuple[int, int]


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("PLACEHOLDER_FIG5_RELIABILITY", "fig_reliability.png", "Figure 5",
               "fig5_reliability", (2100, 780)),
    FigureSpec("PLACEHOLDER_FIG6_SELECTIVE", "fig_selective.png", "Figure 6",
               "fig6_selective", (2100, 840)),
)


def contract_keys() -> list[str]:
    """Every placeholder key listed in CONTRACT.md §4, in contract order."""
    keys: list[str] = []
    for i in range(1, 6):
        keys += [f"T5_R{i}_{m}" for m in ("ACC", "CONF", "ECE", "BRIER", "NLL")]
    keys.append("T5_TEMP")
    for i in range(1, 6):
        keys += [f"T6_R{i}_{m}" for m in ("DACC", "DF1", "BC", "P", "PHOLM")]
    for i in range(1, 7):
        keys += [f"T7_R{i}_{m}" for m in ("AUROC", "AURC", "ACC80", "COV90")]
    keys += ["T1_ECE", "T1_ENC", "T1_AUROC"]
    keys += ["TXT_REPRO", "TXT_CALIBRATION", "TXT_PAIRED", "TXT_SELECTIVE",
             "TXT_IMPLICATION", "TXT_CONCLUSION"]
    return keys


class FillError(Exception):
    """Unrecoverable usage/input error (exit code 2)."""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class FillReport:
    """Everything the fill run did or could not do."""

    docx_in: str = ""
    out: str = ""
    replaced: Counter = field(default_factory=Counter)        # key -> tokens replaced
    missing: Counter = field(default_factory=Counter)         # key -> tokens left (no value)
    unknown_keys: list[str] = field(default_factory=list)     # in fill_values, not in document
    contract_keys_absent: list[str] = field(default_factory=list)  # §4 keys not in fill_values
    split_tokens: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)
    modified_parts: list[str] = field(default_factory=list)
    added_parts: list[str] = field(default_factory=list)
    removed_parts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    allow_missing: bool = False
    written: bool = False

    @property
    def ok(self) -> bool:
        """True when nothing blocks a clean exit (respecting --allow-missing)."""
        if self.errors:
            return False
        if self.allow_missing:
            return True
        return not self.missing and not self.missing_figures

    def to_dict(self) -> dict[str, Any]:
        return {
            "docx_in": self.docx_in, "out": self.out, "ok": self.ok,
            "written": self.written, "allow_missing": self.allow_missing,
            "replaced": dict(sorted(self.replaced.items())),
            "n_tokens_replaced": sum(self.replaced.values()),
            "missing": dict(sorted(self.missing.items())),
            "unknown_keys": self.unknown_keys,
            "contract_keys_absent": self.contract_keys_absent,
            "split_tokens": self.split_tokens, "figures": self.figures,
            "missing_figures": self.missing_figures,
            "modified_parts": self.modified_parts, "added_parts": self.added_parts,
            "removed_parts": self.removed_parts,
            "warnings": self.warnings, "errors": self.errors,
        }

    def format(self) -> str:
        """Human-readable multi-line summary."""
        lines = ["== fill_manuscript report ==", f"input : {self.docx_in}",
                 f"output: {self.out}" + ("" if self.written else " (NOT written)")]
        n_rep = sum(self.replaced.values())
        lines.append(f"tokens replaced: {n_rep} ({len(self.replaced)} keys)")
        for key, n in sorted(self.replaced.items()):
            lines.append(f"  {key:<18} {n}")
        if self.split_tokens:
            lines.append(f"tokens split across runs: {len(self.split_tokens)}")
            for s in self.split_tokens:
                lines.append(f"  {s['key']} in {s['part']} ({s['n_runs']} runs, "
                             f"{'replaced' if s['replaced'] else 'NOT replaced'})")
        if self.missing:
            lines.append("MISSING keys (tokens left in document): "
                         + ", ".join(f"{k} x{n}" for k, n in sorted(self.missing.items())))
        if self.unknown_keys:
            lines.append("unknown keys in fill_values (no token in document): "
                         + ", ".join(self.unknown_keys))
        if self.contract_keys_absent:
            lines.append("contract §4 keys absent from fill_values: "
                         + ", ".join(self.contract_keys_absent))
        lines.append(f"figures replaced: {len(self.figures)}")
        for f in self.figures:
            adj = (f"extent cy {f['cy_old']} -> {f['cy_new']} (aspect adjusted)"
                   if f["aspect_adjusted"] else "extent kept")
            lines.append(f"  {f['placeholder']} -> {f['final_name']}: {f['media_part']} "
                         f"[{f['action']}], {f['png_px'][0]}x{f['png_px'][1]} px, {adj}")
        if self.missing_figures:
            lines.append("MISSING figures: " + "; ".join(self.missing_figures))
        for kind, items in (("parts modified", self.modified_parts),
                            ("parts added", self.added_parts),
                            ("parts removed", self.removed_parts)):
            if items:
                lines.append(f"{kind}: " + ", ".join(items))
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        for e in self.errors:
            lines.append(f"ERROR: {e}")
        if self.ok:
            lines.append("status: OK" + (" (missing items allowed)" if self.allow_missing and
                                          (self.missing or self.missing_figures) else ""))
        else:
            lines.append("status: INCOMPLETE -> exit code 1")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Package access
# ---------------------------------------------------------------------------

_PARSER = etree.XMLParser(remove_blank_text=False, resolve_entities=False,
                          huge_tree=True, strip_cdata=False)
_DECL_RE = re.compile(rb"^(\xef\xbb\xbf)?\s*<\?xml[^>]*\?>\s*")


def rels_part_name(part: str) -> str:
    """Name of the relationships part belonging to ``part`` ('' = package)."""
    directory, base = posixpath.split(part)
    return posixpath.join(directory, "_rels", f"{base}.rels")


def resolve_target(source_part: str, target: str) -> str:
    """Resolve a relationship Target relative to its source part."""
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


class DocxPackage:
    """An OPC package held in memory; unchanged entries are written back verbatim."""

    def __init__(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as zf:
                self.infos: list[zipfile.ZipInfo] = zf.infolist()
                self.blobs: dict[str, bytes] = {i.filename: zf.read(i.filename)
                                                for i in self.infos}
        except (zipfile.BadZipFile, OSError) as exc:
            raise FillError(f"cannot read DOCX {path}: {exc}") from exc
        self._trees: dict[str, etree._ElementTree] = {}
        self._decls: dict[str, bytes] = {}
        self.dirty: set[str] = set()
        self.new_blobs: dict[str, bytes] = {}      # replaced or added binary parts
        self.added: list[str] = []
        self.removed: set[str] = set()

    # -- queries ----------------------------------------------------------
    def has(self, name: str) -> bool:
        return (name in self.blobs or name in self.added) and name not in self.removed

    def names(self) -> list[str]:
        return [i.filename for i in self.infos if i.filename not in self.removed] + [
            n for n in self.added if n not in self.removed]

    def xml(self, name: str) -> etree._Element:
        """Parsed root element of an XML part (cached; edits persist)."""
        if name not in self._trees:
            data = self.new_blobs.get(name, self.blobs.get(name))
            if data is None:
                raise KeyError(name)
            m = _DECL_RE.match(data)
            self._decls[name] = m.group(0) if m else DEFAULT_XML_DECL
            try:
                root = etree.fromstring(data, _PARSER)
            except etree.XMLSyntaxError as exc:
                raise FillError(f"part {name} is not well-formed XML: {exc}") from exc
            self._trees[name] = root.getroottree()
        return self._trees[name].getroot()

    # -- edits ------------------------------------------------------------
    def mark_dirty(self, name: str) -> None:
        self.dirty.add(name)

    def set_blob(self, name: str, data: bytes) -> None:
        self.new_blobs[name] = data
        self._trees.pop(name, None)
        self.dirty.discard(name)

    def add_blob(self, name: str, data: bytes) -> None:
        if name in self.blobs or name in self.added:
            raise ValueError(f"part {name} already exists")
        self.added.append(name)
        self.new_blobs[name] = data

    def add_xml(self, name: str, root: etree._Element) -> None:
        """Register a newly created XML part."""
        self.add_blob(name, b"")
        self._trees[name] = root.getroottree()
        self._decls[name] = DEFAULT_XML_DECL
        self.dirty.add(name)

    def remove(self, name: str) -> None:
        self.removed.add(name)

    def unique_name(self, stem_path: str, ext: str) -> str:
        """First free part name ``stem_path{ext}``, ``stem_path_2{ext}``, ..."""
        existing = {n.lower() for n in self.blobs} | {n.lower() for n in self.added}
        candidate, i = f"{stem_path}{ext}", 1
        while candidate.lower() in existing:
            i += 1
            candidate = f"{stem_path}_{i}{ext}"
        return candidate

    # -- output -----------------------------------------------------------
    def current_bytes(self, name: str) -> bytes:
        if name in self.dirty:
            body = etree.tostring(self._trees[name], encoding="UTF-8", xml_declaration=False)
            return self._decls[name] + body
        if name in self.new_blobs:
            return self.new_blobs[name]
        return self.blobs[name]

    def changed_parts(self) -> list[str]:
        """Pre-existing parts whose bytes differ from the input."""
        return [i.filename for i in self.infos
                if i.filename not in self.removed and i.filename not in self.added
                and (i.filename in self.dirty or i.filename in self.new_blobs)
                and self.current_bytes(i.filename) != self.blobs[i.filename]]

    def save(self, out: Path) -> None:
        """Write the package: original entry order and compression, new parts last."""
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp, "w") as zout:
                for info in self.infos:
                    if info.filename in self.removed:
                        continue
                    zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                    zi.compress_type = info.compress_type
                    zi.external_attr = info.external_attr
                    zi.create_system = info.create_system
                    zi.comment = info.comment
                    zout.writestr(zi, self.current_bytes(info.filename))
                for name in self.added:
                    if name in self.removed:
                        continue
                    zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    zi.compress_type = (zipfile.ZIP_STORED if name.lower().endswith(".png")
                                        else zipfile.ZIP_DEFLATED)
                    zout.writestr(zi, self.current_bytes(name))
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- relationships ----------------------------------------------------
    def relationships(self, source_part: str) -> etree._Element | None:
        name = rels_part_name(source_part)
        return self.xml(name) if self.has(name) else None

    def main_document_part(self) -> str:
        rels = self.relationships("")
        if rels is not None:
            for rel in rels.iter(f"{{{REL_NS}}}Relationship"):
                if rel.get("Type") == REL_TYPE_OFFICE_DOC:
                    return resolve_target("", rel.get("Target", ""))
        if self.has("word/document.xml"):
            return "word/document.xml"
        raise FillError("no main document part found (is this a DOCX?)")

    def story_parts(self) -> list[str]:
        """Main document plus headers, footers, footnotes, endnotes, comments."""
        main = self.main_document_part()
        parts = [main]
        rels = self.relationships(main)
        if rels is not None:
            for rel in rels.iter(f"{{{REL_NS}}}Relationship"):
                if rel.get("TargetMode") == "External":
                    continue
                if (rel.get("Type") or "").endswith(STORY_REL_SUFFIXES):
                    target = resolve_target(main, rel.get("Target", ""))
                    if self.has(target) and target not in parts:
                        parts.append(target)
        return parts

    def all_rels_parts(self) -> list[str]:
        return [n for n in self.names() if n.endswith(".rels")]

    def count_rels_targeting(self, target: str) -> int:
        """Internal relationships (in any .rels part) whose target is ``target``."""
        n = 0
        for rels_name in self.all_rels_parts():
            directory = posixpath.dirname(posixpath.dirname(rels_name))
            base = posixpath.basename(rels_name)[: -len(".rels")]
            source = posixpath.join(directory, base) if base else ""
            for rel in self.xml(rels_name).iter(f"{{{REL_NS}}}Relationship"):
                if rel.get("TargetMode") == "External":
                    continue
                if resolve_target(source, rel.get("Target", "")) == target:
                    n += 1
        return n


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

# Characters that cannot appear in XML 1.0 (lxml refuses them; lone surrogates,
# which json.loads accepts from "\ud800", cannot even be encoded as UTF-8).
# TAB/LF/CR are legal XML but are normalised separately below.
_XML_INVALID_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")
# Line/paragraph breaks and tabs have no meaning inside <w:t>: collapse to one space.
_BREAK_WS_RE = re.compile(r"\s*[\t\r\n\x85  ]+\s*")
_ASCII_MINUS_RE = re.compile(r"(?<![\w.])-(?=\d)")


def _json_object_pairs(report: FillReport, path: Path):
    """``object_pairs_hook`` that warns about duplicate keys (json keeps the last)."""
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                report.warnings.append(f"{path.name}: duplicate key {key!r}; "
                                       "the last occurrence is used")
            seen[key] = value
        return seen
    return hook


def load_fill_values(path: Path, report: FillReport) -> dict[str, str]:
    """Load and validate ``fill_values.json`` (flat ``{KEY: "string"}``).

    Values are normalised to what can go into a single ``<w:t>``: tabs and line
    breaks collapse to one space and surrounding whitespace is stripped (both
    with a warning).  Values that are not strings, are empty, contain characters
    that are illegal in XML or contain token brackets are errors: the token is
    then left unfilled.
    """
    try:
        # utf-8-sig: tolerate a BOM written by Windows editors / PowerShell.
        raw = json.loads(path.read_text(encoding="utf-8-sig"),
                         object_pairs_hook=_json_object_pairs(report, path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FillError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FillError(f"{path} must contain a JSON object {{KEY: value}}")
    values: dict[str, str] = {}
    for key, value in raw.items():
        if not KEY_RE.fullmatch(key):
            report.warnings.append(f"fill_values key {key!r} does not match [A-Z0-9_]+ "
                                   "and can never match a token")
            continue
        if not isinstance(value, str):
            report.errors.append(f"fill_values[{key}] is {type(value).__name__}, "
                                 "expected a formatted string; token left unfilled")
            continue
        if _XML_INVALID_RE.search(value):
            report.errors.append(f"fill_values[{key}] contains characters that are not "
                                 "allowed in XML (control characters, U+FFFE/U+FFFF or "
                                 "unpaired surrogates); token left unfilled")
            continue
        if _BREAK_WS_RE.search(value):
            value = _BREAK_WS_RE.sub(" ", value)
            report.warnings.append(f"fill_values[{key}] contained tabs or line breaks; "
                                   "collapsed to single spaces")
        if value != value.strip():
            value = value.strip()
            report.warnings.append(f"fill_values[{key}] had leading/trailing whitespace; "
                                   "stripped")
        if not value:
            report.errors.append(f"fill_values[{key}] is empty; token left unfilled")
            continue
        if "⟦" in value or "⟧" in value:
            report.errors.append(f"fill_values[{key}] contains token brackets ⟦⟧; "
                                 "token left unfilled")
            continue
        if _ASCII_MINUS_RE.search(value):
            report.warnings.append(f"fill_values[{key}] = {value[:60]!r} uses ASCII '-' as a "
                                   "minus sign (contract requires U+2212)")
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# Token replacement
# ---------------------------------------------------------------------------


def _set_text(t: etree._Element, text: str) -> None:
    """Set ``<w:t>`` text, keeping/adding xml:space="preserve" where needed."""
    t.text = text
    if text != text.strip() or "  " in text:
        t.set(XML_SPACE, "preserve")


def _enclosing_run(el: etree._Element) -> etree._Element | None:
    parent = el.getparent()
    while parent is not None and parent.tag != W_R:
        if parent.tag == W_P:
            return None
        parent = parent.getparent()
    return parent


def _is_deleted_text(t: etree._Element) -> bool:
    """True for a <w:t> inside a tracked deletion or the source of a tracked move.

    Word stores deleted text in <w:delText>, but moved-from text (<w:moveFrom>)
    may use plain <w:t>; neither is part of the final text and both are left
    untouched, exactly like <w:delText>.
    """
    parent = t.getparent()
    while parent is not None and parent.tag != W_P:
        if parent.tag in (W_DEL, W_MOVEFROM):
            return True
        parent = parent.getparent()
    return False


def _in_fallback(el: etree._Element) -> bool:
    """True inside <mc:Fallback>: a duplicate of the <mc:Choice> content.

    Tokens there are replaced too (so both renderings agree) but not counted,
    so that the report counts every visible token once.
    """
    return any(True for _ in el.iterancestors(MC_FALLBACK))


def _paragraph_segments(p: etree._Element) -> list[tuple[etree._Element | None, str]]:
    """Visible text pieces of paragraph ``p`` in order: (w:t, text) or (None, SENTINEL).

    Nested paragraphs (text boxes) are skipped; they are visited on their own.
    """
    segments: list[tuple[etree._Element | None, str]] = []
    stack = list(reversed(p))
    while stack:
        el = stack.pop()
        tag = el.tag
        if not isinstance(tag, str):          # comments, processing instructions
            continue
        if tag == W_T:
            segments.append((el, el.text or ""))
        elif tag in _BREAKING_TAGS:
            segments.append((None, SENTINEL))
        elif tag in (W_P, W_PPR, W_RPR):
            continue
        else:
            stack.extend(reversed(el))
    return segments


def _context(text: str, start: int, end: int, width: int = 30) -> str:
    snippet = text[max(0, start - width): end + width].replace(SENTINEL, "¦")
    return snippet if len(snippet) < 140 else snippet[:137] + "..."


def _report_unsupported_tokens(root: etree._Element, part: str, report: FillReport) -> None:
    """Report tokens in text elements this script does not fill.

    Tokens belong in <w:t> (CONTRACT.md §4).  One in an equation (m:t) or in
    DrawingML text (a:t) would stay visible in the manuscript, so it is an error
    rather than being skipped silently; one in a field code (w:instrText) or any
    other non-rendered text is a warning.  Deleted text is ignored.
    """
    for el in root.iter():
        tag = el.tag
        if not isinstance(tag, str) or tag in (W_T, W_DELTEXT, W_DELINSTRTEXT):
            continue
        text = el.text
        if not text or ("⟦" not in text and "⟧" not in text):
            continue
        if not (TOKEN_RE.search(text) or TOKEN_LIKE_FRAGMENT_RE.search(text)):
            continue
        qname = etree.QName(el)
        where = f"{part}: <{el.prefix + ':' if el.prefix else ''}{qname.localname}>"
        snippet = text if len(text) <= 80 else text[:77] + "..."
        if tag in _VISIBLE_UNSUPPORTED_TEXT:
            report.errors.append(f"{where} contains a token that this script cannot fill "
                                 f"(tokens must be in normal text runs): {snippet!r}")
        else:
            report.warnings.append(f"{where} contains token-like text that is not visible "
                                   f"document text and was left as is: {snippet!r}")


def fill_tokens_in_part(root: etree._Element, part: str, values: dict[str, str],
                        report: FillReport, keep_highlight: bool = False) -> bool:
    """Replace tokens in one story part. Returns True if the XML changed."""
    changed = False
    touched_runs: dict[int, etree._Element] = {}

    # Pass 1: tokens entirely inside one <w:t> (the contract's normal case).
    for t in root.iter(W_T):
        text = t.text or ""
        if "⟦" not in text or _is_deleted_text(t):
            continue
        counted = not _in_fallback(t)

        def _sub(m: re.Match[str], counted: bool = counted) -> str:
            key = m.group(1)
            tally = report.replaced if key in values else report.missing
            if counted:
                tally[key] += 1
            return values[key] if key in values else m.group(0)

        new = TOKEN_RE.sub(_sub, text)
        if new != text:
            _set_text(t, new)
            changed = True
            run = _enclosing_run(t)
            if run is not None:
                touched_runs[id(run)] = run

    # Pass 2: tokens split across runs of one paragraph; leftover fragments.
    for p in root.iter(W_P):
        segments = _paragraph_segments(p)
        full = "".join(s for _, s in segments)
        if "⟦" not in full and "⟧" not in full:
            continue
        counted = not _in_fallback(p)
        starts: list[int] = []
        pos = 0
        for _, s in segments:
            starts.append(pos)
            pos += len(s)
        for m in reversed(list(TOKEN_RE.finditer(full))):
            ms, me = m.span()
            idx = [i for i, (_, s) in enumerate(segments)
                   if (starts[i] < me and starts[i] + len(s) > ms)
                   or (len(s) == 0 and ms < starts[i] < me)]
            t_idx = [i for i in idx if segments[i][0] is not None]
            if len(idx) == 1:
                continue                     # single-<w:t> token: handled in pass 1
            key = m.group(1)
            if SENTINEL in m.group(0):
                report.errors.append(f"{part}: token {key} is interrupted by a tab/break/"
                                     f"field/drawing/tracked deletion (shown as ¦) and was "
                                     f"not replaced: {_context(full, ms, me)!r}")
                continue
            if counted:
                report.split_tokens.append({"key": key, "part": part, "n_runs": len(t_idx),
                                            "text": m.group(0)[:80],
                                            "replaced": key in values})
            if key not in values:
                if counted:
                    report.missing[key] += 1
                continue
            first, last = t_idx[0], t_idx[-1]
            t_first = segments[first][0]
            t_last = segments[last][0]
            assert t_first is not None and t_last is not None
            # Last segment first: it may already hold the value of a later token.
            _set_text(t_last, (t_last.text or "")[me - starts[last]:])
            for i in t_idx[1:-1]:
                t_mid = segments[i][0]
                assert t_mid is not None
                _set_text(t_mid, "")
            _set_text(t_first, (t_first.text or "")[: ms - starts[first]] + values[key])
            if counted:
                report.replaced[key] += 1
            changed = True
            for i in t_idx:
                seg_t = segments[i][0]
                assert seg_t is not None
                run = _enclosing_run(seg_t)
                if run is not None:
                    touched_runs[id(run)] = run

    # Leftover bracket fragments that do not form a token.
    for p in root.iter(W_P):
        full = "".join(s for _, s in _paragraph_segments(p))
        if "⟦" not in full and "⟧" not in full:
            continue
        # Blank out well-formed tokens (missing keys are reported already).
        rest = TOKEN_RE.sub(lambda m: SENTINEL * len(m.group(0)), full)
        frag = TOKEN_LIKE_FRAGMENT_RE.search(rest)
        if frag is not None:
            report.errors.append(f"{part}: malformed/unresolvable token fragment near "
                                 f"{_context(full, frag.start(), frag.end())!r} "
                                 "(¦ = tab/break/field/drawing/tracked deletion)")
            continue
        frag = re.search(r"[⟦⟧]", rest)
        if frag is not None:            # e.g. the notation "⟦...⟧" in an author comment
            report.warnings.append(f"{part}: bracket that is not part of a token left as is "
                                   f"near {_context(full, frag.start(), frag.end())!r}")

    _report_unsupported_tokens(root, part, report)

    if not keep_highlight:
        for run in touched_runs.values():
            run_text = "".join(t.text or "" for t in run.iter(W_T))
            if "⟦" in run_text or "⟧" in run_text:
                continue
            rpr = run.find(W_RPR)
            if rpr is not None:
                for hl in rpr.findall(W_HIGHLIGHT):
                    rpr.remove(hl)
                    changed = True
    return changed


# ---------------------------------------------------------------------------
# Figure replacement
# ---------------------------------------------------------------------------


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) in pixels from a PNG header; ValueError if not a PNG."""
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    if width == 0 or height == 0:
        raise ValueError("PNG has zero width/height")
    return width, height


def ensure_png_content_type(pkg: DocxPackage, report: FillReport) -> None:
    """Make sure [Content_Types].xml has ``<Default Extension="png">``."""
    name = "[Content_Types].xml"
    root = pkg.xml(name)
    defaults = root.findall(f"{{{CT_NS}}}Default")
    for d in defaults:
        if (d.get("Extension") or "").lower() == "png":
            if d.get("ContentType") != "image/png":
                d.set("ContentType", "image/png")
                pkg.mark_dirty(name)
                report.warnings.append("[Content_Types].xml png Default had a non-standard "
                                       "ContentType; set to image/png")
            return
    new = etree.Element(f"{{{CT_NS}}}Default", Extension="png", ContentType="image/png")
    if defaults:
        defaults[-1].addnext(new)
    else:
        root.insert(0, new)
    pkg.mark_dirty(name)


def _set_override(pkg: DocxPackage, part: str, content_type: str | None) -> None:
    """Set (or with None: remove) the content-type Override of ``part``, if any."""
    name = "[Content_Types].xml"
    root = pkg.xml(name)
    for ov in root.findall(f"{{{CT_NS}}}Override"):
        if (ov.get("PartName") or "").lstrip("/").lower() == part.lower():
            if content_type is None:
                root.remove(ov)
                pkg.mark_dirty(name)
            elif ov.get("ContentType") != content_type:
                ov.set("ContentType", content_type)
                pkg.mark_dirty(name)


def _count_rid_refs(root: etree._Element, rid: str) -> int:
    """References to relationship ``rid`` in a part: r:embed/r:link/r:id/... and o:relid."""
    prefix = f"{{{R_NS}}}"
    return sum(1 for el in root.iter() if isinstance(el.tag, str)
               for k, v in el.attrib.items()
               if v == rid and (k.startswith(prefix) or k.endswith("}relid")))


def _find_rel(rels: etree._Element | None, rid: str) -> etree._Element | None:
    if rels is None:
        return None
    for rel in rels.iter(f"{{{REL_NS}}}Relationship"):
        if rel.get("Id") == rid:
            return rel
    return None


def _add_image_rel(pkg: DocxPackage, source_part: str, target_part: str) -> str:
    rels_name = rels_part_name(source_part)
    if pkg.has(rels_name):
        rels = pkg.xml(rels_name)
    else:
        rels = etree.Element(f"{{{REL_NS}}}Relationships", nsmap={None: REL_NS})
        pkg.add_xml(rels_name, rels)
    ids = {rel.get("Id") for rel in rels.iter(f"{{{REL_NS}}}Relationship")}
    n = 1
    while f"rId{n}" in ids:
        n += 1
    rid = f"rId{n}"
    target = posixpath.relpath(target_part, posixpath.dirname(source_part) or ".")
    etree.SubElement(rels, f"{{{REL_NS}}}Relationship", Id=rid, Type=REL_TYPE_IMAGE,
                     Target=target)
    pkg.mark_dirty(rels_name)
    return rid


class _ImageInstaller:
    """Writes figure PNGs into the package, never clobbering shared media parts."""

    def __init__(self, pkg: DocxPackage, report: FillReport) -> None:
        self.pkg = pkg
        self.report = report
        self.written_targets: set[str] = set()
        self.cache: dict[tuple[str, str], tuple[str, str]] = {}   # (source, figure) -> (rid, part)
        self.detached: list[tuple[str, str, str]] = []            # (source, old rid, old part)

    def install(self, source_part: str, root: etree._Element, rid: str, png: bytes,
                spec: FigureSpec) -> tuple[str, str, str]:
        """Return (rid to use, media part, action)."""
        rel = _find_rel(self.pkg.relationships(source_part), rid)
        if rel is None or rel.get("TargetMode") == "External":
            raise ValueError(f"image relationship {rid} of {source_part} is missing or external")
        target = resolve_target(source_part, rel.get("Target", ""))
        cached = self.cache.get((source_part, spec.placeholder))
        if cached is not None:                 # same placeholder twice in one part
            if cached[0] != rid:
                self.detached.append((source_part, rid, target))
            return cached[0], cached[1], "shared-with-previous"
        exclusive = (target.lower().endswith(".png")
                     and self.pkg.has(target)
                     and target not in self.written_targets
                     and _count_rid_refs(root, rid) == 1
                     and self.pkg.count_rels_targeting(target) == 1)
        if exclusive:
            self.pkg.set_blob(target, png)
            _set_override(self.pkg, target, "image/png")
            self.written_targets.add(target)
            self.cache[(source_part, spec.placeholder)] = (rid, target)
            return rid, target, "replaced-in-place"
        media_dir = posixpath.join(posixpath.dirname(source_part), "media")
        new_part = self.pkg.unique_name(posixpath.join(media_dir, spec.media_stem), ".png")
        self.pkg.add_blob(new_part, png)
        new_rid = _add_image_rel(self.pkg, source_part, new_part)
        self.written_targets.add(new_part)
        self.detached.append((source_part, rid, target))
        self.cache[(source_part, spec.placeholder)] = (new_rid, new_part)
        return new_rid, new_part, "new-part"

    def cleanup(self, roots: dict[str, etree._Element]) -> None:
        """Drop relationships/parts orphaned by re-pointing blips to new parts."""
        for source, rid, target in self.detached:
            root = roots.get(source)
            if root is None or _count_rid_refs(root, rid) > 0:
                continue
            rels = self.pkg.relationships(source)
            rel = _find_rel(rels, rid)
            if rels is not None and rel is not None:
                rels.remove(rel)
                self.pkg.mark_dirty(rels_part_name(source))
            if self.pkg.has(target) and self.pkg.count_rels_targeting(target) == 0:
                self.pkg.remove(target)
                _set_override(self.pkg, target, None)
                self.report.removed_parts.append(target)


def replace_figures(pkg: DocxPackage, roots: dict[str, etree._Element],
                    pngs: dict[str, bytes], report: FillReport) -> set[str]:
    """Replace placeholder figures in all story parts. Returns the changed parts."""
    by_name = {spec.placeholder: spec for spec in FIGURES}
    found: Counter = Counter()
    changed: set[str] = set()
    installer = _ImageInstaller(pkg, report)
    for part, root in roots.items():
        for docpr in list(root.iter(WP_DOCPR)):
            spec = by_name.get(docpr.get("name") or "")
            if spec is None:
                continue
            found[spec.placeholder] += 1
            png = pngs.get(spec.placeholder)
            if png is None:
                continue                                   # reported by caller
            container = docpr.getparent()                  # wp:inline / wp:anchor
            scope = container
            anc = docpr.getparent()
            while anc is not None:
                if anc.tag == W_DRAWING:
                    scope = anc
                    break
                anc = anc.getparent()
            blips = list(scope.iter(A_BLIP))
            if len(blips) != 1 or not blips[0].get(R_EMBED):
                report.errors.append(f"{part}: {spec.placeholder} must contain exactly one "
                                     f"embedded picture (found {len(blips)} a:blip)")
                continue
            blip = blips[0]
            old_rid = blip.get(R_EMBED)
            assert old_rid is not None
            try:
                rid, media_part, action = installer.install(part, root, old_rid, png, spec)
            except ValueError as exc:
                report.errors.append(f"{part}: {spec.placeholder}: {exc}")
                continue
            if rid != old_rid:
                blip.set(R_EMBED, rid)
            # A vector (SVG) twin or a crop would hide/distort the new bitmap.
            for ext in list(blip.iter(A_EXT)):
                if ext.get("uri") == SVG_BLIP_EXT_URI:
                    ext.getparent().remove(ext)
                    report.warnings.append(f"{spec.placeholder}: removed SVG twin of the "
                                           "placeholder picture")
            for src in list(scope.iter(A_SRCRECT)):
                src.getparent().remove(src)
                if any(v not in ("0", "") for v in src.attrib.values()):
                    report.warnings.append(f"{spec.placeholder}: removed crop (a:srcRect)")

            w_px, h_px = png_size(png)
            extent = container.find(WP_EXTENT) if container is not None else None
            cx_old = int(extent.get("cx", "0")) if extent is not None else 0
            cy_old = int(extent.get("cy", "0")) if extent is not None else 0
            adjusted = False
            cy_new = cy_old
            if cx_old > 0 and cy_old > 0:
                ratio = (w_px / h_px) / (cx_old / cy_old)
                if abs(ratio - 1.0) > ASPECT_TOLERANCE:
                    adjusted = True
                    cy_new = round(cx_old * h_px / w_px)
                    assert extent is not None
                    extent.set("cy", str(cy_new))
                    for a_ext in scope.iter(A_EXT):
                        parent = a_ext.getparent()
                        if (parent is not None and parent.tag == f"{{{A_NS}}}xfrm"
                                and a_ext.get("cx") is not None):
                            a_ext.set("cy", str(round(int(a_ext.get("cx")) * h_px / w_px)))
            else:
                report.warnings.append(f"{spec.placeholder}: no usable wp:extent; size kept")

            docpr.set("name", spec.final_name)
            for cnv in scope.iter(PIC_CNVPR):
                if cnv.get("name") == spec.placeholder:
                    cnv.set("name", spec.final_name)
            changed.add(part)
            report.figures.append({
                "placeholder": spec.placeholder, "final_name": spec.final_name,
                "part": part, "media_part": media_part, "action": action,
                "png_px": [w_px, h_px], "cx": cx_old, "cy_old": cy_old, "cy_new": cy_new,
                "aspect_adjusted": adjusted,
            })
    installer.cleanup(roots)
    for spec in FIGURES:
        if found[spec.placeholder] == 0:
            report.missing_figures.append(f"{spec.placeholder}: no <wp:docPr name=...> "
                                          "placeholder in document")
        elif spec.placeholder not in pngs:
            report.missing_figures.append(f"{spec.placeholder}: {spec.filename} not "
                                          "available in results directory")
    if report.figures:
        ensure_png_content_type(pkg, report)
    return changed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def fill_manuscript(docx_in: Path, results_dir: Path, out: Path, *,
                    allow_missing: bool = False, keep_highlight: bool = False) -> FillReport:
    """Fill ``docx_in`` from ``results_dir`` and write ``out``.

    Raises FillError for usage errors.  The output is written even when tokens
    or figures are missing; ``report.ok`` tells whether the run is complete.
    """
    docx_in, results_dir, out = Path(docx_in), Path(results_dir), Path(out)
    report = FillReport(docx_in=str(docx_in), out=str(out), allow_missing=allow_missing)
    if not docx_in.is_file():
        raise FillError(f"input DOCX not found: {docx_in}")
    if _same_file(docx_in, out):
        raise FillError(f"refusing to overwrite the input document: {out}")
    if out.is_dir():
        raise FillError(f"--out must be a file path, not a directory: {out}")
    if not results_dir.is_dir():
        raise FillError(f"results directory not found: {results_dir}")
    if out.suffix.lower() != ".docx":
        report.warnings.append(f"output file {out.name} does not end in .docx")

    values_path = results_dir / "fill_values.json"
    if values_path.is_file():
        values = load_fill_values(values_path, report)
    elif allow_missing:
        values = {}
        report.warnings.append(f"{values_path} not found; no tokens filled")
    else:
        raise FillError(f"{values_path} not found (use --allow-missing to fill figures only)")
    absent = [k for k in contract_keys() if k not in values]
    if values and absent:
        report.contract_keys_absent = absent

    pngs: dict[str, bytes] = {}
    for spec in FIGURES:
        path = results_dir / spec.filename
        if not path.is_file():
            continue
        data = path.read_bytes()
        try:
            size = png_size(data)
        except ValueError as exc:
            report.errors.append(f"{path}: {exc}")
            continue
        if size != spec.expected_px:
            report.warnings.append(f"{spec.filename} is {size[0]}x{size[1]} px; contract "
                                   f"expects {spec.expected_px[0]}x{spec.expected_px[1]}")
        pngs[spec.placeholder] = data

    pkg = DocxPackage(docx_in)
    if not pkg.has("[Content_Types].xml"):
        raise FillError(f"{docx_in} has no [Content_Types].xml (not an OPC package)")
    parts = pkg.story_parts()
    roots = {part: pkg.xml(part) for part in parts}
    main_root = roots[parts[0]]
    if main_root.tag != _w("document"):
        raise FillError(f"unsupported main document root {main_root.tag} "
                        "(Strict OOXML is not supported; save as a standard .docx)")

    for part, root in roots.items():
        if fill_tokens_in_part(root, part, values, report, keep_highlight):
            pkg.mark_dirty(part)
    for part in replace_figures(pkg, roots, pngs, report):
        pkg.mark_dirty(part)

    seen = set(report.replaced) | set(report.missing)
    report.unknown_keys = sorted(k for k in values if k not in seen)
    # Choice/Fallback copies of a text box yield identical messages: keep one.
    report.errors = list(dict.fromkeys(report.errors))
    report.warnings = list(dict.fromkeys(report.warnings))

    try:
        pkg.save(out)
    except OSError as exc:
        hint = (" - is the file open in Word? Close it and run again"
                if isinstance(exc, PermissionError) else "")
        raise FillError(f"cannot write {out}: {exc}{hint}") from exc
    report.written = True
    report.modified_parts = pkg.changed_parts()
    report.added_parts = [n for n in pkg.added if n not in pkg.removed]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill the placeholder tokens (U+27E6 KEY U+27E7) and placeholder figures "
                    "of the manuscript DOCX from the stage-2 results (CONTRACT.md section 4).")
    parser.add_argument("--docx", required=True, type=Path, help="input manuscript .docx")
    parser.add_argument("--results", required=True, type=Path,
                        help="results directory with fill_values.json, fig_reliability.png, "
                             "fig_selective.png")
    parser.add_argument("--out", required=True, type=Path,
                        help="output .docx (must differ from --docx)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="exit 0 even if tokens or figures remain unfilled")
    parser.add_argument("--keep-highlight", action="store_true",
                        help="keep the yellow highlight on filled runs (for proofreading)")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="optionally write the fill report as JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):        # legacy Windows code pages
        try:
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        report = fill_manuscript(args.docx, args.results, args.out,
                                 allow_missing=args.allow_missing,
                                 keep_highlight=args.keep_highlight)
    except FillError as exc:
        print(f"fill_manuscript: error: {exc}", file=sys.stderr)
        return 2
    print(report.format())
    if args.report_json is not None:
        try:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report.to_dict(), indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"fill_manuscript: error: cannot write report {args.report_json}: {exc}",
                  file=sys.stderr)
            return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
