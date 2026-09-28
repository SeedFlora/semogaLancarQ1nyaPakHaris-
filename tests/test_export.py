"""Tests for stage 1 (export_per_sample_predictions.py) on a tiny synthetic fixture.

Run from the revisi_dino_eva2 folder::

    python -m pytest tests/test_export.py -q
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import export_per_sample_predictions as exp  # noqa: E402
import make_synthetic_fixture as fx  # noqa: E402

PAIRS = ["dinov3_large__mE5_large", "eva02_large__mE5_large"]
M = 30
# Literal CONTRACT section 2 column lists (deliberately not taken from the module).
BASE_COLS = ["split", "row_id", "label", "pred"] + [f"p{k}" for k in range(9)]
PGS_COLS = (["mi", "pred_entropy", "exp_entropy", "prob_std"]
            + [f"q{k}" for k in range(9)] + ["pred_loglin"])
TOP_LEVEL_KEYS = {"created_utc", "catboost_version", "numpy_version", "virtual_ensembles",
                  "pooling", "n_val", "n_test", "files", "top1_disagreements_pgs_vs_argmax"}
FILE_KEYS = {"file", "image", "text", "checkpoint", "mode", "cbm_path", "cbm_sha256",
             "tree_count", "metrics", "expected_test", "reproduces_manuscript"}
# CSV values are written with %.10g: each value carries <= 5e-10 absolute rounding error
# for magnitudes in [1, 10) (entropies), <= 5e-11 for probabilities.
CSV_ATOL = 1e-9


# ----------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------


def _run_export(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = exp.main(argv)
    return rc, buf.getvalue()


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("fixture") / "artifacts"
    fx.build_fixture(out, n_samples=600, dim=16, pairs=PAIRS, verbose=False)
    return out


@pytest.fixture(scope="session")
def preds_dir(artifacts: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("preds_noexpect")
    # small batch size so that batching is exercised (recomputations below are unbatched)
    rc, _ = _run_export(["--artifacts", str(artifacts), "--out", str(out), "--pairs", *PAIRS,
                         "--no-expect", "--batch-size", "37"])
    assert rc == 0
    return out


@pytest.fixture(scope="session")
def expect_run(artifacts: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("preds_expect")
    rc, stdout = _run_export(["--artifacts", str(artifacts), "--out", str(out),
                              "--pairs", ",".join(PAIRS)])
    assert rc == 0
    return out, stdout


@pytest.fixture(scope="session")
def manifest(preds_dir: Path) -> dict:
    return _load_manifest(preds_dir)


def _load_manifest(d: Path) -> dict:
    def _no_nan(token: str) -> float:
        raise ValueError(f"non-JSON constant {token} in manifest")

    return json.loads((d / "manifest.json").read_text(encoding="utf-8"), parse_constant=_no_nan)


def _read(d: Path, pair: str, ckpt: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(d / f"preds__{pair}__{ckpt}__{mode}.csv.gz")


def _splits(artifacts: Path) -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(artifacts / "splits" / f"{s}.csv") for s in ("val", "test")}


def _notebook_features(artifacts: Path, pair: str) -> np.ndarray:
    """Independent re-implementation of the notebook feature recipe (val then test)."""
    img, txt = pair.split("__")
    image = np.load(artifacts / "embeddings" / "image" / f"{img}.npy", mmap_mode="r")
    text = np.load(artifacts / "embeddings" / "text" / f"{txt}.npy", mmap_mode="r")
    blocks = []
    for split in _splits(artifacts).values():
        idx = split["row_id"].values
        a, b = image[idx], text[idx]
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
        b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
        blocks.append(np.concatenate([a, b], axis=1).astype(np.float32))
    return np.concatenate(blocks)


def _model(artifacts: Path, pair: str, ckpt: str):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier()
    m.load_model(str(artifacts / "models" / "checkpoints" / f"{pair}__{ckpt}.cbm"))
    return m


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=-1)


# ----------------------------------------------------------------------------------------
# pure functions
# ----------------------------------------------------------------------------------------


def test_pool_virtual_ensembles_properties() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, M, 9)) * 3
    out = exp.pool_virtual_ensembles(raw)
    per_member = scipy_softmax(raw, axis=-1)
    np.testing.assert_allclose(out.p, per_member.mean(axis=1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(out.q, scipy_softmax(raw.mean(axis=1), axis=-1), atol=1e-15)
    np.testing.assert_allclose(out.p.sum(axis=1), 1.0, atol=1e-12)
    assert (out.mi >= 0).all()
    np.testing.assert_array_equal(out.mi, np.maximum(out.pred_entropy - out.exp_entropy, 0.0))
    np.testing.assert_allclose(out.prob_std, per_member.std(axis=1).mean(axis=1), atol=1e-15)
    # identical members: no epistemic uncertainty, linear == log-linear pooling
    same = exp.pool_virtual_ensembles(np.repeat(raw[:, :1, :], M, axis=1))
    np.testing.assert_allclose(same.mi, 0.0, atol=1e-12)
    np.testing.assert_allclose(same.prob_std, 0.0, atol=1e-12)
    np.testing.assert_allclose(same.p, same.q, atol=1e-12)
    # uniform distribution has entropy log(9)
    np.testing.assert_allclose(exp.entropy(np.full((1, 9), 1 / 9)), np.log(9), atol=1e-12)


def test_l2_normalize_zero_row_and_dtype() -> None:
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    y = exp.l2_normalize(x)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, [[0.6, 0.8], [0.0, 0.0]], atol=1e-7)


def test_matches_4dp_and_parsers() -> None:
    assert exp.matches_4dp(0.80734, [0.8073])
    assert exp.matches_4dp(0.80736, [0.8073, 0.8074])
    assert not exp.matches_4dp(0.80755, [0.8074])
    assert not exp.matches_4dp(float("nan"), [0.8074])
    assert exp.parse_splits("test,val") == ["val", "test"]
    assert exp.parse_pairs(["a__b,c__d", "a__b"]) == ["a__b", "c__d"]
    with pytest.raises(exp.ExportError):
        exp.parse_pairs(["no_separator"])
    with pytest.raises(exp.ExportError):
        exp.parse_splits("train")


def test_build_features_matches_crm(artifacts: Path) -> None:
    crm_fusion = exp.load_crm(None, artifacts)
    if crm_fusion is None:
        pytest.skip("crm package not importable")
    img = np.load(artifacts / "embeddings" / "image" / "dinov3_large.npy")
    txt = np.load(artifacts / "embeddings" / "text" / "mE5_large.npy")
    idx = _splits(artifacts)["val"]["row_id"].values
    mine = exp.build_features(img, txt, idx)
    ref = crm_fusion.early_fusion(img[idx], txt[idx], l2_per_modality=True)
    assert mine.dtype == np.float32 and mine.shape == (len(idx), 32)
    assert np.array_equal(mine, ref)


def test_pairs_in_checkpoint_dir(artifacts: Path) -> None:
    assert exp.pairs_in_checkpoint_dir(artifacts / "models" / "checkpoints") == sorted(PAIRS)


# ----------------------------------------------------------------------------------------
# exported files
# ----------------------------------------------------------------------------------------


def test_file_names(preds_dir: Path) -> None:
    expected = {f"preds__{p}__{c}__{m}.csv.gz" for p in PAIRS for c in ("cb", "pgs")
                for m in ("argmax", "pgs")} | {"manifest.json"}
    assert {f.name for f in preds_dir.iterdir()} == expected


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("ckpt", ["cb", "pgs"])
@pytest.mark.parametrize("mode", ["argmax", "pgs"])
def test_columns_rows_probabilities(preds_dir: Path, artifacts: Path, pair: str, ckpt: str,
                                    mode: str) -> None:
    df = _read(preds_dir, pair, ckpt, mode)
    # exact column order; pgs-only columns only in pgs mode
    assert list(df.columns) == (BASE_COLS + PGS_COLS if mode == "pgs" else BASE_COLS)
    # rows: val block then test block, each in split-file order, labels copied
    splits = _splits(artifacts)
    assert df["split"].tolist() == ["val"] * len(splits["val"]) + ["test"] * len(splits["test"])
    ref = pd.concat([splits["val"], splits["test"]], ignore_index=True)
    assert df["row_id"].tolist() == ref["row_id"].tolist()
    assert df["label"].tolist() == ref["label_id"].tolist()
    for col in ("row_id", "label", "pred"):
        assert pd.api.types.is_integer_dtype(df[col])
    p = df[[f"p{k}" for k in range(9)]].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(df["pred"].to_numpy(), p.argmax(axis=1))
    if mode == "pgs":
        q = df[[f"q{k}" for k in range(9)]].to_numpy()
        np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(df["pred_loglin"].to_numpy(), q.argmax(axis=1))
        pe, ee, mi = (df[c].to_numpy() for c in ("pred_entropy", "exp_entropy", "mi"))
        assert (mi >= 0).all() and (df["prob_std"] >= 0).all()
        assert (pe >= 0).all() and (pe <= np.log(9) + 1e-9).all() and (ee >= 0).all()
        # mi == max(pe - ee, 0); tolerance covers the %.10g rounding of three values
        np.testing.assert_allclose(mi, np.maximum(pe - ee, 0.0), rtol=0, atol=CSV_ATOL + 1e-12)


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("ckpt", ["cb", "pgs"])
def test_pgs_mode_equals_independent_recompute(preds_dir: Path, artifacts: Path, pair: str,
                                               ckpt: str) -> None:
    """Linear pooling == mean of per-member softmax of one unbatched VirtEnsembles call."""
    X = _notebook_features(artifacts, pair)
    raw = np.asarray(_model(artifacts, pair, ckpt).virtual_ensembles_predict(
        X, prediction_type="VirtEnsembles", virtual_ensembles_count=M), dtype=np.float64)
    assert raw.shape == (X.shape[0], M, 9)
    members = scipy_softmax(raw, axis=-1)
    p_ref = members.mean(axis=1)
    df = _read(preds_dir, pair, ckpt, "pgs")
    got = lambda cols: df[cols].to_numpy()  # noqa: E731
    np.testing.assert_allclose(got([f"p{k}" for k in range(9)]), p_ref, rtol=0, atol=CSV_ATOL)
    np.testing.assert_allclose(got([f"q{k}" for k in range(9)]),
                               scipy_softmax(raw.mean(axis=1), axis=-1), rtol=0, atol=CSV_ATOL)
    pe = _entropy(p_ref)
    ee = _entropy(members).mean(axis=1)
    np.testing.assert_allclose(df["pred_entropy"], pe, rtol=0, atol=CSV_ATOL)
    np.testing.assert_allclose(df["exp_entropy"], ee, rtol=0, atol=CSV_ATOL)
    np.testing.assert_allclose(df["mi"], np.maximum(pe - ee, 0), rtol=0, atol=CSV_ATOL)
    np.testing.assert_allclose(df["prob_std"], members.std(axis=1).mean(axis=1), rtol=0,
                               atol=CSV_ATOL)
    np.testing.assert_array_equal(df["pred"], p_ref.argmax(axis=1))


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("ckpt", ["cb", "pgs"])
def test_argmax_mode_equals_predict_proba(preds_dir: Path, artifacts: Path, pair: str,
                                          ckpt: str) -> None:
    X = _notebook_features(artifacts, pair)
    ref = np.asarray(_model(artifacts, pair, ckpt).predict_proba(X))
    df = _read(preds_dir, pair, ckpt, "argmax")
    np.testing.assert_allclose(df[[f"p{k}" for k in range(9)]].to_numpy(), ref, rtol=0,
                               atol=CSV_ATOL)
    np.testing.assert_array_equal(df["pred"], ref.argmax(axis=1))


def test_batch_size_does_not_change_output(artifacts: Path, preds_dir: Path,
                                           tmp_path: Path) -> None:
    rc, _ = _run_export(["--artifacts", str(artifacts), "--out", str(tmp_path), "--pairs",
                         PAIRS[0], "--no-expect"])
    assert rc == 0
    for f in tmp_path.glob("*.csv.gz"):
        assert _sha(f) == _sha(preds_dir / f.name), f.name


# ----------------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------------


def test_manifest_schema(manifest: dict, preds_dir: Path, artifacts: Path) -> None:
    import catboost

    assert TOP_LEVEL_KEYS <= set(manifest)
    assert manifest["catboost_version"] == catboost.__version__
    assert manifest["numpy_version"] == np.__version__
    assert manifest["virtual_ensembles"] == M
    assert manifest["pooling"] == "linear (mean of per-member softmax)"
    assert manifest["created_utc"].endswith("Z")
    splits = _splits(artifacts)
    assert manifest["n_val"] == len(splits["val"]) and manifest["n_test"] == len(splits["test"])
    assert manifest["expectations_checked"] is False
    assert manifest["problems"] == []
    assert len(manifest["files"]) == 4 * len(PAIRS)
    order = [(f["image"] + "__" + f["text"], f["checkpoint"], f["mode"])
             for f in manifest["files"]]
    assert order == [(p, c, m) for p in PAIRS for c in ("cb", "pgs") for m in ("argmax", "pgs")]

    for entry in manifest["files"]:
        assert FILE_KEYS <= set(entry)
        pair = f"{entry['image']}__{entry['text']}"
        fpath = preds_dir / entry["file"]
        assert entry["file"] == f"preds__{pair}__{entry['checkpoint']}__{entry['mode']}.csv.gz"
        assert fpath.is_file() and entry["sha256"] == _sha(fpath)
        cbm = Path(entry["cbm_path"])
        assert cbm.name == f"{pair}__{entry['checkpoint']}.cbm" and cbm.is_file()
        assert entry["cbm_sha256"] == _sha(cbm)
        assert entry["tree_count"] == _model(artifacts, pair, entry["checkpoint"]).tree_count_
        assert entry["tree_count"] >= 2 * M + 1
        assert entry["posterior_sampling"] is (entry["checkpoint"] == "pgs")
        assert entry["expected_test"] is None and entry["reproduces_manuscript"] is None
        assert entry["virtual_ensembles"] == (M if entry["mode"] == "pgs" else None)

        df = pd.read_csv(fpath)
        for split in ("val", "test"):
            part = df[df["split"] == split]
            got = entry["metrics"][split]
            assert got["accuracy"] == pytest.approx(accuracy_score(part["label"], part["pred"]),
                                                    abs=1e-12)
            assert got["macro_f1"] == pytest.approx(
                f1_score(part["label"], part["pred"], labels=list(range(9)), average="macro",
                         zero_division=0), abs=1e-12)
        test_keys = set(entry["metrics"]["test"])
        extra = {"accuracy_loglin", "macro_f1_loglin", "mean_prob_std", "mean_mi"}
        assert set(entry["metrics"]["val"]) == {"accuracy", "macro_f1"}
        if entry["mode"] == "pgs":
            assert test_keys == {"accuracy", "macro_f1"} | extra
            part = df[df["split"] == "test"]
            t = entry["metrics"]["test"]
            assert t["mean_prob_std"] == pytest.approx(part["prob_std"].mean(), abs=1e-9)
            assert t["mean_mi"] == pytest.approx(part["mi"].mean(), abs=1e-9)
            assert t["accuracy_loglin"] == pytest.approx(
                accuracy_score(part["label"], part["pred_loglin"]), abs=1e-12)
        else:
            assert test_keys == {"accuracy", "macro_f1"}

    # top-1 disagreements of the deployed pair, recomputed from the CSVs
    a = _read(preds_dir, PAIRS[0], "pgs", "argmax")
    b = _read(preds_dir, PAIRS[0], "pgs", "pgs")
    test = a["split"] == "test"
    assert manifest["deployed_pair"] == PAIRS[0]
    assert manifest["top1_disagreements_pgs_vs_argmax"] == int(
        (a.loc[test, "pred"] != b.loc[test, "pred"]).sum())
    assert manifest["top1_disagreements_pgs_vs_argmax_val"] == int(
        (a.loc[~test, "pred"] != b.loc[~test, "pred"]).sum())


def test_expectation_run_warns_and_flags(expect_run: tuple[Path, str]) -> None:
    out, stdout = expect_run
    assert "WARNING" in stdout and "NOT REPRODUCED" in stdout
    man = _load_manifest(out)
    assert man["expectations_checked"] is True
    assert man["top1_disagreements_expected"] == 130
    assert man["manuscript_mismatches"]
    by_key = {(f["image"] + "__" + f["text"], f["checkpoint"], f["mode"]): f
              for f in man["files"]}
    d3 = "dinov3_large__mE5_large"
    assert by_key[(d3, "cb", "argmax")]["expected_test"] == {"accuracy": [0.7996],
                                                            "macro_f1": [0.7684]}
    assert by_key[(d3, "cb", "pgs")]["expected_test"] == {"accuracy": [0.7914],
                                                         "macro_f1": [0.76]}
    assert by_key[(d3, "pgs", "argmax")]["expected_test"] == {"accuracy": [0.8116],
                                                             "macro_f1": [0.7793]}
    assert by_key[(d3, "pgs", "pgs")]["expected_test"] == {"accuracy": [0.8073, 0.8074],
                                                          "macro_f1": [0.7747]}
    assert by_key[("eva02_large__mE5_large", "pgs", "pgs")]["expected_test"] == {
        "macro_f1": [0.7747]}
    for key, entry in by_key.items():
        if entry["expected_test"] is None:
            assert entry["reproduces_manuscript"] is None, key
        else:  # a synthetic fixture cannot hit the manuscript numbers
            assert entry["reproduces_manuscript"] is False, key


def test_expectation_logic_true_when_values_match(monkeypatch: pytest.MonkeyPatch,
                                                  artifacts: Path, preds_dir: Path,
                                                  tmp_path: Path) -> None:
    """Patch the expected table with the fixture's own values: flag must become True."""
    man = _load_manifest(preds_dir)
    d3 = PAIRS[0]
    entry = next(f for f in man["files"] if f["file"] == f"preds__{d3}__pgs__pgs.csv.gz")
    acc = round(entry["metrics"]["test"]["accuracy"], 4)
    f1 = round(entry["metrics"]["test"]["macro_f1"], 4)
    monkeypatch.setattr(exp, "EXPECTED_TEST",
                        {(d3, "pgs", "pgs"): {"accuracy": [0.1, acc], "macro_f1": [f1]}})
    monkeypatch.setattr(exp, "EXPECTED_SPLIT_SIZES", {})
    top1 = man["top1_disagreements_pgs_vs_argmax"]
    monkeypatch.setattr(exp, "EXPECTED_TOP1_DISAGREEMENTS", {d3: top1})
    rc, stdout = _run_export(["--artifacts", str(artifacts), "--out", str(tmp_path),
                              "--pairs", d3])
    assert rc == 0
    got = _load_manifest(tmp_path)
    flags = {f["file"]: f["reproduces_manuscript"] for f in got["files"]}
    assert flags.pop(f"preds__{d3}__pgs__pgs.csv.gz") is True
    assert set(flags.values()) == {None}
    assert got["manuscript_mismatches"] == []
    assert "NOT REPRODUCED" not in stdout


# ----------------------------------------------------------------------------------------
# robustness / validation
# ----------------------------------------------------------------------------------------


def _copy_artifacts(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_missing_checkpoint_is_skipped(artifacts: Path, tmp_path: Path) -> None:
    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    (art / "models" / "checkpoints" / f"{PAIRS[0]}__pgs.cbm").unlink()
    out = tmp_path / "out"
    rc, stdout = _run_export(["--artifacts", str(art), "--out", str(out), "--pairs", PAIRS[0],
                              "--no-expect"])
    assert rc == 0
    names = {f.name for f in out.glob("*.csv.gz")}
    assert names == {f"preds__{PAIRS[0]}__cb__argmax.csv.gz", f"preds__{PAIRS[0]}__cb__pgs.csv.gz"}
    man = _load_manifest(out)
    assert any("not found" in p for p in man["problems"])
    assert man["top1_disagreements_pgs_vs_argmax"] is None
    assert "PROBLEMS" in stdout


def test_label_out_of_range_is_fatal(artifacts: Path, tmp_path: Path) -> None:
    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    val = pd.read_csv(art / "splits" / "val.csv")
    val.loc[0, "label_id"] = 9
    val.to_csv(art / "splits" / "val.csv", index=False)
    with contextlib.redirect_stderr(io.StringIO()) as err:
        rc, _ = _run_export(["--artifacts", str(art), "--out", str(tmp_path / "o"),
                             "--no-expect"])
    assert rc == 2 and "label_id outside" in err.getvalue()


def test_row_id_beyond_embeddings_skips_pair(artifacts: Path, tmp_path: Path) -> None:
    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    test = pd.read_csv(art / "splits" / "test.csv")
    test.loc[0, "row_id"] = 10_000
    test.to_csv(art / "splits" / "test.csv", index=False)
    out = tmp_path / "o"
    rc, _ = _run_export(["--artifacts", str(art), "--out", str(out), "--pairs", PAIRS[0],
                         "--no-expect"])
    assert rc == 1  # nothing exported
    assert any("row_id 10000" in p for p in _load_manifest(out)["problems"])


def test_test_split_only(artifacts: Path, tmp_path: Path) -> None:
    rc, _ = _run_export(["--artifacts", str(artifacts), "--out", str(tmp_path), "--pairs",
                         PAIRS[0], "--splits", "test", "--no-expect"])
    assert rc == 0
    man = _load_manifest(tmp_path)
    assert man["n_val"] == 0 and man["n_test"] == len(_splits(artifacts)["test"])
    df = _read(tmp_path, PAIRS[0], "pgs", "pgs")
    assert set(df["split"]) == {"test"}
    assert set(man["files"][0]["metrics"]) == {"test"}


def test_fixture_is_realistic(artifacts: Path) -> None:
    """Quoted multi-line split fields parse; zero image rows (failed loads) reach val/test."""
    info = json.loads((artifacts / "fixture_info.json").read_text(encoding="utf-8"))
    splits = {s: pd.read_csv(artifacts / "splits" / f"{s}.csv") for s in ("train", "val", "test")}
    for s, df in splits.items():
        assert len(df) == info["split_sizes"][s]
        assert list(df.columns) == ["row_id", "gambar", "laporan", "label", "label_id"]
        assert df["laporan"].str.contains("\n").all() and df["laporan"].str.contains(",").all()
    zero = set(info["zero_image_rows"])
    img = np.load(artifacts / "embeddings" / "image" / "dinov3_large.npy")
    assert not np.abs(img[sorted(zero)]).any()
    for s in ("val", "test"):
        assert zero & set(splits[s]["row_id"]), s


def test_zero_image_rows_exported_like_crm(preds_dir: Path, artifacts: Path) -> None:
    info = json.loads((artifacts / "fixture_info.json").read_text(encoding="utf-8"))
    df = _read(preds_dir, PAIRS[0], "pgs", "pgs")
    rows = df[df["row_id"].isin(info["zero_image_rows"])]
    assert len(rows) >= 2
    assert np.isfinite(rows.drop(columns="split").to_numpy(dtype=np.float64)).all()


def test_non_multiclass_loss_is_skipped(artifacts: Path, tmp_path: Path) -> None:
    """A MultiClassOneVsAll model also returns (N, M, 9) raw values; it must not be pooled."""
    from catboost import CatBoostClassifier

    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    X = _notebook_features(art, PAIRS[0])
    y = pd.concat(_splits(art).values())["label_id"].to_numpy()
    ova = CatBoostClassifier(iterations=70, depth=2, loss_function="MultiClassOneVsAll",
                             verbose=False, random_seed=0, allow_writing_files=False)
    ova.fit(X, y)
    ova.save_model(str(art / "models" / "checkpoints" / f"{PAIRS[0]}__cb.cbm"))
    out = tmp_path / "out"
    rc, _ = _run_export(["--artifacts", str(art), "--out", str(out), "--pairs", PAIRS[0],
                         "--no-expect"])
    assert rc == 0
    man = _load_manifest(out)
    assert {f["checkpoint"] for f in man["files"]} == {"pgs"}
    assert any("MultiClassOneVsAll" in p for p in man["problems"])


def test_swapped_checkpoints_are_reported(artifacts: Path, tmp_path: Path) -> None:
    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    ck = art / "models" / "checkpoints"
    cb, pgs = ck / f"{PAIRS[0]}__cb.cbm", ck / f"{PAIRS[0]}__pgs.cbm"
    tmp = ck / "tmp.bin"
    cb.rename(tmp)
    pgs.rename(cb)
    tmp.rename(pgs)
    out = tmp_path / "out"
    rc, _ = _run_export(["--artifacts", str(art), "--out", str(out), "--pairs", PAIRS[0],
                         "--no-expect"])
    assert rc == 0
    problems = _load_manifest(out)["problems"]
    assert sum("posterior_sampling" in p for p in problems) == 2


def test_label_text_mismatch_is_reported(artifacts: Path, tmp_path: Path) -> None:
    art = _copy_artifacts(artifacts, tmp_path / "artifacts")
    val = pd.read_csv(art / "splits" / "val.csv")
    val.loc[0, "label"] = fx.TARGET_CLASSES[(int(val.loc[0, "label_id"]) + 1) % 9]
    val.to_csv(art / "splits" / "val.csv", index=False)
    out = tmp_path / "out"
    rc, _ = _run_export(["--artifacts", str(art), "--out", str(out), "--pairs", PAIRS[0],
                         "--no-expect"])
    assert rc == 0
    assert any("split val: 1 rows" in p for p in _load_manifest(out)["problems"])


def test_crm_feature_mismatch_is_fatal(artifacts: Path, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    if exp.load_crm(None, artifacts) is None:
        pytest.skip("crm package not importable")
    real = exp.build_features
    monkeypatch.setattr(exp, "build_features",
                        lambda *a: (real(*a) * np.float32(1.0001)).astype(np.float32))
    with contextlib.redirect_stderr(io.StringIO()) as err:
        rc, _ = _run_export(["--artifacts", str(artifacts), "--out", str(tmp_path),
                             "--pairs", PAIRS[0], "--no-expect"])
    assert rc == 2 and "differ from crm.fusion.early_fusion" in err.getvalue()
    assert not list(tmp_path.glob("*.csv.gz"))


def test_old_manifest_replaced_and_stale_files_reported(artifacts: Path, preds_dir: Path,
                                                        tmp_path: Path) -> None:
    for f in preds_dir.iterdir():  # previous full run: both pairs + manifest
        shutil.copy(f, tmp_path / f.name)
    rc, _ = _run_export(["--artifacts", str(artifacts), "--out", str(tmp_path), "--pairs",
                         PAIRS[0], "--no-expect"])
    assert rc == 0
    man = _load_manifest(tmp_path)
    assert {f["image"] for f in man["files"]} == {"dinov3_large"}
    stale = [p for p in man["problems"] if "earlier run" in p]
    assert len(stale) == 1 and "eva02_large" in stale[0]


@pytest.mark.parametrize("script", ["export_per_sample_predictions.py",
                                    "make_synthetic_fixture.py"])
def test_cli_help(script: str) -> None:
    res = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True,
                         text=True, timeout=120)
    assert res.returncode == 0 and "usage" in res.stdout
