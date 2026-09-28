"""Tests for analyze_uncertainty.py (stage 2).

All inputs are synthetic and generated here (or by tests/synthetic_per_sample.py) in the exact
CONTRACT.md section 2 schema; nothing depends on the stage-1 export script.
Run with:  python -m pytest tests/test_analysis.py -q
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analyze_uncertainty as au  # noqa: E402
from synthetic_per_sample import make_synthetic_preds, softmax  # noqa: E402

ASCII_MINUS_RE = re.compile(r"(?<![\w.])-(?=\d)")   # same rule as fill_manuscript.py


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------


def naive_ece(probs: np.ndarray, labels: np.ndarray) -> float:
    conf = probs.max(axis=1)
    acc = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, 16)
    total = 0.0
    for i in range(15):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            total += m.sum() / conf.size * abs(acc[m].mean() - conf[m].mean())
    return total


def sample_labels(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(p.shape[0])[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def naive_selective(score: np.ndarray, correct: np.ndarray) -> tuple[float, float, float]:
    order = sorted(range(score.size), key=lambda i: (score[i], i))
    n = score.size
    errors, risks, accs = 0, [], []
    for k, i in enumerate(order, start=1):
        errors += int(not correct[i])
        risks.append(errors / k)
        accs.append(1 - errors / k)
    aurc = sum(risks) / n
    k80 = int(round(0.8 * n))
    acc80 = accs[k80 - 1]
    cov90 = 0.0
    for k in range(1, n + 1):
        if accs[k - 1] >= 0.9 - 1e-12:
            cov90 = k / n
    return aurc, acc80, cov90


def count_sentences(text: str) -> int:
    t = text.replace("vs. ", "vs ").replace("e.g. ", "eg ")
    return len([s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", t) if s.strip()])


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("synthetic")
    preds, out = base / "preds", base / "results"
    make_synthetic_preds(preds, n_val=1500, n_test=1500, seed=11)
    results = au.run(preds, out, n_boot=200, seed=42)
    return preds, out, results


# ------------------------------------------------------------------------------------------
# (0) formatting helpers
# ------------------------------------------------------------------------------------------


def test_formatting_helpers():
    m = au.MINUS
    assert au.fmt_metric(0.81164) == "0.8116"
    assert au.fmt_pp(-0.0043) == f"{m}0.43"
    assert au.fmt_pp(0.0012) == "+0.12"
    assert au.fmt_pp(-0.00001) == "0.00"
    assert au.fmt_pp_ci(-0.003, 0.0055) == f"[{m}0.30, 0.55]"
    assert au.fmt_ci(0.0101, 0.015) == "[0.0101, 0.0150]"
    assert au.fmt_metric_ci(0.0123, 0.0101, 0.015) == "0.0123 [0.0101, 0.0150]"
    assert au.fmt_p(0.04213) == "0.0421"
    assert au.fmt_p(0.2371) == "0.237"
    assert au.fmt_p(0.5) == "0.500"
    assert au.fmt_p(1.0) == "1.00"
    assert au.fmt_p(0.99951) == "1.00"
    assert au.fmt_p(0.001) == "0.00100"
    assert au.fmt_p(0.00099) == "< 0.001"
    assert au.fmt_p(1e-30) == "< 0.001"
    assert au.p_phrase(0.0004) == "p < 0.001"
    assert au.p_phrase(0.912) == "p = 0.912"
    assert au.fmt_int(9266) == "9,266"
    assert au.fmt_temp(1.2345) == "1.23"
    assert au.fmt_cov(0.78149) == "0.781"
    assert au.fmt_aurc(0.06123) == "6.12"
    assert au.fmt_fixed(-0.00004, 4) == "0.0000"


# ------------------------------------------------------------------------------------------
# (1) ECE
# ------------------------------------------------------------------------------------------


def test_ece_calibrated_is_small_and_overconfident_is_large():
    rng = np.random.default_rng(0)
    n = 20000
    z = rng.normal(size=(n, 9)) * 2.5
    p = softmax(z)
    labels = sample_labels(p, rng)
    assert au.ece_score(p, labels) < 0.02
    labels_soft = sample_labels(softmax(z / 3.0), rng)      # truth much softer than p -> overconfident
    ece_over = au.ece_score(p, labels_soft)
    assert ece_over > 0.10
    conf_gap = p.max(axis=1).mean() - (p.argmax(axis=1) == labels_soft).mean()
    assert conf_gap > 0.10


def test_ece_equals_naive_loop_including_bin_edges():
    rng = np.random.default_rng(1)
    n = 3000
    p = softmax(rng.normal(size=(n, 9)) * 2.0)
    labels = rng.integers(0, 9, size=n)
    assert au.ece_score(p, labels) == pytest.approx(naive_ece(p, labels), abs=1e-12)
    # confidences exactly on bin edges belong to the lower bin (i/15, (i+1)/15]
    edges = np.linspace(0, 1, 16)
    conf = np.concatenate([edges[2:], rng.uniform(0.12, 1.0, 200)])
    q = np.zeros((conf.size, 9))
    q[:, 0] = conf
    q[:, 1:] = ((1 - conf) / 8)[:, None]
    lab = rng.integers(0, 9, size=conf.size)
    assert au.ece_score(q, lab) == pytest.approx(naive_ece(q, lab), abs=1e-12)
    naive_bins = np.array([next(i for i in range(15) if edges[i] < c <= edges[i + 1]) for c in conf])
    assert np.array_equal(au.bin_index(conf), naive_bins)


def test_bootstrap_ece_matches_resampled_naive():
    rng = np.random.default_rng(2)
    n = 1200
    p = softmax(rng.normal(size=(n, 9)) * 2.0)
    labels = sample_labels(p, rng)
    boot = au.Bootstrap(n, 50, seed=42)
    row = au.calibration_row(p, labels, p.argmax(axis=1), boot)
    assert row["ece"] == pytest.approx(naive_ece(p, labels), abs=1e-12)
    # recompute the first resamples naively
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels).astype(float)
    gap = np.zeros((n, 15))
    gap[np.arange(n), au.bin_index(conf)] = correct - conf
    sums = boot.sums(gap) / n
    for r in range(5):
        idx = boot.first_indices[r]
        assert np.abs(sums[r]).sum() == pytest.approx(naive_ece(p[idx], labels[idx]), abs=1e-12)
    # the index matrix is default_rng(seed).integers(0, n, size=(B, n))
    ref = np.random.default_rng(42).integers(0, n, size=(50, n))
    assert np.array_equal(boot.first_indices, ref[:5])
    assert np.array_equal(boot.counts[7], np.bincount(ref[7], minlength=n))


def test_bootstrap_speed_full_size():
    rng = np.random.default_rng(3)
    n = 9266
    p = softmax(rng.normal(size=(n, 9)) * 2.0)
    labels = sample_labels(p, rng)
    t0 = time.perf_counter()
    boot = au.Bootstrap(n, 2000, seed=42)
    row = au.calibration_row(p, labels, p.argmax(axis=1), boot)
    elapsed = time.perf_counter() - t0
    assert row["ece_ci"][0] <= row["ece"] + 0.01
    assert elapsed < 30.0, f"ECE/accuracy bootstrap took {elapsed:.1f} s"


# ------------------------------------------------------------------------------------------
# (2) McNemar and Holm
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed,pa,pb,n", [(0, 0.8, 0.8, 500), (1, 0.82, 0.78, 3000), (2, 0.9, 0.6, 200),
                                          (3, 0.5, 0.5, 40)])
def test_mcnemar_matches_statsmodels(seed, pa, pb, n):
    rng = np.random.default_rng(seed)
    a = rng.random(n) < pa
    b = rng.random(n) < pb
    bb, cc, p = au.mcnemar_exact(a, b)
    assert bb == int(np.sum(a & ~b)) and cc == int(np.sum(~a & b))
    table = [[int(np.sum(a & b)), bb], [cc, int(np.sum(~a & ~b))]]
    assert p == pytest.approx(mcnemar(table, exact=True).pvalue, rel=1e-9, abs=1e-15)


def test_mcnemar_no_discordant_pairs():
    a = np.array([True, False, True])
    assert au.mcnemar_exact(a, a) == (0, 0, 1.0)


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(4)
    for m in (2, 3, 5):
        p = rng.uniform(0, 0.2, m)
        assert np.allclose(au.holm(p), multipletests(p, method="holm")[1])
    assert au.holm([0.03, 0.03]) == [0.06, 0.06]


# ------------------------------------------------------------------------------------------
# (3) paired bootstrap and vectorised macro-F1
# ------------------------------------------------------------------------------------------


def _view(name: str, label: np.ndarray, pred: np.ndarray) -> au.View:
    probs = np.full((label.size, 9), 0.02)
    probs[np.arange(label.size), pred] = 0.84
    return au.View(name=name, row_id=np.arange(label.size), label=label, pred=pred, probs=probs, extra={})


def _noisy_pred(label: np.ndarray, acc: float, rng: np.random.Generator) -> np.ndarray:
    wrong = (label + rng.integers(1, 9, size=label.size)) % 9
    return np.where(rng.random(label.size) < acc, label, wrong)


def test_paired_bootstrap_ci_contains_true_difference():
    """Coverage study: the 95% paired interval must contain the true Δaccuracy (0.03) in ~95% of datasets.

    A single draw misses the truth 5% of the time by construction, so 60 independent data sets are
    simulated (seeded, deterministic); nominal coverage 0.95, binomial SD ~0.03 -> require >= 0.85.
    """
    hits, reps = 0, 60
    for s in range(reps):
        rng = np.random.default_rng(100 + s)
        n = 2000
        label = rng.integers(0, 9, size=n)
        a = _view("A", label, _noisy_pred(label, 0.82, rng))
        b = _view("B", label, _noisy_pred(label, 0.79, rng))
        r = au.paired_comparison(a, b, au.Bootstrap(n, 400, seed=42))
        lo, hi = r["d_accuracy_ci"]
        hits += lo < 0.03 < hi
        assert lo <= r["d_accuracy"] <= hi
        assert r["d_macro_f1_ci"][0] <= r["d_macro_f1"] <= r["d_macro_f1_ci"][1]
        assert r["d_accuracy"] == pytest.approx(a.correct.mean() - b.correct.mean())
    assert hits / reps >= 0.85, f"coverage {hits / reps:.2f}"
    # full-size paired run (2,000 x 9,266): interval brackets the observed difference
    rng = np.random.default_rng(5)
    n = 9266
    label = rng.integers(0, 9, size=n)
    a = _view("A", label, _noisy_pred(label, 0.82, rng))
    b = _view("B", label, _noisy_pred(label, 0.79, rng))
    r = au.paired_comparison(a, b, au.Bootstrap(n, 2000, seed=42))
    assert r["d_accuracy_ci"][0] < r["d_accuracy"] < r["d_accuracy_ci"][1]
    assert r["d_accuracy_ci"][1] - r["d_accuracy_ci"][0] == pytest.approx(
        2 * 1.96 * np.sqrt(np.var(a.correct.astype(float) - b.correct.astype(float)) / n), rel=0.1)


def test_vectorised_macro_f1_equals_sklearn():
    rng = np.random.default_rng(6)
    n = 300
    prior = np.array([0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.007, 0.003])
    label = rng.choice(9, size=n, p=prior / prior.sum())
    pred = _noisy_pred(label, 0.7, rng)
    onehot = au.confusion_onehot(label, pred)
    full = au.macro_f1_from_confusion(onehot.sum(axis=0).reshape(9, 9))
    assert full == pytest.approx(f1_score(label, pred, average="macro", labels=list(range(9)), zero_division=0),
                                 abs=1e-12)
    boot = au.Bootstrap(n, 40, seed=42)
    cms = boot.sums(onehot).reshape(-1, 9, 9)
    f1s = au.macro_f1_from_confusion(cms)
    for r in range(5):
        idx = boot.first_indices[r]
        ref = f1_score(label[idx], pred[idx], average="macro", labels=list(range(9)), zero_division=0)
        assert f1s[r] == pytest.approx(ref, abs=1e-12)
    # a class absent from both y_true and y_pred scores 0 (zero_division=0)
    y = np.array([0, 0, 1, 1, 2])
    yp = np.array([0, 1, 1, 1, 2])
    cm = au.confusion_onehot(y, yp).sum(axis=0).reshape(9, 9)
    assert au.macro_f1_from_confusion(cm) == pytest.approx(
        f1_score(y, yp, average="macro", labels=list(range(9)), zero_division=0))


# ------------------------------------------------------------------------------------------
# (4) AUROC and selective prediction
# ------------------------------------------------------------------------------------------


def test_auroc_perfect_and_random():
    rng = np.random.default_rng(7)
    n = 20000
    incorrect = rng.random(n) < 0.2
    perfect = np.where(incorrect, 1.0 + rng.random(n), rng.random(n))
    assert au.auroc(perfect, incorrect) == pytest.approx(1.0)
    assert au.auroc(rng.random(n), incorrect) == pytest.approx(0.5, abs=0.02)
    assert np.isnan(au.auroc(rng.random(10), np.zeros(10, dtype=bool)))


def test_weighted_auroc_matches_sklearn_with_ties_and_resamples():
    rng = np.random.default_rng(8)
    n = 800
    incorrect = rng.random(n) < 0.25
    score = np.round(np.where(incorrect, rng.normal(0.6, 0.2, n), rng.normal(0.4, 0.2, n)), 2)  # many ties
    one = au.auroc_from_weights(np.ones((1, n)), score, incorrect)[0]
    assert one == pytest.approx(roc_auc_score(incorrect, score), abs=1e-12)
    boot = au.Bootstrap(n, 30, seed=42)
    vals = boot.map(lambda w: au.auroc_from_weights(w, score, incorrect))
    for r in range(5):
        idx = boot.first_indices[r]
        assert vals[r] == pytest.approx(roc_auc_score(incorrect[idx], score[idx]), abs=1e-12)


@pytest.mark.parametrize("seed", [9, 10])
def test_selective_metrics_match_naive_loops(seed):
    rng = np.random.default_rng(seed)
    n = 1500
    correct = rng.random(n) < 0.8
    score = np.round(np.where(correct, rng.normal(0.3, 0.2, n), rng.normal(0.6, 0.2, n)), 2)  # ties
    res = au.selective_metrics(score, correct)
    aurc, acc80, cov90 = naive_selective(score, correct)
    assert res.aurc == pytest.approx(aurc, abs=1e-12)
    assert res.acc_at_coverage == pytest.approx(acc80, abs=1e-12)
    assert res.coverage_at_accuracy == pytest.approx(cov90, abs=1e-12)


def test_coverage_at_90_is_zero_when_never_reached():
    correct = np.array([False, True, True, False, True])
    score = np.arange(5, dtype=float)
    assert au.selective_metrics(score, correct).coverage_at_accuracy == 0.0


# ------------------------------------------------------------------------------------------
# (5) temperature scaling
# ------------------------------------------------------------------------------------------


def test_temperature_scaling_recovers_t2():
    rng = np.random.default_rng(12)
    n = 20000
    z = rng.normal(size=(n, 9)) * 3.0
    labels = sample_labels(softmax(z / 2.0), rng)     # true logits are z / 2 -> T = 2
    fit = au.fit_temperature(softmax(z), labels)
    assert fit["temperature"] == pytest.approx(2.0, abs=0.1)
    assert fit["val_nll_after"] < fit["val_nll_before"]
    scaled = au.apply_temperature(softmax(z), fit["temperature"])
    assert np.array_equal(scaled.argmax(axis=1), z.argmax(axis=1))
    assert au.ece_score(scaled, labels) < au.ece_score(softmax(z), labels)


# ------------------------------------------------------------------------------------------
# (6) end-to-end outputs: fill_values, prose, claims
# ------------------------------------------------------------------------------------------


def test_fill_values_cover_required_keys_and_use_unicode_minus(synthetic_run):
    _, out, _ = synthetic_run
    fv = json.loads((out / "fill_values.json").read_text(encoding="utf-8"))
    assert list(fv) == au.REQUIRED_KEYS
    assert len(au.REQUIRED_KEYS) == len(set(au.REQUIRED_KEYS)) == 25 + 1 + 25 + 24 + 3 + 6
    for k, v in fv.items():
        assert isinstance(v, str) and v.strip() and "\n" not in v, k
        assert not ASCII_MINUS_RE.search(v), f"{k}: ASCII hyphen-minus used as a negative sign: {v!r}"
        assert "nan" not in v.lower().split() and "n/a" not in v, k
        if k.startswith(("T5_", "T6_", "T7_")):
            assert "-" not in v, f"{k}: numeric field contains an ASCII hyphen-minus: {v!r}"
    num = r"[0-9]\.[0-9]{4}"
    assert re.fullmatch(num, fv["T5_R3_ACC"])
    assert re.fullmatch(rf"{num} \[{num}, {num}\]", fv["T5_R4_ECE"])
    assert re.fullmatch(rf"{num} \[{num}, {num}\]", fv["T7_R5_AUROC"])
    assert re.fullmatch(r"[0-9]+\.[0-9]{2}", fv["T5_TEMP"])
    assert re.fullmatch(r"[0-9,]+ / [0-9,]+", fv["T6_R4_BC"])
    pp = r"(?:[+−][0-9]+\.[0-9]{2}|0\.00)"
    ppci = r"(?:−?[0-9]+\.[0-9]{2})"
    for r in range(1, 6):
        assert re.fullmatch(rf"{pp} \[{ppci}, {ppci}\]", fv[f"T6_R{r}_DACC"]), fv[f"T6_R{r}_DACC"]
        assert re.fullmatch(r"< 0\.001|[01]\.[0-9]+", fv[f"T6_R{r}_P"])
    assert re.fullmatch(r"[0-9]\.[0-9]{3}", fv["T7_R1_COV90"])
    assert re.fullmatch(rf"{num} vs\. {num}", fv["T1_ECE"])
    assert re.fullmatch(rf"Δmacro-F1 = {pp} pp; Holm-adjusted p (?:= [01]\.[0-9]+|< 0\.001)", fv["T1_ENC"])


def test_prose_numbers_come_from_table_values(synthetic_run):
    _, out, _ = synthetic_run
    fv = json.loads((out / "fill_values.json").read_text(encoding="utf-8"))
    tables = " ".join(v for k, v in fv.items() if k.startswith(("T1_", "T5_", "T6_", "T7_")))
    # supplementary paired ΔAUROC (tables.md / results.json) may be quoted to avoid "comparable" claims
    rj = json.loads((out / "results.json").read_text(encoding="utf-8"))
    tables += " " + " ".join(f"{au.fmt_fixed(d['d_auroc'], 4, plus=True)} {au.fmt_ci(*d['d_auroc_ci'])}"
                             for d in rj["selective"]["paired_auroc_differences"])
    for k in ("TXT_CALIBRATION", "TXT_PAIRED", "TXT_SELECTIVE", "TXT_IMPLICATION", "TXT_CONCLUSION"):
        for tok in re.findall(r"[+−]?[0-9]+\.[0-9]{2,4}", fv[k]):
            if re.fullmatch(r"[0-9]+\.[0-9]{2}", tok):        # unsigned 2-dp: T, gaps in pp
                continue
            assert tok in tables, f"{k}: number {tok} does not appear in any table value"
    bounds = {"TXT_CALIBRATION": (2, 4), "TXT_PAIRED": (2, 4), "TXT_SELECTIVE": (2, 4),
              "TXT_IMPLICATION": (2, 3), "TXT_CONCLUSION": (1, 2)}
    for k, (lo, hi) in bounds.items():
        assert lo <= count_sentences(fv[k]) <= hi, (k, fv[k])
    assert fv["TXT_REPRO"][0].islower() and fv["TXT_REPRO"].endswith(".")


def test_outputs_and_figure_sizes(synthetic_run):
    _, out, results = synthetic_run
    for name in ("results.json", "fill_values.json", "tables.md", "claims_check.md", "fig_reliability.png",
                 "fig_selective.png"):
        assert (out / name).is_file(), name
    for name in ("table1_additions.csv", "table4_reproduction.csv", "table5_calibration.csv",
                 "table6_paired.csv", "table7_selective.csv", "reliability_bins.csv", "selective_curves.csv"):
        assert (out / "tables" / name).is_file(), name
    with Image.open(out / "fig_reliability.png") as im:
        assert im.size == au.FIG_RELIABILITY_PX == (2100, 780)
    with Image.open(out / "fig_selective.png") as im:
        assert im.size == au.FIG_SELECTIVE_PX == (2100, 840)
    rj = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert rj["meta"]["bootstrap_resamples"] == 200 and rj["meta"]["n_test"] == 1500
    assert [c["id"] for c in rj["claims"]] == ["C1", "C2", "C3", "C4", "C5", "C6"]
    assert all(c["verdict"] in {"SUPPORTED", "QUALIFIED", "CONTRADICTED"} for c in rj["claims"])
    claims_md = (out / "claims_check.md").read_text(encoding="utf-8")
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert f"## {cid}" in claims_md
    # recomputed Table 4 numbers equal the synthetic manifest (expected_test = own rounded values)
    assert all(r["reproduces_computed"] in (True, None) for r in rj["reproduction"]["rows"])
    manifest = json.loads((synthetic_run[0] / "manifest.json").read_text(encoding="utf-8"))
    assert rj["reproduction"]["top1_disagreements_pgs_vs_argmax"] == manifest["top1_disagreements_pgs_vs_argmax"]


def test_figures_are_written_at_exact_size_for_edge_data(synthetic_run, tmp_path):
    """Figure code must keep the pixel contract even when MI has no zeros and all bins are sparse."""
    _, _, results = synthetic_run
    res = copy.copy(results)
    sel = dict(results["selective"])
    scores = dict(sel["_scores"])
    scores["S5"] = scores["S5"] + 1e-6
    sel["_scores"] = scores
    res["selective"] = sel
    au.figure_selective(res, tmp_path / "s.png")
    au.figure_reliability(res, tmp_path / "r.png")
    with Image.open(tmp_path / "s.png") as im:
        assert im.size == (2100, 840)
    with Image.open(tmp_path / "r.png") as im:
        assert im.size == (2100, 780)


# ------------------------------------------------------------------------------------------
# prose / claim branches (every outcome direction)
# ------------------------------------------------------------------------------------------


def _row(results, section, key, value):
    return next(r for r in results[section]["rows"] if r[key] == value)


def _check_text(t: str) -> None:
    assert t and t.endswith(".") and "nan" not in t.lower().split()
    assert not ASCII_MINUS_RE.search(t), t


def test_calibration_prose_branches(synthetic_run):
    _, _, base = synthetic_run
    cases = {
        "improves": ((0.06, 0.055, 0.065), (0.03, 0.025, 0.035), "improves calibration"),
        "worsens": ((0.03, 0.025, 0.035), (0.06, 0.055, 0.065), "worsens calibration"),
        "unchanged": ((0.04, 0.035, 0.05), (0.042, 0.037, 0.052), "does not change calibration detectably"),
    }
    for _, (e3, e4, phrase) in cases.items():
        res = copy.deepcopy({k: v for k, v in base.items() if k != "selective"})
        r3, r4 = _row(res, "calibration", "row", "R3"), _row(res, "calibration", "row", "R4")
        r3["ece"], r3["ece_ci"] = e3[0], e3[1:]
        r4["ece"], r4["ece_ci"] = e4[0], e4[1:]
        t = au.txt_calibration(res)
        _check_text(t)
        assert phrase in t
        assert 2 <= count_sentences(t) <= 4
    res = copy.deepcopy({k: v for k, v in base.items() if k != "selective"})
    r3, r4, r5 = (_row(res, "calibration", "row", r) for r in ("R3", "R4", "R5"))
    r3["mean_confidence"], r3["accuracy"] = 0.80, 0.85        # under-confident served path
    r4["mean_confidence"], r4["accuracy"] = 0.90, 0.85        # over-confident PGS path
    r5["ece"], r5["ece_ci"] = r3["ece"] + 0.01, (r3["ece_ci"][0] + 0.01, r3["ece_ci"][1] + 0.01)
    res["calibration"]["temperature"]["temperature"] = 0.8
    t = au.txt_calibration(res)
    _check_text(t)
    assert "underconfident" in t and "overconfident" in t and "sharpens" in t and "does not reduce" in t
    res["calibration"]["temperature"]["temperature"] = 1.001
    r5["ece"] = r3["ece"]
    assert "practically unchanged" in au.txt_calibration(res)
    r3["brier"], r4["brier"] = 0.30, 0.25
    r3["nll"], r4["nll"] = 0.70, 0.80
    t = au.txt_calibration(res)
    assert "whereas the NLL increases" in t and "the Brier score decreases" in t


def _set_paired(res, row, *, d, ci, p, p_holm, df1=None, f1ci=None):
    r = _row(res, "paired", "row", row)
    r.update(d_accuracy=d, d_accuracy_ci=ci, p_mcnemar=p, p_holm=p_holm,
             d_macro_f1=d if df1 is None else df1, d_macro_f1_ci=ci if f1ci is None else f1ci)
    return r


def test_paired_prose_and_claims_branches(synthetic_run):
    _, _, base = synthetic_run
    res = copy.deepcopy({k: v for k, v in base.items() if not k.startswith("_")})
    for row in ("R1", "R2", "R3"):
        _set_paired(res, row, d=0.001, ci=(-0.004, 0.006), p=0.6, p_holm=1.0)
    _set_paired(res, "R4", d=-0.0043, ci=(-0.007, -0.002), p=0.001, p_holm=0.002)
    _set_paired(res, "R5", d=0.0063, ci=(-0.002, 0.014), p=0.2, p_holm=0.2)
    t = au.txt_paired(res)
    _check_text(t)
    assert "none of the three" in t and "not evidence of equivalence" in t
    assert "confirm that the naive +0.63 pp macro-F1 difference" in t and "is not statistically significant" in t
    claims = {c["id"]: c for c in au.evaluate_claims(res)}
    assert claims["C1"]["verdict"] == "SUPPORTED"
    assert claims["C2"]["verdict"] == "SUPPORTED" and claims["C2"]["refinement"]
    assert claims["C3"]["verdict"] == "SUPPORTED"

    _set_paired(res, "R1", d=0.01, ci=(0.004, 0.016), p=0.001, p_holm=0.003)
    _set_paired(res, "R5", d=0.0063, ci=(0.001, 0.012), p=0.01, p_holm=0.02)
    _set_paired(res, "R4", d=-0.002, ci=(-0.005, 0.001), p=0.3, p_holm=0.3)
    t = au.txt_paired(res)
    _check_text(t)
    assert "separates DINOv3-L from EVA-02-L" in t and "differences between" in t
    assert "is statistically significant (" in t
    claims = {c["id"]: c for c in au.evaluate_claims(res)}
    assert claims["C1"]["verdict"] == "QUALIFIED"
    assert claims["C2"]["verdict"] == "CONTRADICTED" and claims["C2"]["suggestion"]
    assert claims["C3"]["verdict"] == "CONTRADICTED"

    for row in ("R1", "R2", "R3"):
        _set_paired(res, row, d=0.01, ci=(0.004, 0.016), p=0.001, p_holm=0.003)
    _set_paired(res, "R5", d=0.0063, ci=(-0.001, 0.012), p=0.01, p_holm=0.07)   # McNemar only, not after Holm
    _set_paired(res, "R4", d=0.004, ci=(0.001, 0.007), p=0.01, p_holm=0.02)
    t = au.txt_paired(res)
    _check_text(t)
    # only the unadjusted McNemar p is below 0.05: the prose must not call the difference significant
    assert "all three" in t and "disagree" not in t
    assert "is not statistically significant after Holm adjustment" in t and "only nominally significant" in t
    claims = {c["id"]: c for c in au.evaluate_claims(res)}
    assert claims["C1"]["verdict"] == "CONTRADICTED"
    assert claims["C2"]["verdict"] == "CONTRADICTED"
    assert "only nominally significant" in claims["C2"]["suggestion"]
    assert "Holm-adjusted p = 0.0700" in claims["C2"]["suggestion"]

    _set_paired(res, "R5", d=0.0063, ci=(-0.001, 0.012), p=0.01, p_holm=0.02)   # McNemar also after Holm
    t = au.txt_paired(res)
    _check_text(t)
    assert "disagree" in t and "only nominally" not in t
    c2 = {c["id"]: c for c in au.evaluate_claims(res)}["C2"]
    assert c2["verdict"] == "CONTRADICTED" and "only nominally" not in c2["suggestion"]

    _set_paired(res, "R1", d=0.001, ci=(0.0005, 0.003), p=0.2, p_holm=0.4)   # CI excludes 0, test n.s.
    _set_paired(res, "R5", d=0.0063, ci=(0.001, 0.012), p=0.2, p_holm=0.2)
    t = au.txt_paired(res)
    _check_text(t)
    assert "do not fully agree" in t and "excludes zero, but McNemar" in t
    assert {c["id"]: c for c in au.evaluate_claims(res)}["C3"]["verdict"] == "QUALIFIED"
    assert count_sentences(t) <= 4


def test_selective_prose_and_c4_branches(synthetic_run):
    _, _, base = synthetic_run
    res = copy.deepcopy({k: v for k, v in base.items() if not k.startswith("_")})
    res["selective"] = {k: v for k, v in base["selective"].items() if not k.startswith("_")}
    res["selective"] = copy.deepcopy(res["selective"])
    s1, s5 = _row(res, "selective", "score", "S1"), _row(res, "selective", "score", "S5")
    # QUALIFIED: MI above chance but clearly worse than served 1 - max p
    s1.update(auroc=0.88, auroc_ci=(0.87, 0.89))
    s5.update(auroc=0.75, auroc_ci=(0.73, 0.77))
    claims = au.evaluate_claims(res)
    assert {c["id"]: c for c in claims}["C4"]["verdict"] == "QUALIFIED"
    for fn in (au.txt_selective, lambda r: au.txt_implication(r, claims), lambda r: au.txt_conclusion(r, claims)):
        _check_text(fn(res))
    assert "lower than that of" in au.txt_selective(res)
    assert "qualify the claim" in au.txt_implication(res, claims)
    assert "without PGS" in au.txt_conclusion(res, claims)
    # CONTRADICTED: MI interval includes 0.5
    s5.update(auroc=0.51, auroc_ci=(0.49, 0.53))
    claims = au.evaluate_claims(res)
    assert {c["id"]: c for c in claims}["C4"]["verdict"] == "CONTRADICTED"
    assert "chance" in au.txt_selective(res) and "do not support" in au.txt_implication(res, claims)
    # SUPPORTED and MI best (non-overlapping): deferral via MI needs native PGS
    for r in res["selective"]["rows"]:
        r.update(auroc=0.80, auroc_ci=(0.79, 0.81))
    s5.update(auroc=0.93, auroc_ci=(0.92, 0.94))
    claims = au.evaluate_claims(res)
    assert {c["id"]: c for c in claims}["C4"]["verdict"] == "SUPPORTED"
    t = au.txt_selective(res)
    assert "the mutual information has the highest error-detection AUROC" in t and "higher than that of" in t
    assert "requires native PGS inference" in au.txt_implication(res, claims)
    # no coverage reaches 90%, and deferral does not help
    s5.update(coverage_at_90=0.0, k_at_90=0, acc_at_80=0.5)
    assert "no coverage level reaches" in au.txt_selective(res)
    assert "not recommended" in au.txt_implication(res, claims)
    assert 2 <= count_sentences(au.txt_implication(res, claims)) <= 3


def test_repro_prose_branches(synthetic_run):
    _, _, base = synthetic_run
    res = copy.deepcopy({k: v for k, v in base.items() if not k.startswith("_")})
    dep = res["meta"]["deployed"]
    assert au.txt_repro(res, dep) == "to the fourth decimal place for all four checkpoint–mode combinations."
    rows = [r for r in res["reproduction"]["rows"] if r["pair"] == dep]
    bad = next(r for r in rows if (r["checkpoint"], r["mode"]) == ("pgs", "pgs"))
    bad["checks"]["accuracy"].update(matches=False, expected=[0.8073, 0.8074], deviation_pp=0.12)
    t = au.txt_repro(res, dep)
    _check_text(t)
    assert t.startswith("to the fourth decimal place for three of the four") and "instead of 0.8073 or 0.8074" in t
    for r in rows:
        r["checks"] = {}
    assert "cannot be verified" in au.txt_repro(res, dep)
    claims = {c["id"]: c for c in au.evaluate_claims(res)}
    assert claims["C6"]["verdict"] in {"SUPPORTED", "CONTRADICTED"}


# ------------------------------------------------------------------------------------------
# schema validation
# ------------------------------------------------------------------------------------------


def _copy_preds(src: Path, dst: Path) -> Path:
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_schema_violations_are_rejected(synthetic_run, tmp_path):
    src = synthetic_run[0]
    name = "preds__dinov3_large__mE5_large__pgs__argmax.csv.gz"

    d1 = _copy_preds(src, tmp_path / "cols")
    df = pd.read_csv(d1 / name)
    df[["label", "row_id"] + [c for c in df.columns if c not in ("label", "row_id")]].to_csv(
        d1 / name, index=False, compression="gzip")
    with pytest.raises(au.SchemaError, match="columns"):
        au.load_prediction_dir(d1)

    d2 = _copy_preds(src, tmp_path / "labels")
    df = pd.read_csv(d2 / name)
    test_rows = df.index[df["split"] == "test"]
    df.loc[test_rows[0], "label"] = (df.loc[test_rows[0], "label"] + 1) % 9
    df.to_csv(d2 / name, index=False, float_format="%.10g", compression="gzip")
    with pytest.raises(au.SchemaError, match="labels differ"):
        au.run(d2, tmp_path / "out2", n_boot=10, make_figures=False)

    d3 = _copy_preds(src, tmp_path / "pred")
    df = pd.read_csv(d3 / name)
    df.loc[0, "pred"] = (int(np.argmax(df.loc[0, au.PROB_COLS].to_numpy())) + 1) % 9
    df.to_csv(d3 / name, index=False, float_format="%.10g", compression="gzip")
    with pytest.raises(au.SchemaError, match="not the argmax"):
        au.load_prediction_dir(d3)

    d4 = _copy_preds(src, tmp_path / "missing")
    (d4 / "preds__eva02_large__mE5_large__pgs__pgs.csv.gz").unlink()
    with pytest.raises(au.SchemaError, match="missing"):
        au.load_prediction_dir(d4)


# ------------------------------------------------------------------------------------------
# review regressions: hedging / overclaiming in generated prose and claim suggestions
# ------------------------------------------------------------------------------------------


def _plain(base):
    res = copy.deepcopy({k: v for k, v in base.items() if not k.startswith("_") and k != "selective"})
    res["selective"] = copy.deepcopy({k: v for k, v in base["selective"].items() if not k.startswith("_")})
    return res


def _set_d51(res, d, ci):
    for x in res["selective"]["paired_auroc_differences"]:
        if (x["a"], x["b"]) == ("S5", "S1"):
            x.update(d_auroc=d, d_auroc_ci=list(ci))


def test_repro_every_branch_continues_the_fixed_opening(synthetic_run):
    """TXT_REPRO follows "The exported records reproduce ... values of Table 4 " and must stay grammatical."""
    _, _, base = synthetic_run
    res = _plain(base)
    dep = res["meta"]["deployed"]
    rows = [r for r in res["reproduction"]["rows"] if r["pair"] == dep]
    for r in rows:
        r["checks"] = {}
    t = au.txt_repro(res, dep)
    assert t.startswith("to an extent that cannot be verified here") and "reproduce" not in t
    res = _plain(base)
    rows = [r for r in res["reproduction"]["rows"] if r["pair"] == dep]
    for r in rows:
        for c in r["checks"].values():
            c.update(matches=False, deviation_pp=0.3)
    t = au.txt_repro(res, dep)
    assert t.startswith("only approximately: they give ") and "exported records" not in t
    rows[0]["checks"]["accuracy"].update(matches=True)
    rows[0]["checks"]["macro_f1"].update(matches=True)
    t = au.txt_repro(res, dep)
    assert t.startswith("to the fourth decimal place for one of the four checkpoint–mode combinations but give ")


def test_calibration_prose_never_calls_overlap_unchanged_and_respects_argmax_flips(synthetic_run):
    _, _, base = synthetic_run
    res = _plain(base)
    r3, r4 = _row(res, "calibration", "row", "R3"), _row(res, "calibration", "row", "R4")
    r3["ece"], r3["ece_ci"] = 0.0652, [0.059, 0.072]
    r4["ece"], r4["ece_ci"] = 0.0590, [0.054, 0.066]
    claims = au.evaluate_claims(res)
    for t in (au.txt_calibration(res), au.txt_conclusion(res, claims)):
        assert "unchanged" not in t.split("Temperature")[0] and "materially" not in t
        assert "detectably" in t
    res["calibration"]["temperature"].update(temperature=1.32, test_argmax_changes=3)
    assert "without changing any top-1 prediction" not in au.txt_calibration(res)
    res["calibration"]["temperature"].update(test_argmax_changes=0)
    assert "without changing any top-1 prediction" in au.txt_calibration(res)
    res["reproduction"]["manifest_meta"]["virtual_ensembles"] = 50
    assert "(M = 50)" in au.txt_calibration(res)


def test_mi_vs_served_prose_uses_paired_difference_not_just_overlap(synthetic_run):
    _, _, base = synthetic_run
    res = _plain(base)
    s1, s5 = _row(res, "selective", "score", "S1"), _row(res, "selective", "score", "S5")
    s1.update(auroc=0.8894, auroc_ci=[0.8825, 0.8967])
    s5.update(auroc=0.8853, auroc_ci=[0.8781, 0.8929])
    _set_d51(res, -0.0041, (-0.0074, -0.0010))            # marginal CIs overlap, paired CI excludes 0
    claims = au.evaluate_claims(res)
    c4 = {c["id"]: c for c in claims}["C4"]
    assert c4["verdict"] == "SUPPORTED"                    # contract rule: overlap -> not "clearly worse"
    assert "paired bootstrap difference" in c4["refinement"]
    sel, imp = au.txt_selective(res), au.txt_implication(res, claims)
    for t in (sel, imp):
        _check_text(t)
        assert "comparabl" not in t
    assert "slightly lower than that of" in sel and "excludes zero" in sel and "−0.0041" in sel
    assert "slightly less well than" in imp and "−0.0041" in imp
    _set_d51(res, -0.0020, (-0.0050, 0.0010))              # paired CI includes 0
    claims = au.evaluate_claims(res)
    sel, imp = au.txt_selective(res), au.txt_implication(res, claims)
    assert "not significantly different from that of" in sel and "includes zero" in sel
    assert "with no significant difference from" in imp
    assert 2 <= count_sentences(imp) <= 3 and 2 <= count_sentences(sel) <= 4


def test_c4_contradicted_below_chance_is_not_called_indistinguishable_from_chance(synthetic_run):
    _, _, base = synthetic_run
    res = _plain(base)
    _row(res, "selective", "score", "S5").update(auroc=0.40, auroc_ci=[0.38, 0.42])
    claims = au.evaluate_claims(res)
    assert {c["id"]: c for c in claims}["C4"]["verdict"] == "CONTRADICTED"
    t = au.txt_implication(res, claims)
    _check_text(t)
    assert "not distinguishable from chance" not in t and "below chance" in t


def test_claim_suggestions_follow_the_direction_of_r4(synthetic_run):
    _, _, base = synthetic_run
    res = _plain(base)
    # R4 significantly positive: C1 contradicted, C4 refinement must not say "does not improve accuracy"
    _set_paired(res, "R4", d=0.004, ci=(0.001, 0.007), p=0.01, p_holm=0.02)
    _row(res, "selective", "score", "S5").update(auroc=0.89, auroc_ci=[0.88, 0.90])
    _row(res, "selective", "score", "S1").update(auroc=0.89, auroc_ci=[0.88, 0.90])
    cl = {c["id"]: c for c in au.evaluate_claims(res)}
    assert cl["C1"]["verdict"] == "CONTRADICTED" and "raises accuracy significantly" in cl["C1"]["suggestion"]
    assert cl["C4"]["verdict"] == "SUPPORTED"
    assert "does not improve" not in cl["C4"]["refinement"] and "raises accuracy" in cl["C4"]["refinement"]
    # R5 significant, R4 positive but n.s.: "cannot be attributed", never "does not raise macro-F1 at"
    _set_paired(res, "R4", d=0.001, ci=(-0.001, 0.003), p=0.4, p_holm=0.4)
    _set_paired(res, "R5", d=0.0063, ci=(0.002, 0.011), p=0.004, p_holm=0.008)
    s2 = {c["id"]: c for c in au.evaluate_claims(res)}["C2"]["suggestion"]
    assert "cannot be attributed to PGS-averaged inference" in s2 and "does not raise macro-F1 at" not in s2
    _set_paired(res, "R4", d=-0.0043, ci=(-0.007, -0.002), p=0.001, p_holm=0.002)
    s2 = {c["id"]: c for c in au.evaluate_claims(res)}["C2"]["suggestion"]
    assert "arises from the change of checkpoint" in s2
    # C3 SUPPORTED refinement must not assert equivalence
    for row in ("R1", "R2", "R3"):
        _set_paired(res, row, d=0.0001, ci=(-0.004, 0.004), p=0.9, p_holm=1.0)
    ref3 = {c["id"]: c for c in au.evaluate_claims(res)}["C3"]["refinement"]
    assert "indistinguishable on the test set" not in ref3 and "not a formal equivalence test" in ref3
