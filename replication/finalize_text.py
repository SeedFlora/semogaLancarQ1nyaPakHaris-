"""Final manuscript wording for the replication run (Section 3.4 prose + result-dependent edits).

Adds what the generic stage-2 text cannot know:
* paired bootstrap differences for calibration (dECE, dBrier, dNLL on identical test samples) —
  the generated prose compared *marginal* intervals, the argument Section 3.3 itself calls invalid
  for paired data;
* paired dAUROC wording for the error detectors (already in results.json);
* the replication framing (retrained head, later snapshot, test n) wherever replication numbers appear;
* tracked edits elsewhere in the manuscript for claims the replication qualifies (claims_check C4),
  for "calibrated via PGS" wording the calibration results do not support, and for the fixes
  accepted by the adversarial review of the filled manuscript.

Every number is taken from results.json / the per-sample CSVs; verbs follow the confidence intervals.
Writes results/fill_values_final.json, results/calibration_paired.json and _replication/extra_edits.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REVISI = HERE.parent
sys.path.insert(0, str(REVISI))
import analyze_uncertainty as au  # noqa: E402

REP = REVISI / "_replication"
RES = REP / "results"
DEP = "dinov3_large__mE5_large"


def paired_calibration(served: au.View, pgs: au.View, t: float, boot: au.Bootstrap) -> dict:
    """Paired bootstrap of B - A for ECE, Brier, NLL (A = served argmax; B = PGS-averaged or TS)."""
    n = served.label.shape[0]

    def per_sample(v_probs, pred):
        conf = v_probs.max(axis=1)
        correct = (pred == served.label).astype(np.float64)
        gap = np.zeros((n, au.N_BINS))
        gap[np.arange(n), au.bin_index(conf)] = correct - conf
        return gap, au.brier_per_sample(v_probs, served.label), au.nll_per_sample(v_probs, served.label)

    ts_probs = au.apply_temperature(served.probs, t)
    arms = {"served": (served.probs, served.pred), "pgs": (pgs.probs, pgs.pred),
            "ts": (ts_probs, ts_probs.argmax(axis=1))}
    stats = {k: per_sample(*v) for k, v in arms.items()}
    point = {k: {"ece": float(np.abs(g.sum(0)).sum() / n), "brier": float(b.mean()), "nll": float(l.mean())}
             for k, (g, b, l) in stats.items()}
    boot_vals = {k: {"ece": np.abs(boot.sums(g)).sum(1) / n, "brier": boot.sums(b) / n, "nll": boot.sums(l) / n}
                 for k, (g, b, l) in stats.items()}
    out = {"point": point}
    for other in ("pgs", "ts"):
        out[f"{other}_minus_served"] = {
            m: {"delta": point[other][m] - point["served"][m],
                "ci": list(boot.interval(boot_vals[other][m] - boot_vals["served"][m]))}
            for m in ("ece", "brier", "nll")}
    return out


def direction(ci: list[float]) -> str:
    return "higher" if ci[0] > 0 else "lower" if ci[1] < 0 else "unchanged"


def sign4(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.4f}"


def main() -> None:
    results = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    values = json.loads((RES / "fill_values.json").read_text(encoding="utf-8"))
    summary = json.loads((REP / "replication_summary.json").read_text(encoding="utf-8"))
    _, files = au.load_prediction_dir(REP / "per_sample_predictions")

    def view(ckpt: str, mode: str) -> au.View:
        img, txt = DEP.split("__")
        for k, pf in files.items():
            if tuple(k) in ((DEP, ckpt, mode), (img, txt, ckpt, mode)):
                return au.make_view(pf, "test")
        raise KeyError((DEP, ckpt, mode))

    served, pgs = view("pgs", "argmax"), view("pgs", "pgs")
    assert np.array_equal(served.row_id, pgs.row_id) and np.array_equal(served.label, pgs.label)
    cal = results["calibration"]
    t = float(cal["temperature"]["temperature"])
    assert cal["temperature"]["test_argmax_changes"] == 0
    rows = {r["row"]: r for r in cal["rows"]}
    served_acc = rows["R3"]["accuracy"]
    boot = au.Bootstrap(served.label.shape[0], 2000, 42)
    pc = paired_calibration(served, pgs, t, boot)
    (RES / "calibration_paired.json").write_text(json.dumps({"T": t, **pc}, indent=2), encoding="utf-8")

    f4, fci, fpp = au.fmt_metric, au.fmt_ci, au.fmt_pp
    P, d, dts = pc["point"], pc["pgs_minus_served"], pc["ts_minus_served"]

    # ---------------------------------------------------------------- calibration prose (paired)
    ece_dir, br_dir, nll_dir = (direction(d[m]["ci"]) for m in ("ece", "brier", "nll"))
    if ece_dir == "higher":
        verb = "worsens calibration"
    elif ece_dir == "lower":
        verb = "improves calibration"
    elif "higher" in (br_dir, nll_dir):
        verb = "does not improve calibration"
    else:
        verb = "does not change calibration detectably"
    ece_clause = (f"a paired difference of {sign4(d['ece']['delta'])} (95% CI {fci(*d['ece']['ci'])})"
                  + ("" if ece_dir != "unchanged" else ", an interval that includes zero"))
    prop = [f"the {name} from {f4(P['served'][m])} to {f4(P['pgs'][m])}{unit} (paired difference "
            f"{sign4(d[m]['delta'])}, 95% CI {fci(*d[m]['ci'])})"
            for m, name, unit in (("brier", "Brier score", ""), ("nll", "NLL", " nats"))]
    if br_dir == nll_dir and br_dir != "unchanged":
        prop_sentence = (f"The two strictly proper scores {'rise' if br_dir == 'higher' else 'fall'} as well: "
                         + "; ".join(prop) + ".")
    else:
        prop_sentence = "For the strictly proper scores, " + "; ".join(prop) + "."
    gap_served = -100 * rows["R3"]["confidence_minus_accuracy"]
    gap_pgs = -100 * rows["R4"]["confidence_minus_accuracy"]
    conf_word = "underconfident" if gap_served > 0 and gap_pgs > 0 else "miscalibrated in opposite directions"
    ts_dir = direction(dts["ece"]["ci"])
    if ts_dir == "lower":
        ts_ece = f"lowers the ECE of the served path to {f4(P['ts']['ece'])}"
    elif ts_dir == "higher":
        ts_ece = f"raises the ECE of the served path to {f4(P['ts']['ece'])}"
    elif dts["ece"]["delta"] < 0:
        ts_ece = f"lowers the ECE of the served path to {f4(P['ts']['ece'])}, but not significantly"
    else:
        ts_ece = f"leaves the ECE of the served path at {f4(P['ts']['ece'])}"
    ts_prop = [m for m in ("brier", "nll") if direction(dts[m]["ci"]) == "lower"]
    names = {"brier": "Brier score", "nll": "NLL"}
    ts_tail = ("; it does lower the " + " and the ".join(
        f"{names[m]} (paired difference {sign4(dts[m]['delta'])}, 95% CI {fci(*dts[m]['ci'])})" for m in ts_prop)
        if ts_prop else "")
    values["TXT_CALIBRATION"] = " ".join((
        f"At the fixed PGS checkpoint, PGS-averaged inference (M = 30) {verb} relative to the served argmax path: "
        f"the ECE is {f4(P['pgs']['ece'])} versus {f4(P['served']['ece'])}, {ece_clause}. {prop_sentence} "
        f"Both inference paths are {conf_word} on average, with mean confidence falling short of accuracy by "
        f"{gap_served:.2f} pp (served argmax) and {gap_pgs:.2f} pp (PGS-averaged). Temperature scaling fitted on "
        f"the validation partition (T = {values['T5_TEMP']}) leaves every top-1 prediction unchanged and {ts_ece} "
        f"(paired difference {sign4(dts['ece']['delta'])}, 95% CI {fci(*dts['ece']['ci'])}){ts_tail}. Paired "
        "differences are computed from unrounded values and may differ in the last digit from differences of the "
        "rounded entries in Table 5."
    ).split())

    # ---------------------------------------------------------------- paired-test prose (accuracy vs macro-F1)
    prow = {r["row"]: r for r in results["paired"]["rows"]}
    enc_names = {"R1": ("DINOv3-L", "EVA-02-L"), "R2": ("DINOv3-L", "DINOv2-L"), "R3": ("EVA-02-L", "DINOv2-L")}
    enc_bits, f1_bits = [], []
    any_holm = any(prow[k]["p_holm"] < 0.05 for k in enc_names)
    for k, (a, b) in enc_names.items():
        r = prow[k]
        extra = ""
        if r["p_mcnemar"] < 0.05 and r["p_holm"] >= 0.05:
            extra = (f", although the unadjusted p = {au.fmt_p(r['p_mcnemar'])} and the unadjusted 95% interval "
                     f"{au.fmt_pp_ci(*r['d_accuracy_ci'])} exclude a zero difference")
        enc_bits.append(f"{a} vs. {b}: Δaccuracy = {fpp(r['d_accuracy'])} pp, adjusted p = "
                        f"{au.fmt_p(r['p_holm'])}{extra}")
        f1_bits.append(fpp(r["d_macro_f1"]))
    f1_all_zero = all(prow[k]["d_macro_f1_ci"][0] <= 0 <= prow[k]["d_macro_f1_ci"][1] for k in enc_names)
    lead = prow["R1"]
    leader, other = ("EVA-02-L", "DINOv3-L") if lead["d_macro_f1"] < 0 else ("DINOv3-L", "EVA-02-L")
    r4, r5 = prow["R4"], prow["R5"]
    assert "pgs__pgs" in r4["source_a"] and "pgs__argmax" in r4["source_b"]
    assert "pgs__pgs" in r5["source_a"] and "cb__argmax" in r5["source_b"]
    r4_sig, r5_sig = r4["p_mcnemar"] < 0.05, r5["p_mcnemar"] < 0.05
    r4_f1_zero = r4["d_macro_f1_ci"][0] <= 0 <= r4["d_macro_f1_ci"][1]
    values["TXT_PAIRED"] = " ".join((
        "Among the image encoders, each fused with mE5-L and evaluated with PGS-averaged inference, "
        + ("none of the three pairwise differences in top-1 correctness is significant after Holm adjustment"
           if not any_holm else "at least one pairwise difference in top-1 correctness is significant after "
                                "Holm adjustment")
        + " (McNemar’s exact test; " + "; ".join(enc_bits) + ")"
        + (", and all three paired-bootstrap intervals for Δmacro-F1 include zero ("
           if f1_all_zero else "; the paired-bootstrap Δmacro-F1 values are ")
        + ", ".join(f1_bits[:-1]) + ", and " + f1_bits[-1] + " pp; Table 6). "
        f"{leader} is thus numerically ahead of {other} in this run, and a non-significant difference is not "
        "evidence of equivalence. "
        f"At the fixed PGS checkpoint, PGS-averaged inference changes accuracy by {fpp(r4['d_accuracy'])} pp "
        f"(95% CI {au.fmt_pp_ci(*r4['d_accuracy_ci'])}) and macro-F1 by {fpp(r4['d_macro_f1'])} pp (95% CI "
        f"{au.fmt_pp_ci(*r4['d_macro_f1_ci'])}) relative to the served argmax path; with {r4['b']} reports correct "
        f"only under PGS averaging and {r4['c']} only under argmax, the accuracy difference is "
        + ("statistically significant" if r4_sig else "not statistically significant")
        + f" (McNemar p = {au.fmt_p(r4['p_mcnemar'])}; Holm-adjusted p = {au.fmt_p(r4['p_holm'])})"
        + (", whereas the macro-F1 interval includes zero. " if r4_f1_zero else ". ")
        + "For the naive comparison between the PGS configuration and the baseline checkpoint with argmax "
        "inference, the replication gives a smaller difference than the original run (+0.63 pp macro-F1 there; "
        f"here {fpp(r5['d_macro_f1'])} pp macro-F1, 95% CI {au.fmt_pp_ci(*r5['d_macro_f1_ci'])}, and "
        f"{fpp(r5['d_accuracy'])} pp accuracy), which is "
        + ("statistically significant" if r5_sig else "not statistically significant")
        + f" in top-1 correctness (McNemar p = {au.fmt_p(r5['p_mcnemar'])}, Holm-adjusted p = "
        f"{au.fmt_p(r5['p_holm'])}); the replication checkpoints also differ less in tree count than the original "
        "ones (Table 5 note)."
    ).split())
    values["T1_ENC"] = (f"Δmacro-F1 = {fpp(lead['d_macro_f1'])} pp {au.fmt_pp_ci(*lead['d_macro_f1_ci'])}; McNemar "
                        f"(top-1 correctness) Holm-adjusted p = {au.fmt_p(lead['p_holm'])}")

    # ---------------------------------------------------------------- selective prose (paired dAUROC)
    diffs = {(x["a"], x["b"]): (x["d_auroc"], x["d_auroc_ci"])
             for x in results["selective"]["paired_auroc_differences"]}
    s21, s21ci = diffs[("S2", "S1")]
    s51, s51ci = diffs[("S5", "S1")]
    a1, a2, a5 = values["T7_R1_AUROC"], values["T7_R2_AUROC"], values["T7_R5_AUROC"]
    top = "slightly but significantly above" if s21ci[1] < 0 else "not significantly different from"
    values["TXT_SELECTIVE"] = " ".join((
        f"The maximum class probability of the served argmax path is the best error detector (AUROC {a1}), {top} "
        f"the same score computed from the PGS-averaged probabilities (AUROC {a2}; paired difference {sign4(s21)}, "
        f"95% CI {fci(*s21ci)}), whereas the epistemic mutual information is a much weaker detector (AUROC {a5}; "
        f"paired difference to the served score {sign4(s51)}, 95% CI {fci(*s51ci)}); the aleatoric and total "
        f"entropy terms lie in between (AUROC {values['T7_R4_AUROC'].split(' ')[0]} and "
        f"{values['T7_R3_AUROC'].split(' ')[0]}). Deferring the 20% least certain reports according to the served "
        f"maximum probability raises the accuracy of the accepted 80% to {values['T7_R1_ACC80']} (from "
        f"{f4(served_acc)} at full coverage), and automatic routing can cover up to "
        f"{float(values['T7_R1_COV90']) * 100:.1f}% of the reports while the accepted ones keep at least 90% accuracy."
    ).split())

    # ---------------------------------------------------------------- implication + conclusion
    cal_short = {"worsens calibration": "worsens calibration", "improves calibration": "improves calibration",
                 "does not improve calibration": "does not improve calibration",
                 "does not change calibration detectably": "leaves calibration unchanged"}[verb]
    assert s51ci[1] < 0, "wording below assumes the epistemic score is the weaker detector"
    values["TXT_IMPLICATION"] = " ".join((
        "Taken together, the replication qualifies the suggestion of Section 3.3 that the value of PGS, if any, "
        "lies in its uncertainty signal rather than in accuracy: at a fixed checkpoint, PGS averaging lowers accuracy "
        f"({fpp(r4['d_accuracy'])} pp; McNemar p = {au.fmt_p(r4['p_mcnemar'])}) and {cal_short}, and although its "
        f"mutual information carries an error signal (AUROC {a5.split(' ')[0]}), the maximum class probability of the "
        "served argmax path — which needs no PGS and survives the ONNX export — flags misrouted reports far better "
        f"(AUROC {a1.split(' ')[0]}). Operationally, a deferral rule on this score is the practical uncertainty "
        "mechanism for the prototype: in the replication, sending the 20% least certain reports to a human operator "
        f"would raise the accuracy of the automatically routed 80% from {f4(served_acc)} to {values['T7_R1_ACC80']}, "
        "and the deferral threshold for the deployed head should be set on its own validation data."
    ).split())
    d_sum = summary["data"]
    values["TXT_CONCLUSION"] = " ".join((
        f"A replication of the pipeline on a later snapshot of the same portal ({d_sum['clean_rows']:,} pairs; test "
        f"n = {d_sum['split_sizes']['test']:,}) confirmed that PGS averaging lowers accuracy at a fixed checkpoint and "
        f"showed that it {verb} (ECE {f4(P['pgs']['ece'])} versus {f4(P['served']['ece'])} under ordinary argmax "
        "inference, the mode served in production). In that replication, the maximum class probability of the argmax "
        f"path flagged misrouted reports better than the PGS epistemic signal (AUROC {a1.split(' ')[0]} versus "
        f"{a5.split(' ')[0]}), and deferring the 20% least certain reports to operators would raise the accuracy of "
        f"the 80% routed automatically from {f4(served_acc)} to {values['T7_R1_ACC80']}."
    ).split())
    (RES / "fill_values_final.json").write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- result-dependent manuscript edits
    served_better_cal = ece_dir == "higher" or ("higher" in (br_dir, nll_dir) and ece_dir != "lower")
    cal_phrase = ("slightly worsens calibration" if ece_dir == "higher" else
                  "does not improve calibration" if served_better_cal else "leaves calibration essentially unchanged")
    naive = (f"in the replication, its counterpart ({fpp(r5['d_macro_f1'])} pp macro-F1, {fpp(r5['d_accuracy'])} pp "
             f"accuracy) was {'' if r5_sig else 'not '}statistically significant (McNemar p = "
             f"{au.fmt_p(r5['p_mcnemar'])})")
    n_te = f"{d_sum['split_sizes']['test']:,}"
    edits = [
        # abstract
        {"needle": "Classification is subsequently performed using a CatBoost gradient-boosting head",
         "old": "head calibrated via Posterior Gaussian Sampling (PGS).",
         "new": "head with Posterior Gaussian Sampling (PGS) virtual ensembles."},
        {"needle": "Classification is subsequently performed using a CatBoost gradient-boosting head",
         "after": "by utilizing Haversine distance calculations.",
         "insert": " Paired significance tests and a replication of the pipeline on a later snapshot of the same "
                   f"portal, with per-sample predictions, show that PGS averaging lowers accuracy at a fixed checkpoint "
                   f"and {cal_phrase}, whereas the plain maximum class probability of ordinary argmax inference, the "
                   "mode served in production, is the stronger signal for deferring uncertain reports to human "
                   "operators."},
        # Section 1
        {"needle": "This study extends the Hanif et al. (2026) lineage",
         "old": "subjecting the PGS calibration mechanism to a confound-controlled",
         "new": "subjecting the PGS mechanism to a confound-controlled"},
        {"needle": "This study addresses that gap by combining a frozen DINOv3-Large visual encoder",
         "old": "head calibrated with Posterior Gaussian Sampling", "new": "head that uses Posterior Gaussian Sampling"},
        # Section 2.3
        {"needle": "selected as the sole classification algorithm on eight grounds",
         "old": "native probability calibration via Posterior Gaussian Sampling",
         "new": "native posterior sampling via Posterior Gaussian Sampling"},
        {"needle": "selected as the sole classification algorithm on eight grounds",
         "old": "which simultaneously yields an uncertainty estimate usable in the client interface",
         "new": "which yields an uncertainty estimate intended for the client interface (its loss at ONNX export is "
                "reported in Section 3.5 and its calibration and error-detection value in Section 3.4)"},
        {"needle": "Key hyperparameters were maximum 1,500 iterations",
         "old": "50-round early stopping on the validation macro-F1",
         "new": "early stopping on the validation multiclass log-loss (patience of 50 rounds for the baseline and 200 "
                "rounds for the PGS variant)"},
        # Section 2.4
        {"needle": "a PGS on/off ablation evaluated the effect of Posterior Gaussian Sampling",
         "old": "comparing accuracy, macro-F1, and mean epistemic uncertainty with and without PGS calibration.",
         "new": "comparing accuracy and macro-F1 with and without PGS, together with the mean across-member dispersion "
                "of the virtual-ensemble probabilities (Section 2.3)."},
        {"needle": "The baseline and PGS checkpoints were originally trained as two separate runs",
         "old": "conflates the calibration mechanism with a near doubling",
         "new": "conflates the PGS mechanism with a near doubling"},
        # Section 3 intro
        {"needle": "Unless stated otherwise, all classification metrics are computed once on the held-out test partition",
         "old": "which was never used for training, early stopping, or encoder selection.",
         "new": "which was never used for training, early stopping, or encoder selection; the only exception is "
                f"Section 3.4, which analyses a replication run on a later data snapshot (test n = {n_te}; "
                "Section 2.6)."},
        # Section 3.3
        {"needle": "A naive comparison of the two trained checkpoints suggests",
         "old": "conflates the calibration mechanism with a near-doubling",
         "new": "conflates the PGS mechanism with a near-doubling"},
        {"needle": "The bootstrap analysis points the same way",
         "old": "so even the naive +0.63 pp difference cannot be distinguished from sampling variance.",
         "new": "which suggests, but cannot establish, that even the naive +0.63 pp difference lies within sampling "
                "variance."},
        {"needle": "The documented value of PGS in this system",
         "old": "PGS-calibrated macro-F1", "new": "PGS-averaged macro-F1"},
        {"needle": "The documented value of PGS in this system",
         "old": "is therefore an uncertainty signal at negligible inference cost (about 0.05 ms per sample for the "
                "head with 30 virtual members), not an accuracy gain;",
         "new": "therefore lies, if anywhere, in an uncertainty signal at negligible inference cost (about 0.05 ms per "
                "sample for the head with 30 virtual members) rather than in an accuracy gain, and the replication in "
                "Section 3.4 shows that even this signal is weaker than the maximum class probability of the argmax "
                "(served) inference path;"},
        # Section 3.5
        {"needle": "is not available in the running prototype",
         "after": "is not available in the running prototype.",
         "insert": " The replication in Section 3.4 suggests that this loss is less consequential than it appears: under "
                   "ordinary argmax inference, the mode that is served, the maximum class probability is "
                   + ("better calibrated than the PGS-averaged probabilities and " if served_better_cal else "")
                   + "a markedly stronger error detector than the PGS epistemic signal."},
        # Section 3.9 practical lessons
        {"needle": "Three practical lessons emerge", "old": "Three practical lessons", "new": "Four practical lessons"},
        {"needle": "Three practical lessons emerge", "old": "; and (iii) routing quality", "new": "; (iii) routing quality"},
        {"needle": "Three practical lessons emerge", "after": "office registry as on the classifier",
         "insert": "; and (iv) the plain maximum class probability of the argmax (served) inference path is the baseline "
                   "against which any ensemble-based uncertainty signal should be judged, because in the replication run "
                   "(Section 3.4) it was a stronger deferral signal than any PGS-based uncertainty score"},
        # Table 13 (uncertainty row)
        {"needle": "Shows the naive +0.63 pp gain is not significant",
         "old": "Shows the naive +0.63 pp gain is not significant",
         "new": "Shows the naive +0.63 pp gain is confounded with tree count (its paired replication counterpart is "
                + ("" if r5_sig else "not ") + "significant, Table 6)"},
        {"needle": "Shows the naive +0.63 pp gain is not significant",
         "after": "(≈0.05 ms per sample) (Table 4)",
         "insert": "; in a replication its epistemic signal is a weaker error detector than the maximum class "
                   "probability (Table 7)"},
        # Conclusion
        {"needle": "The value of PGS in this setting lies",
         "old": "and the naive +0.63 pp difference is not distinguishable from sampling variance.",
         "new": "and the naive +0.63 pp difference, which is confounded with the tree count, could not be tested on "
                f"paired samples in the original run; {naive}."},
        {"needle": "The value of PGS in this setting lies",
         "old": "The value of PGS in this setting lies in providing a low-cost uncertainty signal, not in improving "
                "accuracy,",
         "new": "PGS does not improve accuracy in this setting,"},
        {"needle": "The value of PGS in this setting lies",
         "old": "and lacks the uncertainty signal;", "new": "and lacks the PGS uncertainty terms;"},
        {"needle": "Future work should complete encoder-level parity verification",
         "old": "restore the uncertainty signal in the serving path",
         "new": "restore the PGS uncertainty terms in the serving path if they prove informative beyond the maximum "
                "class probability (Section 3.4)"},
        # Section 5.1
        {"needle": "We extend this research across four aspects",
         "old": "scaling up the dataset to 61,773 image-text pairs covering nine agencies; and re-evaluating the "
                "previously employed PGS mechanism by implementing confound control.",
         "new": "scaling up the dataset to 61,773 image-text pairs covering nine agencies; re-evaluating the previously "
                "employed PGS mechanism by implementing confound control; and carrying the selected model through ONNX "
                "export validation, a production FastAPI/Flutter deployment, and real-user usability testing."},
        {"needle": "The true value of PGS lies",
         "old": "The true value of PGS lies in its function as a signal of epistemic uncertainty, rather than as a "
                "source of improved accuracy.",
         "new": "PGS is therefore not a source of improved accuracy, and a replication with per-sample predictions "
                "(Section 3.4) shows that its epistemic-uncertainty signal is also weaker than the plain maximum class "
                "probability for flagging misrouted reports."},
        {"needle": "Methodological contribution: a confound-controlled re-evaluation",
         "old": "a confound-controlled re-evaluation of an uncertainty-calibration method.",
         "new": "a confound-controlled re-evaluation of an ensemble-based uncertainty method."},
        {"needle": "Methodological contribution: a confound-controlled re-evaluation",
         "old": "that explains why the residual gain is structurally small)",
         "new": "that explains why averaging the virtual-ensemble members yields no accuracy gain here)"},
        # Limitations (ii)
        {"needle": "Consistent with the underlying dissertation, the manuscript should retain",
         "old": "the PGS ablation's headline comparison conflates calibration with iteration count",
         "new": "the PGS ablation's headline comparison conflates PGS inference with iteration count"},
        # Section 6.2 author notes
        {"needle": "Pairwise statistical significance testing among the top encoder pairs", "after": None,
         "insert": " [Addressed in Section 3.4 (replication run).]"},
        {"needle": "A reliability diagram and Expected Calibration Error (ECE)", "after": None,
         "insert": " [Addressed in Section 3.4 (replication run).]"},
        {"needle": "A correlation analysis between PGS epistemic uncertainty", "after": None,
         "insert": " [Addressed in Section 3.4 (replication run); the result does not support this argument: the "
                   f"epistemic signal separates errors only weakly (AUROC {a5.split(' ')[0]}) and is outperformed by the "
                   f"maximum class probability (AUROC {a1.split(' ')[0]}).]"},
        {"needle": "A sensitivity analysis of a selective-prediction (reject-option) threshold", "after": None,
         "insert": " [Addressed in Section 3.4 (replication run).]"},
    ]
    (REP / "extra_edits.json").write_text(json.dumps(edits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"calibration_paired": pc, "verb": verb, "n_edits": len(edits),
                      **{k: values[k] for k in ("TXT_CALIBRATION", "TXT_PAIRED", "TXT_SELECTIVE", "TXT_IMPLICATION",
                                                 "TXT_CONCLUSION", "T1_ENC")}},
                     indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
