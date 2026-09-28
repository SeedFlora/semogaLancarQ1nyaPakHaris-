# Interface contract — per-sample uncertainty analysis for SmartCitty_IJOST_Rev_1

This folder closes the gaps listed in `kurangan dari DINO-EVA2.docx`:
ECE + reliability diagrams (with/without PGS), McNemar + paired bootstrap between
encoders, and selective prediction (accuracy–coverage, AUROC of uncertainty vs. error).
All of these need **per-sample** records, which only exist where the embedding cache and
CatBoost checkpoints live (the RunPod `/workspace` of the thesis, NOT this laptop).

Pipeline (three scripts, three stages, strict file contracts between them):

```
[RunPod]  export_per_sample_predictions.py  --artifacts smartCityReport/artifacts --out per_sample_predictions/
[local ]  analyze_uncertainty.py            --preds per_sample_predictions/ --out results/
[local ]  fill_manuscript.py                --docx SmartCitty_IJOST_Rev_1.docx --results results/ --out SmartCitty_IJOST_Rev_1_filled.docx
```

Environment for development/tests on this machine: Windows 11, `python` = Python 3.13 with
numpy, pandas, scipy 1.17, scikit-learn 1.9, statsmodels, matplotlib 3.11, catboost 1.2.10,
python-docx 1.2, lxml. No LibreOffice; MS Word is installed (COM automation possible).
Scripts must also run on Linux (RunPod) — use `pathlib`, no Windows-only APIs.

---------------------------------------------------------------------------------------------
## 1. Source layout on RunPod (from notebooks 06/07/08 in smartCityReport/notebooks)

```
artifacts/
  embeddings/image/{dinov3_large,eva02_large,dinov2_large,hiera_large}.npy   # (N_all, 1024) float32, row i = metadata_clean row i
  embeddings/text/{mE5_large,bge_m3,indobert,cendol_mt5,...}.npy              # (N_all, D)
  splits/{train,val,test}.csv        # columns include row_id (int index into the .npy) and label_id (0..8)
  models/checkpoints/{img}__{txt}__cb.cbm    # baseline CatBoost, posterior_sampling=False, 1500 its, ES 50
  models/checkpoints/{img}__{txt}__pgs.cbm   # posterior_sampling=True, 3000 its, ES 200 (≈2,989 trees for dinov3+mE5)
```

Feature construction must be byte-identical to the notebooks:
`idx = split['row_id'].values`; image part `img_emb[idx]`, text part `txt_emb[idx]`; each
L2-normalised row-wise with `x / clip(norm, 1e-9, None)` (float64 math on the float32
inputs is fine), then `np.concatenate([img, txt], axis=1).astype(np.float32)`.
Class order = `crm.TARGET_CLASSES` (index 0..8):
0 Dinas Bina Marga, 1 Satuan Polisi Pamong Praja, 2 Dinas Perhubungan, 3 Kelurahan,
4 Dinas Pertamanan dan Hutan, 5 Dinas Sumber Daya Air,
6 Dinas Cipta Karya, Tata Ruang, dan Pertanahan, 7 Badan Pembinaan Badan Usaha Milik Daerah,
8 Instansi lain.

Inference modes:
* `argmax` — `model.predict_proba(X)` (full model; for the PGS checkpoint this is the path
  served by ONNX).
* `pgs` — `model.virtual_ensembles_predict(X, prediction_type='VirtEnsembles',
  virtual_ensembles_count=M)` with M = 30 → raw logits (N, M, 9). Per-member softmax, then
  **linear pooling** `p = mean_m softmax(z_m)` (this is what notebooks 07/08 did and what
  produced the headline numbers). Also compute **log-linear pooling** `q = softmax(mean_m z_m)`
  for audit only. Works for both `cb` and `pgs` checkpoints (Table 4 row 2 applies PGS
  inference to the baseline checkpoint) — do NOT call `crm.pgs.validate_pgs_model` on `cb`.
  Uncertainty terms (nats), from the per-member probabilities P (N, M, 9):
  `pred_entropy = H(mean_m P)`, `exp_entropy = mean_m H(P_m)`, `mi = max(pred_entropy - exp_entropy, 0)`,
  `prob_std = mean_k std_m(P[:, m, k])` (numpy default ddof=0; this is the legacy score whose
  test mean was reported as 0.00339). `H(p) = -sum p log clip(p, 1e-12, 1)`.

---------------------------------------------------------------------------------------------
## 2. Export output (stage 1 → stage 2)

Directory `per_sample_predictions/` containing:

* One gzip CSV per (pair, checkpoint, mode):
  `preds__{img}__{txt}__{ckpt}__{mode}.csv.gz`, ckpt ∈ {cb, pgs}, mode ∈ {argmax, pgs}.
  Rows: every validation and every test sample (val first, then test, in split-file order).
  Columns, in this order:
  - `split` ∈ {val, test}
  - `row_id` (int), `label` (int 0..8), `pred` (int, argmax of p0..p8, ties → lowest index)
  - `p0` … `p8` — float probabilities, written with `float_format='%.10g'`, rows sum to 1 ± 1e-6
  - only when mode == pgs, additionally: `mi`, `pred_entropy`, `exp_entropy`, `prob_std`,
    `q0` … `q8`, `pred_loglin`
* `manifest.json`:
  ```json
  {
    "created_utc": "...", "catboost_version": "...", "numpy_version": "...",
    "virtual_ensembles": 30, "pooling": "linear (mean of per-member softmax)",
    "n_val": 9266, "n_test": 9266,
    "files": [
      {"file": "preds__dinov3_large__mE5_large__pgs__pgs.csv.gz", "image": "dinov3_large",
       "text": "mE5_large", "checkpoint": "pgs", "mode": "pgs",
       "cbm_path": "...", "cbm_sha256": "...", "tree_count": 2989,
       "metrics": {"val": {"accuracy": 0.0, "macro_f1": 0.0}, "test": {"accuracy": 0.0, "macro_f1": 0.0,
                    "accuracy_loglin": 0.0, "macro_f1_loglin": 0.0, "mean_prob_std": 0.0, "mean_mi": 0.0}},
       "expected_test": {"accuracy": [0.8073, 0.8074], "macro_f1": [0.7747]},
       "reproduces_manuscript": true}
    ]
  }
  ```
  `expected_test` holds the 4-decimal values printed in the manuscript (a list = any of them
  is acceptable); `reproduces_manuscript` is true iff every exported metric rounds (4 dp) to one
  of the listed values; null when no expectation exists. Expected values (test set):
  | pair | ckpt | mode | accuracy | macro-F1 |
  |---|---|---|---|---|
  | dinov3_large__mE5_large | cb  | argmax | 0.7996 | 0.7684 |
  | dinov3_large__mE5_large | cb  | pgs    | 0.7914 | 0.7600 |
  | dinov3_large__mE5_large | pgs | argmax | 0.8116 | 0.7793 |
  | dinov3_large__mE5_large | pgs | pgs    | 0.8073 or 0.8074 | 0.7747 |
  | eva02_large__mE5_large  | pgs | pgs    | —      | 0.7747 |
  | dinov2_large__mE5_large | pgs | pgs    | —      | 0.7736 |
  Additionally for dinov3+mE5 pgs/pgs: mean test `prob_std` expected ≈ 0.00339 (report, don't fail).
  Also: number of test samples whose top-1 differs between dinov3 `pgs/argmax` and `pgs/pgs`
  is expected to be 130 — record it as `"top1_disagreements_pgs_vs_argmax": 130` at the top level.

Default pairs exported: the three `{dinov3_large, eva02_large, dinov2_large} × mE5_large`, all
four (ckpt, mode) combinations each. `--all-pairs` exports every pair found in `checkpoints/`.

---------------------------------------------------------------------------------------------
## 3. Analysis output (stage 2 → stage 3)

Directory `results/` containing:
* `results.json` — every computed number (machine-readable, nested, self-describing).
* `fill_values.json` — flat object `{KEY: "formatted string"}` covering **every key listed in §4**.
* `fig_reliability.png` — **exactly 7.0 in × 2.6 in at 300 dpi = 2100 × 780 px** (do not use
  bbox_inches='tight'; use fixed figsize + constrained layout / subplots_adjust).
* `fig_selective.png` — **exactly 7.0 in × 2.8 in at 300 dpi = 2100 × 840 px**.
* `tables.md` (human-readable Tables 5–7 + Table 1 additions) and `tables/*.csv`.
* `claims_check.md` — list of manuscript claims with verdict SUPPORTED / CONTRADICTED /
  QUALIFIED and the exact manuscript sentence(s) to revise if not supported (see §5).

Statistical definitions (fixed; also stated in the manuscript Section 2.6):
* Deployed pair = `dinov3_large__mE5_large` (CLI `--deployed`).
* ECE: top-label, 15 equal-width bins on (0, 1], bin i = (i/15, (i+1)/15], confidence =
  max p; ECE = Σ_b (n_b/N)|acc_b − conf_b|. 95% CI: percentile bootstrap, 2,000 resamples of
  test indices, `numpy.random.default_rng(42)`.
* Brier score: mean over samples of Σ_k (p_k − 1[y=k])² (range 0–2).
* NLL: mean −log clip(p_y, 1e-12, 1) (nats).
* Temperature scaling: T minimises validation NLL of softmax(log clip(p,1e-12,1)/T) for the
  `pgs/argmax` (served) probabilities; bounded scalar search T ∈ [0.05, 20]; apply to test.
* Paired tests between A and B on the same test samples: b = #(A correct, B wrong),
  c = #(A wrong, B correct); McNemar exact two-sided p = `scipy.stats.binomtest(min(b,c), b+c, 0.5).pvalue`
  (p = 1 if b + c = 0). Paired bootstrap: 2,000 resamples of shared indices (rng seed 42), Δ =
  metric(A) − metric(B) for accuracy and macro-F1 (sklearn `f1_score(average='macro',
  labels=range(9), zero_division=0)`); 95% percentile CI. Holm adjustment within each family.
  Families: encoders (A/B among dinov3_large, eva02_large, dinov2_large; each `pgs/pgs`):
  R1 dinov3 vs eva02, R2 dinov3 vs dinov2, R3 eva02 vs dinov2. Inference modes (deployed pair):
  R4 = `pgs/pgs` (A) vs `pgs/argmax` (B); R5 = `pgs/pgs` (A) vs `cb/argmax` (B) (the naive
  comparison).
* Selective prediction (deployed pair, PGS checkpoint). Scores (higher = less certain):
  S1 `1 − max p` of `pgs/argmax` (correctness w.r.t. `pgs/argmax` preds);
  S2 `1 − max p` of `pgs/pgs`; S3 `pred_entropy`; S4 `exp_entropy`; S5 `mi`; S6 `prob_std`
  (S2–S6 correctness w.r.t. `pgs/pgs` preds). Sort ascending by score (stable, ties by
  row order) = most certain first. Coverage k/N for k = 1..N.
  - AUROC for error detection: `roc_auc_score(incorrect, score)`; 95% bootstrap CI.
  - AURC = mean over k = 1..N of (errors in first k)/k; report ×100.
  - Accuracy at 80% coverage: accuracy of the first round(0.8 N) samples.
  - Coverage at 90% accuracy: max k/N such that accuracy of first k ≥ 0.90 (0 if none).

---------------------------------------------------------------------------------------------
## 4. Placeholder keys in the manuscript (stage 3)

Tokens appear in the DOCX as `⟦KEY⟧` or `⟦KEY: human description⟧` (regex
`⟦([A-Z0-9_]+)(?::[^⟧]*)?⟧`), always entirely inside one `<w:t>`, in yellow-highlighted runs.
Formatting conventions (match the manuscript): metrics 4 dp ("0.8116"); pp 2 dp with sign and
Unicode minus ("−0.43"); CI "[−0.12, 0.35]"; p-values: 3 significant digits ("0.0421",
"0.237"), "< 0.001" below 0.001; Unicode minus "−" (U+2212) everywhere, never "-" for negatives.

Table 5 (calibration; deployed pair, test set). Rows R1 `cb/argmax`, R2 `cb/pgs`,
R3 `pgs/argmax`, R4 `pgs/pgs`, R5 `pgs/argmax` + temperature scaling:
`T5_R{1..5}_ACC`, `T5_R{1..5}_CONF` (mean confidence), `T5_R{1..5}_ECE` ("0.0123 [0.0101, 0.0150]"),
`T5_R{1..5}_BRIER`, `T5_R{1..5}_NLL`, and `T5_TEMP` (e.g. "1.23").

Table 6 (paired tests). Rows R1–R5 as in §3:
`T6_R{1..5}_DACC` ("+0.12 [−0.30, 0.55]"), `T6_R{1..5}_DF1`, `T6_R{1..5}_BC` ("84 / 45"),
`T6_R{1..5}_P`, `T6_R{1..5}_PHOLM`.

Table 7 (selective prediction). Rows R1–R6 = S1–S6:
`T7_R{1..6}_AUROC` ("0.8123 [0.8012, 0.8230]"), `T7_R{1..6}_AURC` ("6.12"),
`T7_R{1..6}_ACC80` ("0.9012"), `T7_R{1..6}_COV90` ("0.781").

Table 1 (overview) additions: `T1_ECE` (e.g. "0.0123 vs. 0.0150"), `T1_ENC`
(e.g. "Δmacro-F1 = +0.01 pp; p = 0.912"), `T1_AUROC` (e.g. "0.8123 vs. 0.7512").

Generated prose (full English sentences, academic register, no Markdown, may contain several
sentences; must be numerically consistent with the tables and never overclaim; each must
handle every outcome direction):
* `TXT_REPRO` — one sentence fragment continuing "The exported records reproduce the
  accuracy and macro-F1 values of Table 4 " e.g. "to the fourth decimal place for all four
  checkpoint–mode combinations." or an honest description of any deviation.
* `TXT_CALIBRATION` — 2–4 sentences: ECE/Brier/NLL of served path vs PGS-averaged at the
  fixed PGS checkpoint (whether PGS improves, worsens or leaves calibration unchanged, using
  whether the bootstrap CIs overlap), over-/under-confidence direction (mean confidence vs
  accuracy), effect of temperature scaling (T value, ECE after).
* `TXT_PAIRED` — 2–4 sentences: encoder comparisons (significance after Holm; note that a
  non-significant difference is not proof of equivalence), then R4 and R5 (explicitly say
  whether the naive +0.63 pp macro-F1 difference is significant under the paired test).
* `TXT_SELECTIVE` — 2–4 sentences: best error-detection score by AUROC, how MI compares with
  1 − max p (CIs), accuracy at 80% coverage and coverage at 90% accuracy for the best score.
* `TXT_IMPLICATION` — 2–3 sentences: what this means for the claim "the value of PGS lies in
  its uncertainty signal, not in accuracy" (supported / qualified / not supported) and the
  operational deferral recommendation (e.g., defer the 20% least certain reports).
* `TXT_CONCLUSION` — 1–2 sentences for the Conclusion summarising calibration and
  error-detection findings.

Figure placeholders: inline images whose `<wp:docPr name=...>` is
`PLACEHOLDER_FIG5_RELIABILITY` (→ `fig_reliability.png`) and `PLACEHOLDER_FIG6_SELECTIVE`
(→ `fig_selective.png`). The placeholder PNGs have the same pixel size as the final figures.

---------------------------------------------------------------------------------------------
## 5. Claims to check (`claims_check.md`)

C1 PGS averaging lowers accuracy at fixed trees (Sections 3.3, 4) — needs R4 Δacc < 0 and p < 0.05.
C2 "the naive +0.63 pp difference cannot be distinguished from sampling variance"
   (Sections 3.3, 4, 5.1) — CONTRADICTED if R5 McNemar p < 0.05 or the ΔF1 CI excludes 0.
C3 "DINOv3-L and EVA-02-L are indistinguishable on accuracy" (Section 3.2, 5.1) — SUPPORTED if R1
   Holm p ≥ 0.05 and ΔF1 CI includes 0 (add the caveat: not an equivalence test).
C4 "the value of PGS lies in its uncertainty signal" (Sections 3.3, 4, 5.1) — SUPPORTED if S5 (MI)
   AUROC CI lies above 0.5 AND S5 is not clearly worse than S1; QUALIFIED if S5 > 0.5 but its
   AUROC is lower than S1 with non-overlapping CIs (the plain max probability of the served
   path already carries the signal); CONTRADICTED if the S5 AUROC CI includes 0.5.
C5 Export reproduces the manuscript numbers (manifest `reproduces_manuscript` for all rows).
C6 Top-1 disagreements between `pgs/argmax` and `pgs/pgs` = 130.
