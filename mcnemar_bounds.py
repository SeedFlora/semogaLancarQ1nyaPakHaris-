"""Worst-case McNemar bounds from the aggregate accuracies already reported.

McNemar's test only depends on the discordant counts b (model A right, B wrong)
and c (A wrong, B right).  The reported accuracies fix b - c exactly; b + c is
unknown but bounded.  The test statistic is weakest when b + c is as large as
the data allow, so evaluating the exact test there gives the largest p-value
that is consistent with the published numbers.  If even that p-value is small,
the difference is significant under every possible per-sample configuration.

No per-sample data are needed; every input is a number printed in the
manuscript (n = 9,266 test samples).
"""

from __future__ import annotations

from math import ceil, floor

from scipy.stats import binomtest, chi2

N = 9266


def counts_for(acc4: float, n: int = N) -> list[int]:
    """All integer correct-counts k whose k/n rounds to the 4-decimal value."""
    lo, hi = floor((acc4 - 5e-5) * n), ceil((acc4 + 5e-5) * n)
    return [k for k in range(lo, hi + 1) if round(k / n, 4) == round(acc4, 4)]


def worst_case(k_better: int, k_worse: int, max_discordant: int | None = None, n: int = N):
    d = k_better - k_worse
    assert d > 0
    # c <= errors of the better model; b = c + d <= errors of the worse model
    c_max = min(n - k_better, (n - k_worse) - d)
    bc_max = 2 * c_max + d
    if max_discordant is not None:
        cap = max_discordant if (max_discordant - d) % 2 == 0 else max_discordant - 1
        bc_max = min(bc_max, cap)
    c = (bc_max - d) // 2
    b = c + d
    p_exact = binomtest(c, b + c, 0.5, alternative="two-sided").pvalue
    p_cc = chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1)
    return dict(delta=d, b=b, c=c, discordant=b + c, p_exact=p_exact, p_chi2_cc=p_cc)


def report(name, acc_better, acc_worse, max_discordant=None):
    worst = None
    for kb in counts_for(acc_better):
        for kw in counts_for(acc_worse):
            r = worst_case(kb, kw, max_discordant)
            r.update(k_better=kb, k_worse=kw)
            if worst is None or r["p_exact"] > worst["p_exact"]:
                worst = r
    print(f"{name}\n  counts better={counts_for(acc_better)} worse={counts_for(acc_worse)}")
    print(
        "  worst case: b={b} c={c} (b+c={discordant}, b-c={delta}) "
        "exact p={p_exact:.2e}  chi2-cc p={p_chi2_cc:.2e}".format(**worst)
    )
    return worst


if __name__ == "__main__":
    # Table 3: fusion vs text-only (same head variant)
    report("Fusion vs text-only, CatBoost+PGS (Table 3)", 0.8074, 0.7874)
    report("Fusion vs text-only, CatBoost+PGS, if fused acc = 0.8073", 0.8073, 0.7874)
    report("Fusion vs text-only, CatBoost without PGS (Table 3)", 0.7996, 0.7850)
    report("Fusion vs image-only, CatBoost+PGS (Table 3)", 0.8074, 0.6238)
    # Table 4 rows 3 vs 4 (same PGS checkpoint): 130 top-1 disagreements (Table 5 note)
    report("PGS checkpoint: argmax vs PGS-averaged (<=130 disagreements)", 0.8116, 0.8073, 130)
    report("PGS checkpoint: argmax vs PGS-averaged, if PGS acc = 0.8074", 0.8116, 0.8074, 130)
    # Table 4 rows 1 vs 2 (baseline checkpoint) - no disagreement count reported
    report("Baseline checkpoint: argmax vs PGS-averaged (unbounded)", 0.7996, 0.7914)
    # naive comparison: PGS ckpt PGS-mode vs baseline ckpt argmax
    report("Naive: PGS ckpt+PGS vs baseline ckpt argmax (unbounded)", 0.8073, 0.7996)
