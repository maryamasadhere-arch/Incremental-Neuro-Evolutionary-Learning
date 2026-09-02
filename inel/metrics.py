"""
Evaluation metrics (report Sec. 2.5, Objectives O4/O6): Retention Accuracy
(RA), Forgetting Rate (FR), Forward Transfer (FT), Evolvability Ceiling (EC).

`runs` is a list of per-run accuracy matrices, one row per training episode
`ep` (0-indexed), each row holding the test accuracy on every task trained
so far, i.e. `runs[r][ep][ti]` = accuracy on task `ti` after the episode-`ep`
training round, for `ti <= ep`. This is exactly what
`models.{backprop,ea,neat}.run_*` produce.
"""

from __future__ import annotations


def compute_metrics(runs: list[list[list[float]]], name: str, N: int = 5,
                     ec_prior_threshold: float = 0.70,
                     ec_current_threshold: float = 0.85) -> dict:
    n = len(runs)

    # Retention Accuracy: per-task accuracy after the full task sequence.
    ra = []
    for ti in range(N):
        v = [runs[r][-1][ti] for r in range(n) if ti < len(runs[r][-1])]
        ra.append(round(sum(v) / len(v) * 100, 1))
    ra_mean = round(sum(ra) / N, 1)

    # Forgetting Rate: peak accuracy on task i minus its accuracy once the
    # full sequence has been trained. Undefined for the final task (it is
    # never "forgotten" - nothing is trained after it), so range(N-1).
    fr = []
    for ti in range(N - 1):
        v = []
        for r in range(n):
            mx = max(runs[r][ep][ti] for ep in range(ti, N)
                      if ep < len(runs[r]) and ti < len(runs[r][ep]))
            fn = runs[r][-1][ti] if ti < len(runs[r][-1]) else 0
            v.append(max(0, mx - fn))
        fr.append(round(sum(v) / n * 100, 1))
    fr_mean = round(sum(fr) / len(fr), 1) if fr else 0.0

    # Forward Transfer: accuracy on task i the moment it is first trained,
    # relative to chance (50% for binary classification). Task 0 is
    # excluded - forward transfer measures the benefit of *prior* task
    # knowledge, which task 0 by definition has none of.
    ft = []
    for ti in range(1, N):
        v = [runs[r][ti][ti] * 100 - 50 for r in range(n)
             if ti < len(runs[r]) and ti < len(runs[r][ti])]
        if v:
            ft.append(round(sum(v) / len(v), 1))
    ft_mean = round(sum(ft) / len(ft), 1) if ft else 0.0

    # Evolvability Ceiling (report Sec. 2.5): the longest *contiguous*
    # prefix of the task sequence for which every prior task retains RA
    # above `ec_prior_threshold` AND the just-trained task exceeds
    # `ec_current_threshold`. This uses two distinct thresholds (not one),
    # and stops counting at the first episode that fails the criterion -
    # a "ceiling" that could be regained after a mid-sequence collapse
    # would not describe a stable ceiling at all.
    ec = []
    for r in range(n):
        s = 0
        for ep in range(N):
            accs = runs[r][ep]
            prior_ok = all(a >= ec_prior_threshold for a in accs[:-1])
            current_ok = accs[-1] >= ec_current_threshold
            if prior_ok and current_ok:
                s = ep + 1
            else:
                break
        ec.append(s)
    ec_mean = round(sum(ec) / n, 1)

    return {
        "name": name,
        "RA": ra, "RA_mean": ra_mean,
        "FR": fr, "FR_mean": fr_mean,
        "FT": ft, "FT_mean": ft_mean,
        "EC": ec_mean, "EC_runs": ec,
    }


def print_summary(metrics_list: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("COMPARATIVE RESULTS  -  Objective O6")
    print("=" * 65)
    print(f"{'Metric':<32}{'Baseline':>11}{'2007 EA':>11}{'NEAT':>11}")
    print("-" * 65)
    keys = [("Mean RA (%)", "RA_mean"), ("Mean FR (%)", "FR_mean"),
            ("Mean FT (%)", "FT_mean"), ("EC (/5)", "EC")]
    for label, k in keys:
        vals = [m[k] for m in metrics_list]
        print(f"  {label:<30}" + "".join(f"{v:>11}" for v in vals))
    print("-" * 65)
    mb, me, mn = metrics_list
    print(f"\n  EA vs Baseline RA gain:   {me['RA_mean'] - mb['RA_mean']:+.1f} pp")
    print(f"  NEAT vs Baseline RA gain: {mn['RA_mean'] - mb['RA_mean']:+.1f} pp")
    print(f"  EA FR / NEAT FR: {me['FR_mean']}% / {mn['FR_mean']}%")
    print(f"  EA EC / NEAT EC: {me['EC']}/5 / {mn['EC']}/5")
