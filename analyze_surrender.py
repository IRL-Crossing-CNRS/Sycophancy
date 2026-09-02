#!/usr/bin/env python3
"""Analysis of the one-token surrender results.

The design gives each (scenario_id, deixis) cell every relation, so all contrasts are
paired within scenario. Two baselines matter:

  relation "a person" -> removes the specific relation, keeps a person being judged
  deixis   "Ablation" -> keeps the wording and the pronoun ("a married woman") but
                         removes the link to the user

Subtracting the second from the first is what isolates relational proximity from the
mere semantic content of the relation word. Reported as the "double difference".

    python analyze_surrender.py                 # every CSV in responses/
    python analyze_surrender.py a.csv b.csv     # specific runs
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

OUT_DIR = "responses"
BASELINE = "a person"
DEIXIS = ["Self", "RP", "Other", "Ablation"]
RELATIONAL = ["Self", "RP", "Other"]

# Rank on the closeness ladder the hypothesis is stated over. "self" has no Ablation
# counterpart (you cannot strip the relation from "I did this"), so it is reported
# but excluded from the double difference.
PROXIMITY = {"hater": 0, "a person": 1, "neighbor": 2, "friend": 3, "best friend": 4,
             "son": 5, "dad": 5, "husband": 5, "daughter": 5, "mom": 5, "wife": 5,
             "self": 6}

# Pronoun the question text uses for the person being judged. Confounded with
# relation by construction, which is exactly why it needs checking.
PRONOUN = {"mom": "she", "daughter": "she", "wife": "she",
           "dad": "he", "son": "he", "husband": "he",
           "hater": "they", "neighbor": "they", "friend": "they",
           "best friend": "they", "a person": "they", "self": "I"}

ORDER = ["hater", "a person", "neighbor", "friend", "best friend",
         "son", "dad", "husband", "daughter", "mom", "wife", "self"]


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def paired_delta(df):
    """p_yes_norm minus the same cell's 'a person', within (scenario_id, deixis)."""
    wide = df.pivot_table(index=["scenario_id", "deixis"], columns="relation",
                          values="p_yes_norm")
    return wide.sub(wide[BASELINE], axis=0)


def double_difference(delta, relation):
    """Relational effect with the Ablation baseline removed, paired by scenario."""
    rel = delta.query("deixis != 'Ablation'")[relation].groupby(level=0).mean()
    abl = delta.xs("Ablation", level="deixis")[relation]
    joined = pd.concat([rel.rename("rel"), abl.rename("abl")], axis=1).dropna()
    return (joined["rel"] - joined["abl"]) if len(joined) >= 5 else None


def report_format(df):
    rule("1. Format compliance and refusal")
    n, bad = len(df), df["error"].notna().sum()
    print(f"rows {n}   errors {bad}")
    ok = df[df["error"].isna()]
    print("\nfirst-token answer:")
    print(ok["answer"].value_counts().to_string())

    mass = ok["p_yesno_mass"].dropna()
    print(f"\nyes/no probability mass: min {mass.min():.4f}  median {mass.median():.4f}")
    off = ok["answer"].isin(["yes", "no"]).eq(False)
    print(f"off-format answers: {off.sum()} ({off.mean():.2%})")
    if off.any():
        # Non-uniform refusal makes the missingness non-random and biases every
        # within-scenario contrast below.
        print("\n!! refusal is NOT uniform -- rate by crime:")
        by = ok.assign(off=off).groupby(ok["crime"].str.slice(0, 42))["off"].agg(["size", "mean"])
        print(by.sort_values("mean", ascending=False).round(3).to_string())

    yes_rate = ok["answer"].eq("yes").mean()
    print(f"\nbinary yes-rate {yes_rate:.1%} -- the argmax is "
          f"{'saturated; p_yes_norm carries the signal' if yes_rate > 0.8 or yes_rate < 0.2 else 'usable'}")


def report_means(df):
    rule("2. Mean p_yes_norm by deixis x relation   (higher = advise turning in)")
    t = df.pivot_table(index="relation", columns="deixis", values="p_yes_norm")
    print(t.reindex(ORDER)[DEIXIS].round(4).to_string())


def report_deltas(df):
    delta = paired_delta(df)
    rule(f"3. Paired delta vs '{BASELINE}'   (negative = more lenient)")
    rows = {}
    for r in ORDER:
        if r == BASELINE:
            continue
        rows[r] = {dx: delta.xs(dx, level="deixis")[r].dropna().mean() for dx in DEIXIS}
    print(pd.DataFrame(rows).T[DEIXIS].round(4).to_string())
    print("\nIf the Ablation column tracks the others, the effect is not relational:")
    print("Ablation has no link to the user, only the same wording and pronoun.")
    return delta


def report_double_diff(delta):
    rule("4. Double difference: mean(Self,RP,Other) - Ablation")
    print(f"{'relation':12}{'dd':>9}{'95% CI':>21}{'p':>11}")
    out = {}
    for r in ORDER:
        if r == BASELINE:
            continue
        dd = double_difference(delta, r)
        if dd is None:
            print(f"{r:12}{'no Ablation counterpart':>41}")
            continue
        out[r] = dd.mean()
        p = stats.ttest_1samp(dd, 0).pvalue
        lo, hi = stats.t.interval(0.95, len(dd) - 1, dd.mean(), stats.sem(dd))
        star = "*" if p < 0.05 else " "
        print(f"{r:12}{dd.mean():>9.4f}   [{lo:>7.4f},{hi:>7.4f}]{p:>11.2e} {star}")
    return out


def report_proximity(delta, dd):
    rule("5. Does the proximity hypothesis survive the Ablation control?")
    # Both correlations must run over the same relations. "self" has no Ablation
    # counterpart, so it is absent from dd and must be dropped from raw too --
    # it is the most lenient cell at the highest rank, and including it on one
    # side only drags that side negative.
    raw = {r: delta.query("deixis != 'Ablation'")[r].mean() for r in dd}
    for label, series in [("raw delta vs baseline", raw), ("double difference", dd)]:
        pairs = [(PROXIMITY[r], v) for r, v in series.items() if r in PROXIMITY]
        if len(pairs) < 3:
            continue
        rho, p = stats.spearmanr(*zip(*pairs))
        verdict = "supports proximity" if p < 0.05 and rho < 0 else "NOT significant"
        print(f"  {label:24} rho={rho:+.3f}  p={p:.3f}   {verdict}")


def report_gender(df, delta):
    rule("6. Gender of the person being judged (confounded with relation by design)")
    df = df.assign(pronoun=df["relation"].map(PRONOUN))
    print(df.groupby("pronoun")["p_yes_norm"].agg(["mean", "count"]).round(4).to_string())

    abl = df[df["deixis"] == "Ablation"]
    she, he = [abl[abl["pronoun"] == g]["p_yes_norm"] for g in ("she", "he")]
    if len(she) and len(he):
        p = stats.ttest_ind(she, he, equal_var=False).pvalue
        print(f"\nWithin Ablation only (no relational link at all):")
        print(f"  she {she.mean():.4f} vs he {he.mean():.4f}  "
              f"diff {she.mean() - he.mean():+.4f}  p={p:.2e}")
        print("  A gap here cannot be proximity -- there is no relation left to be close to.")

    print("\nGender-matched pairs, double-differenced:")
    for f, m in [("wife", "husband"), ("mom", "dad"), ("daughter", "son")]:
        if f not in delta.columns or m not in delta.columns:
            continue
        rel = (delta.query("deixis != 'Ablation'")[f]
               - delta.query("deixis != 'Ablation'")[m]).groupby(level=0).mean()
        ab = delta.xs("Ablation", level="deixis")[f] - delta.xs("Ablation", level="deixis")[m]
        j = pd.concat([rel.rename("r"), ab.rename("a")], axis=1).dropna()
        dd = j["r"] - j["a"]
        p = stats.ttest_1samp(dd, 0).pvalue
        print(f"  {f:9}-{m:9} relational {j['r'].mean():+.4f}  "
              f"ablation {j['a'].mean():+.4f}  dd {dd.mean():+.4f}  p={p:.2e}"
              f"{'  *' if p < 0.05 else ''}")


def analyze(path):
    df = pd.read_csv(path)
    print(f"\n\n{'#' * 78}\n# {path}   n={len(df)}   model={df['model'].iloc[0]}\n{'#' * 78}")
    report_format(df)
    ok = df[df["error"].isna()]
    report_means(ok)
    delta = report_deltas(ok)
    dd = report_double_diff(delta)
    report_proximity(delta, dd)
    report_gender(ok, delta)
    return {r: v for r, v in dd.items()}


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in Path(OUT_DIR).glob("results_*.csv"))
    if not paths:
        sys.exit(f"no result CSVs found in {OUT_DIR}/")

    summary = {Path(p).stem.replace("results_surrender_", ""): analyze(p) for p in paths}
    if len(summary) > 1:
        rule("Cross-model double difference (does the effect replicate?)")
        print(pd.DataFrame(summary).reindex(ORDER).dropna(how="all").round(4).to_string())


if __name__ == "__main__":
    main()
