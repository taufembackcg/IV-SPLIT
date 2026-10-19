#!/usr/bin/env python3
"""Turn Monte Carlo summary CSVs into the LaTeX tables included by paper/main.tex.

    python scripts/make_tables.py --fast results/fast/summary.csv [more.csv ...] \
        --lam1 results/lam1/summary.csv --nn results/nn/summary.csv

Later --fast files override earlier ones cell by cell (scenario, n, estimator).
"""

import argparse
import csv
import pathlib

LABELS = {
    "ols": "OLS", "oracle": "2SLS oracle", "sp": "SP", "lewbel": "Lewbel", "copula": "Copula CF",
    "cumulant": "Cumulant-ICA (L1)", "cumulant_aux": "Cumulant-ICA (L2)",
    "neural_iv": "Neural split IV", "neural_cf": "Neural split CF",
    "anchored": "Anchored split (L1)", "anchored_aux": "Anchored split (L2)",
    "anchored_xfit": "Anchored split (L1, cross-fit)",
    "gan_hausman": "GAN, Hausman filter", "gan_ctest": "GAN, C-test filter",
    "gan_split": "GAN, sample split", "gan_partialled": "GAN, partialled Sargan",
    "gan_hsic_eval": "GAN + HSIC test", "gan_hsic_train": "GAN + HSIC in training",
}
SCEN = {
    "s1_latent_nongaussian": "S1 non-Gaussian factor",
    "s1_latent_nonlinear_g": "S1 non-linear $g$",
    "s2_nonlinear_confounding": "S2 non-linear confounding",
    "s3_gaussian": "S3 Gaussian",
    "s4_correlated_sources": "S4 $\\rho=0.3$",
    "s4_exclusion_violation": "S4 direct effect $0.3$",
    "s5_placebo": "S5 placebo ($\\lambda=0$)",
    "aux_heteroskedastic": "Aux heteroskedastic",
    "s1_asymmetric": "S1 asymmetric sources",
    "m2_latent_nongaussian": "$p=2$, Laplace",
    "m2_gaussian": "$p=2$, Gaussian",
}

DEFAULT_COLS = [
    ("Bias", lambda r: fmt(r["median_bias"])),
    ("RMSE", lambda r: fmt(r["rmse"])),
    ("Cov.", lambda r: fmt(r["coverage"], 2)),
    ("NI", lambda r: fmt(r["not_identified"], 2)),
]
GAN_COLS = [
    ("Bias", lambda r: fmt(r["median_bias"])),
    ("Cov.", lambda r: fmt(r["coverage"], 2)),
    ("Pass", lambda r: fmt(1 - float(r["not_identified"]), 2) if fmt(r.get("mean_sargan_p")) != "--" else "--"),
    ("$\\bar p_S$", lambda r: fmt(r.get("mean_sargan_p"), 2)),
    ("Kept", lambda r: fmt(r.get("mean_n_kept"), 1)),
    ("$\\rho_u$", lambda r: fmt(r.get("mean_corr_u"), 2)),
]
HSIC_COLS = [
    ("Bias", lambda r: fmt(r["median_bias"])),
    ("Pass", lambda r: fmt(1 - float(r["not_identified"]), 2) if fmt(r.get("mean_sargan_p")) != "--" else "--"),
    ("Rej.\\ HSIC", lambda r: fmt(r.get("mean_hsic_reject"), 2)),
    ("$\\rho_u$", lambda r: fmt(r.get("mean_corr_u"), 2)),
]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if f != f else f"{f:.{nd}f}"


def table(rows, caption, label, estimators, cols=None):
    cols = cols or DEFAULT_COLS
    k = len(cols)
    ns = sorted({int(r["n"]) for r in rows})
    scen = [s for s in SCEN if any(r["scenario"] == s for r in rows)]
    reps = max(int(r["reps"]) for r in rows)
    out = [r"\begin{table}[p]" if len(rows) > 40 else r"\begin{table}[ht]", r"\centering", r"\scriptsize",
           r"\setlength{\tabcolsep}{2.5pt}",
           rf"\caption{{{caption} {reps} replications per cell; $\beta=1$; $\lambda$ as in the text unless fixed by the scenario.}}\label{{{label}}}",
           r"\begin{tabular}{@{}ll" + "r" * k * len(ns) + "@{}}", r"\toprule"]
    out.append(" & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{$n={n}$}}" for n in ns) + r" \\")
    out.append(" ".join(rf"\cmidrule(lr){{{3 + k * i}-{2 + k * (i + 1)}}}" for i in range(len(ns))))
    out.append("Scenario & Estimator & " + " & ".join([" & ".join(h for h, _ in cols)] * len(ns)) + r" \\")
    out.append(r"\midrule")
    for s in scen:
        first = True
        for e in estimators:
            cells = []
            for n in ns:
                m = [r for r in rows if r["scenario"] == s and r["estimator"] == e and int(r["n"]) == n]
                if not m:
                    cells += ["--"] * k
                    continue
                cells += [fn(m[0]) for _, fn in cols]
            if all(c == "--" for c in cells):
                continue
            name = SCEN[s] if first else ""
            first = False
            out.append(f"{name} & {LABELS.get(e, e)} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--lam1", required=True)
    ap.add_argument("--nn", required=True)
    ap.add_argument("--gan")
    ap.add_argument("--anchored")
    ap.add_argument("--hsic")
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for path in a.fast:
        for r in read(path):
            merged[(r["scenario"], r["n"], r["estimator"])] = r
    fast = list(merged.values())
    est_fast = [e for e in LABELS if any(r["estimator"] == e for r in fast)]
    (out / "pilot_fast.tex").write_text(table(fast, "Pilot Monte Carlo, closed-form estimators. ",
                                              "tab:pilot-fast", est_fast))
    lam1 = read(a.lam1)
    est_l1 = [e for e in LABELS if any(r["estimator"] == e for r in lam1)]
    (out / "pilot_lam1.tex").write_text(table(lam1, "Pilot Monte Carlo, S1 versus S3 with $\\lambda=1$. ",
                                              "tab:pilot-lam1", est_l1))
    nn = read(a.nn)
    est_nn = [e for e in LABELS if any(r["estimator"] == e for r in nn)]
    (out / "pilot_nn.tex").write_text(table(nn, "Pilot Monte Carlo, neural split (60 epochs). ",
                                            "tab:pilot-nn", est_nn))
    if a.gan:
        gan = read(a.gan)
        est_g = [e for e in LABELS if any(r["estimator"] == e for r in gan)]
        (out / "mc_gan.tex").write_text(table(
            gan, "Generated instruments (GAN), $\\lambda=1$. Pass: share of replications in which "
            "every endogenous regressor passes the weak-instrument and Sargan tests; $\\bar p_S$: "
            "mean of the smallest per-variable Sargan $p$-value; Kept: candidates left after "
            "filtering; $\\rho_u$: correlation of the fitted first stage of $x_1$ with $u$. ",
            "tab:mc-gan", est_g, GAN_COLS))
    if a.hsic:
        hs = read(a.hsic)
        est_h = [e for e in LABELS if any(r["estimator"] == e for r in hs)]
        (out / "mc_gan_hsic.tex").write_text(table(
            hs, "Generated instruments with the HSIC test, $\\lambda=1$. All rows train on one half "
            "and test on the other. Pass: share of replications passing every test (weak instruments, "
            "Sargan and, for the HSIC rows, the HSIC test at 5\\%); Rej.\\ HSIC: rejection rate of the "
            "HSIC test between the W-partialled instruments and the 2SLS residual; $\\rho_u$ as in "
            "Table~\\ref{tab:mc-gan}. ", "tab:mc-gan-hsic", est_h, HSIC_COLS))
    if a.anchored:
        anc = read(a.anchored)
        est_a = [e for e in LABELS if any(r["estimator"] == e for r in anc)]
        (out / "mc_anchored.tex").write_text(table(
            anc, "Anchored split versus the cumulant layer, $\\lambda=1$. ", "tab:mc-anchored", est_a))


if __name__ == "__main__":
    main()
