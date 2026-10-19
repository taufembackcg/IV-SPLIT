#!/usr/bin/env python3
"""Turn Monte Carlo summary CSVs into the LaTeX tables included by paper/main.tex.

    python scripts/make_tables.py --fast results/fast/summary.csv --nn results/nn/summary.csv
"""

import argparse
import csv
import pathlib

LABELS = {
    "ols": "OLS", "oracle": "2SLS oracle", "sp": "SP", "lewbel": "Lewbel", "copula": "Copula CF",
    "cumulant": "Cumulant-ICA (L1)", "cumulant_aux": "Cumulant-ICA (L2)",
    "neural_iv": "Neural split IV", "neural_cf": "Neural split CF",
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
}


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if f != f else f"{f:.{nd}f}"


def table(rows, caption, label, estimators):
    ns = sorted({int(r["n"]) for r in rows})
    scen = [s for s in SCEN if any(r["scenario"] == s for r in rows)]
    reps = max(int(r["reps"]) for r in rows)
    out = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
           rf"\caption{{{caption} {reps} replications per cell; $\beta=1$.}}\label{{{label}}}",
           r"\begin{tabular}{@{}ll" + "rrrr" * len(ns) + "@{}}", r"\toprule"]
    out.append(" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$n={n}$}}" for n in ns) + r" \\")
    out.append(" ".join(rf"\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(ns))))
    out.append("Scenario & Estimator & " + " & ".join(["Bias & RMSE & Cov. & NI"] * len(ns)) + r" \\")
    out.append(r"\midrule")
    for s in scen:
        first = True
        for e in estimators:
            cells = []
            for n in ns:
                m = [r for r in rows if r["scenario"] == s and r["estimator"] == e and int(r["n"]) == n]
                if not m:
                    cells += ["--"] * 4
                    continue
                r = m[0]
                cells += [fmt(r["median_bias"]), fmt(r["rmse"]), fmt(r["coverage"], 2), fmt(r["not_identified"], 2)]
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
    ap.add_argument("--fast", required=True)
    ap.add_argument("--nn", required=True)
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    note = ("Bias is the median bias; Cov.\\ is the coverage of the nominal 95\\% interval "
            "(Wald with robust standard errors, percentile bootstrap for Cumulant-ICA); NI is the share "
            "of replications in which the estimator reports that $\\beta$ is not identified "
            "($D$ below the $\\chi^2_1$ 95\\% critical value; for the split also $F<10$ or an "
            "unbounded Anderson--Rubin set).")
    fast = read(a.fast)
    est_fast = [e for e in LABELS if any(r["estimator"] == e for r in fast)]
    (out / "pilot_fast.tex").write_text(table(fast, "Pilot Monte Carlo, closed-form estimators. " + note,
                                              "tab:pilot-fast", est_fast))
    nn = read(a.nn)
    est_nn = [e for e in LABELS if any(r["estimator"] == e for r in nn)]
    (out / "pilot_nn.tex").write_text(table(nn, "Pilot Monte Carlo, neural split (60 epochs). " + note,
                                            "tab:pilot-nn", est_nn))


if __name__ == "__main__":
    main()
