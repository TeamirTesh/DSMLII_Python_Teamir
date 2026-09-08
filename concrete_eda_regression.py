"""
Project 1 - Concrete Compressive Strength
Exploratory Data Analysis and Simple Regression (Python / statsmodels portion)

Runs the full 5-step workflow and writes every artifact to ./output/:

  1. Prepare    - string / missing / duplicate / outlier checks (decisions documented)
  2. Summarize  - statistical summary for every feature and the response
  3. Explore    - correlation matrix + heatmap, pick the top two predictors
  4. Model      - separate statsmodels OLS simple regressions for both predictors
  5. Communicate- fit reports, short interpretations, observed-vs-fitted + residual plots

Usage:
    python concrete_eda_regression.py [path_to_Concrete_Data.xls]

AI Usage:
    I used AI to help format the output directly into a LaTex report/file so that one can easily
    rerun the script and generate a new report automatically without editing LaTex repeatedly.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DEFAULT_XLS = HERE.parent / "Concrete_Data.xls"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# Figures named/styled to drop straight into the group LaTeX report.
REPORTFIG = OUT / "report_figures"
REPORTFIG.mkdir(exist_ok=True)

# tidy snake_case name -> CamelCase stem used in the LaTeX \includegraphics calls
FIG_STEM = {
    "cement": "Cement",
    "slag": "Slag",
    "fly_ash": "FlyAsh",
    "water": "Water",
    "superplasticizer": "Superplasticizer",
    "coarse_agg": "CoarseAgg",
    "fine_agg": "FineAgg",
    "age": "Age",
}

# 8 input variables + 1 output, in the column order documented in Concrete_Readme.txt
COLUMNS = [
    "cement",            # kg/m^3
    "slag",              # blast furnace slag, kg/m^3
    "fly_ash",           # kg/m^3
    "water",             # kg/m^3
    "superplasticizer",  # kg/m^3
    "coarse_agg",        # coarse aggregate, kg/m^3
    "fine_agg",          # fine aggregate, kg/m^3
    "age",               # days (1..365)
    "strength",          # concrete compressive strength, MPa  (RESPONSE)
]
RESPONSE = "strength"
PREDICTORS = [c for c in COLUMNS if c != RESPONSE]

sns.set_theme(context="notebook", style="whitegrid")


def rule(title: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}"


def _latex_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Escape LaTeX-special chars in column and index labels before to_latex."""
    def esc(x):
        return str(x).replace("_", r"\_").replace("%", r"\%")
    out = frame.copy()
    out.columns = [esc(c) for c in out.columns]
    out.index = [esc(i) for i in out.index]
    out.index.name = esc(frame.index.name) if frame.index.name else None
    return out


# --------------------------------------------------------------------------- #
# Report figures
# --------------------------------------------------------------------------- #
def make_eda_plots(df: pd.DataFrame) -> None:
    """One 'response vs feature' scatter per predictor - used for the visual
    outlier inspection described in the Prepare step."""
    for p in PREDICTORS:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[p], df[RESPONSE], s=16, alpha=0.45, edgecolor="none")
        # mark the single most extreme response value for reference
        imax = df[RESPONSE].idxmax()
        ax.scatter(df.loc[imax, p], df.loc[imax, RESPONSE], s=70,
                   facecolor="none", edgecolor="crimson", linewidth=1.6,
                   label=f"max {RESPONSE} = {df[RESPONSE].max():.1f} MPa")
        ax.set_xlabel(p)
        ax.set_ylabel(f"{RESPONSE} (MPa)")
        ax.set_title(f"{RESPONSE} vs {p}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORTFIG / f"{FIG_STEM[p]}EDA.png", dpi=150)
        plt.close(fig)
    print(f"  wrote {len(PREDICTORS)} EDA scatter plots to {REPORTFIG.name}/")


def make_y_vs_yhat_ordered(df: pd.DataFrame, predictor: str, model) -> None:
    """Actual vs predicted response, samples ordered by the predictor
    (same diagnostic style as the Auto MPG section)."""
    order = np.argsort(df[predictor].to_numpy())
    y = df[RESPONSE].to_numpy()[order]
    yhat = np.asarray(model.fittedvalues)[order]
    idx = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(idx, y, s=14, alpha=0.35, edgecolor="none", color="steelblue",
               label="actual $y$")
    ax.plot(idx, yhat, color="crimson", lw=2, label=r"predicted $\hat{y}$")
    ax.set_xlabel(f"sample index (ordered by {predictor})")
    ax.set_ylabel(f"{RESPONSE} (MPa)")
    ax.set_title(f"y vs yhat ordered by {FIG_STEM[predictor]}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(REPORTFIG / f"y_vs_yhat_{predictor}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_data(xls_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xls_path, engine="xlrd")
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, got {raw.shape[1]}: {list(raw.columns)}"
        )
    df = raw.copy()
    df.columns = COLUMNS  # rename long header text -> tidy snake_case
    return df, raw


# --------------------------------------------------------------------------- #
# Step 1 - Prepare
# --------------------------------------------------------------------------- #
def step1_prepare(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    lines: list[str] = []
    lines.append(rule("STEP 1 - PREPARE"))
    lines.append(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
    lines.append(f"Original headers:\n  " + "\n  ".join(map(str, raw.columns)))
    lines.append(f"Renamed to: {COLUMNS}")

    # --- string / type check -------------------------------------------------
    lines.append("\n-- Column dtypes --")
    for c in df.columns:
        lines.append(f"  {c:<18} {df[c].dtype}")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    coerced_bad = {}
    for c in df.columns:
        coerced = pd.to_numeric(df[c], errors="coerce")
        n_bad = int(coerced.isna().sum() - df[c].isna().sum())
        if n_bad:
            coerced_bad[c] = n_bad
    lines.append(
        "\nString-value check: "
        + (
            "all 9 columns are numeric; no stray text / categorical codes found. "
            "Decision: no type conversion or category encoding needed."
            if not non_numeric and not coerced_bad
            else f"non-numeric columns={non_numeric}, values that fail numeric coercion={coerced_bad}"
        )
    )

    # --- missing values ----------------------------------------------------
    miss = df.isna().sum()
    total_missing = int(miss.sum())
    lines.append("\n-- Missing values per column --")
    for c in df.columns:
        lines.append(f"  {c:<18} {int(miss[c])}")
    lines.append(
        f"\nMissing-value check: {total_missing} missing cells total. "
        + (
            "Matches the dataset documentation ('Missing Attribute Values: None'). "
            "Decision: no imputation or row dropping required."
            if total_missing == 0
            else "Decision: see notes above."
        )
    )

    # --- duplicate rows --------------------------------------------------
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    n_unique = len(df) - n_dup
    lines.append(
        textwrap.dedent(
            f"""
            -- Duplicate rows --
              Fully-duplicated rows (all 9 columns identical): {n_dup}
              Distinct rows: {n_unique} of {len(df)}
            Decision: KEEP duplicates. Each row is a concrete mix + curing age whose
            strength was measured in the lab; identical mix designs tested at the same
            age are legitimate experimental replicates, not data-entry errors. Removing
            them would bias the summary statistics toward rarely-repeated mixes and
            discard real measurement information. The count is reported here for
            transparency and can be revisited if the modeling step shows leverage
            problems. (A de-duplicated copy is also written: concrete_unique.csv.)
            """
        ).rstrip()
    )

    # --- outliers (IQR rule, per feature) -----------------------------------
    lines.append("\n-- Outlier scan (Tukey 1.5*IQR fences, per column) --")
    rows = []
    for c in df.columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below = int((df[c] < lo).sum())
        above = int((df[c] > hi).sum())
        n_out = below + above
        rows.append(
            {
                "feature": c,
                "lower_fence": round(lo, 3),
                "upper_fence": round(hi, 3),
                "n_below": below,
                "n_above": above,
                "n_outliers": n_out,
                "pct_outliers": round(100 * n_out / len(df), 2),
                "min": round(df[c].min(), 3),
                "max": round(df[c].max(), 3),
                "skew": round(df[c].skew(), 3),
            }
        )
    outlier_tbl = pd.DataFrame(rows).set_index("feature")
    outlier_tbl.to_csv(OUT / "outlier_report.csv")
    lines.append(outlier_tbl.to_string())
    lines.append(
        textwrap.dedent(
            """
            Decision: DO NOT remove any rows for outliers.
              * slag, fly_ash, superplasticizer contain many exact zeros - these are
                structural zeros (the ingredient was simply not used in that mix), not
                errors, so the IQR rule flags them spuriously.
              * age is strongly right-skewed (28-day and 90/365-day tests are standard);
                large values are the real design of the study, not anomalies.
              * The remaining flagged points on water, fine_agg, superplasticizer, age
                are physically plausible mix quantities within civil-engineering ranges.
              * n is only 1030 and this is raw (unscaled) experimental data; dropping
                tails would shrink the response range and inflate the regression fit.
            The fences and counts above are kept as documentation. If a single
            regression later shows a very high-leverage point, it can be revisited.
            """
        ).rstrip()
    )

    (OUT / "prepare_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # write cleaned copies (no row changes; just tidy names + explicit dtypes)
    clean = df.astype(float)
    clean.to_csv(OUT / "concrete_clean.csv", index=False)
    clean.drop_duplicates().to_csv(OUT / "concrete_unique.csv", index=False)
    return clean


# --------------------------------------------------------------------------- #
# Step 2 - Summarize
# --------------------------------------------------------------------------- #
def step2_summarize(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    desc["range"] = desc["max"] - desc["min"]
    desc["skew"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    desc["cv"] = desc["std"] / desc["mean"]  # coefficient of variation
    desc = desc[
        ["count", "mean", "std", "cv", "min", "25%", "50%", "75%", "max", "range",
         "skew", "kurtosis"]
    ].round(3)

    # order: predictors first, response last, clearly labeled
    desc = desc.reindex(PREDICTORS + [RESPONSE])
    desc.index.name = "variable"
    desc.to_csv(OUT / "summary_statistics.csv")
    _latex_safe(desc).to_latex(
        OUT / "summary_statistics.tex",
        float_format="%.3f",
        caption="Summary statistics for the Concrete Compressive Strength features and response (MPa).",
        label="tab:concrete-summary",
    )

    print(rule("STEP 2 - SUMMARIZE  (also -> summary_statistics.csv / .tex)"))
    print(desc.to_string())
    return desc


# --------------------------------------------------------------------------- #
# Step 3 - Explore
# --------------------------------------------------------------------------- #
def step3_explore(df: pd.DataFrame) -> list[str]:
    corr = df.corr(method="pearson").round(4)
    corr.to_csv(OUT / "correlation_matrix.csv")

    target_corr = (
        corr[RESPONSE]
        .drop(RESPONSE)
        .sort_values(key=np.abs, ascending=False)
        .rename("pearson_r_with_strength")
        .to_frame()
    )
    target_corr["abs_r"] = target_corr["pearson_r_with_strength"].abs().round(4)
    target_corr.to_csv(OUT / "correlation_with_target.csv")

    top2 = target_corr.index[:2].tolist()

    # heatmap
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Concrete data - Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=150)
    fig.savefig(REPORTFIG / "ConcreteCorrelationHeatmap.png", dpi=150)
    plt.close(fig)

    print(rule("STEP 3 - EXPLORE  (also -> correlation_matrix.csv, correlation_heatmap.png)"))
    print("Correlation with response (|r| descending):")
    print(target_corr.to_string())
    print(f"\nTop two predictors by |Pearson r| with {RESPONSE}: {top2}")
    return top2


# --------------------------------------------------------------------------- #
# Steps 4 & 5 - Model + Communicate
# --------------------------------------------------------------------------- #
def fit_and_report(df: pd.DataFrame, predictor: str) -> dict:
    model = smf.ols(f"{RESPONSE} ~ {predictor}", data=df).fit()

    # ---- fit report (Step 5) ----
    report_path = OUT / f"fit_{predictor}.txt"
    b0, b1 = model.params["Intercept"], model.params[predictor]
    ci = model.conf_int().loc[predictor].tolist()
    interp = textwrap.dedent(
        f"""
        SIMPLE LINEAR REGRESSION  ({RESPONSE} ~ {predictor})
        {'-' * 60}
        Fitted line : {RESPONSE} = {b0:.4f} + {b1:.4f} * {predictor}
        Slope 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]
        R-squared   : {model.rsquared:.4f}   (Adj. {model.rsquared_adj:.4f})
        RMSE        : {np.sqrt(model.mse_resid):.4f} MPa
        F p-value   : {model.f_pvalue:.3e}      slope p-value: {model.pvalues[predictor]:.3e}
        n           : {int(model.nobs)}

        Interpretation:
          * Each +1 kg/m^3 (or +1 day, for age) of '{predictor}' is associated with a
            change of {b1:+.4f} MPa in predicted compressive strength.
          * '{predictor}' alone explains {model.rsquared * 100:.1f}% of the variance in
            strength; the remaining {100 - model.rsquared * 100:.1f}% reflects the other
            ingredients, curing age, and nonlinearity that a one-variable straight-line
            model cannot capture.
          * The slope is {'statistically significant' if model.pvalues[predictor] < 0.05
            else 'NOT statistically significant'} at alpha = 0.05.
          * Typical prediction error (RMSE) is about {np.sqrt(model.mse_resid):.1f} MPa
            against a response that ranges roughly {df[RESPONSE].min():.0f}-{df[RESPONSE].max():.0f} MPa.
        """
    ).strip()
    report_path.write_text(
        interp + "\n\n" + "=" * 78 + "\nFULL statsmodels SUMMARY\n" + "=" * 78 + "\n"
        + model.summary().as_text(),
        encoding="utf-8",
    )

    # ---- plot 1: scatter + fitted line ----
    xs = np.linspace(df[predictor].min(), df[predictor].max(), 200)
    pred = model.get_prediction(pd.DataFrame({predictor: xs})).summary_frame()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[predictor], df[RESPONSE], s=14, alpha=0.4, label="observed")
    ax.plot(xs, pred["mean"], color="crimson", lw=2, label="OLS fit")
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"],
                    color="crimson", alpha=0.15, label="95% mean CI")
    ax.set_xlabel(predictor)
    ax.set_ylabel(f"{RESPONSE} (MPa)")
    ax.set_title(f"{RESPONSE} vs {predictor}  (R^2={model.rsquared:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"plot_regression_{predictor}.png", dpi=150)
    plt.close(fig)

    # ---- plot 2: observed vs fitted (Step 5 requirement) ----
    fitted = model.fittedvalues
    lo = min(df[RESPONSE].min(), fitted.min())
    hi = max(df[RESPONSE].max(), fitted.max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fitted, df[RESPONSE], s=14, alpha=0.4)
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1, label="perfect fit (y=x)")
    ax.set_xlabel(f"fitted {RESPONSE} (MPa)")
    ax.set_ylabel(f"observed {RESPONSE} (MPa)")
    ax.set_title(f"Observed vs fitted - {RESPONSE} ~ {predictor}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"plot_obs_vs_fitted_{predictor}.png", dpi=150)
    plt.close(fig)

    # ---- plot 3: residuals vs fitted ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, model.resid, s=14, alpha=0.4)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(f"fitted {RESPONSE} (MPa)")
    ax.set_ylabel("residual (MPa)")
    ax.set_title(f"Residuals vs fitted - {RESPONSE} ~ {predictor}")
    fig.tight_layout()
    fig.savefig(OUT / f"plot_residuals_{predictor}.png", dpi=150)
    plt.close(fig)

    # ---- plot 4: y vs yhat ordered by the predictor (LaTeX report style) ----
    make_y_vs_yhat_ordered(df, predictor, model)

    icpt_se, icpt_t, icpt_p = (
        model.bse["Intercept"], model.tvalues["Intercept"], model.pvalues["Intercept"]
    )
    icpt_ci = model.conf_int().loc["Intercept"].tolist()
    return {
        "predictor": predictor,
        "intercept": b0,
        "intercept_se": float(icpt_se),
        "intercept_t": float(icpt_t),
        "intercept_p": float(icpt_p),
        "intercept_ci_low": float(icpt_ci[0]),
        "intercept_ci_high": float(icpt_ci[1]),
        "slope": b1,
        "slope_se": float(model.bse[predictor]),
        "slope_t": float(model.tvalues[predictor]),
        "slope_ci_low": ci[0],
        "slope_ci_high": ci[1],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "rmse": float(np.sqrt(model.mse_resid)),
        "fvalue": float(model.fvalue),
        "slope_p_value": float(model.pvalues[predictor]),
        "f_p_value": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "df_resid": int(model.df_resid),
        "n": int(model.nobs),
        "_model": model,
    }


def steps45_model(df: pd.DataFrame, top2: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    print(rule("STEPS 4 & 5 - MODEL + COMMUNICATE"))
    results = [fit_and_report(df, p) for p in top2]
    tbl = (
        pd.DataFrame([{k: v for k, v in r.items() if k != "_model"} for r in results])
        .set_index("predictor")
        .round(4)
    )
    tbl.to_csv(OUT / "regression_results.csv")
    _latex_safe(tbl).to_latex(
        OUT / "regression_results.tex",
        float_format="%.4f",
        caption="Simple linear regression of compressive strength on each top predictor (statsmodels OLS).",
        label="tab:concrete-regression",
    )
    for p in top2:
        print(f"\n--- fit_{p}.txt ---")
        print((OUT / f"fit_{p}.txt").read_text(encoding="utf-8").split("FULL statsmodels")[0].strip())
    print("\nRegression comparison table (also -> regression_results.csv / .tex):")
    print(tbl.to_string())
    return tbl, results


# --------------------------------------------------------------------------- #
# Rollup
# --------------------------------------------------------------------------- #
def write_results_md(summary: pd.DataFrame, top2: list[str], reg: pd.DataFrame) -> None:
    best = reg["r_squared"].idxmax()
    md = []
    md.append("# Concrete Compressive Strength - Python results rollup\n")
    md.append("Generated by `concrete_eda_regression.py`. All artifacts are in `output/`.\n")
    md.append("## 1. Prepare\n")
    md.append("- 1030 rows x 9 columns; all columns numeric, **0 missing cells** "
              "(matches dataset docs). No type conversion needed.\n"
              "- Fully-duplicated rows are reported in `prepare_report.txt`; decision: "
              "**kept** (legitimate lab replicates of identical mix+age).\n"
              "- Outliers scanned with the 1.5*IQR rule (`outlier_report.csv`); decision: "
              "**no rows dropped** - zeros are structural, `age` skew is by design, and n is small.\n")
    md.append("## 2. Summarize\n")
    md.append("See `summary_statistics.csv` / `.tex`. Key points:\n")
    md.append(f"- Response `strength`: mean {summary.loc['strength','mean']:.2f} MPa, "
              f"sd {summary.loc['strength','std']:.2f}, range "
              f"{summary.loc['strength','min']:.2f}-{summary.loc['strength','max']:.2f}.\n")
    md.append(f"- Most skewed inputs: "
              + ", ".join(f"`{i}` ({summary.loc[i,'skew']:+.2f})"
                          for i in summary['skew'].drop('strength').sort_values(key=np.abs, ascending=False).index[:3])
              + ".\n")
    md.append("## 3. Explore\n")
    tc = pd.read_csv(OUT / "correlation_with_target.csv", index_col=0)
    md.append("Pearson r with `strength` (|r| desc):\n\n")
    md.append(tc.to_markdown() + "\n\n")
    md.append(f"**Top two predictors selected: `{top2[0]}` and `{top2[1]}`.**\n")
    md.append("## 4 & 5. Model + Communicate\n")
    md.append("Simple OLS regressions (statsmodels), one predictor each:\n\n")
    md.append(reg.to_markdown() + "\n\n")
    md.append(f"- Better single-predictor fit: **`{best}`** "
              f"(R^2 = {reg.loc[best,'r_squared']:.3f}, RMSE = {reg.loc[best,'rmse']:.2f} MPa).\n")
    md.append("- Per-model fit reports with interpretations: "
              + ", ".join(f"`fit_{p}.txt`" for p in top2) + ".\n")
    md.append("- Plots per model: `plot_regression_<p>.png`, "
              "`plot_obs_vs_fitted_<p>.png`, `plot_residuals_<p>.png`.\n")
    (OUT / "RESULTS.md").write_text("".join(md), encoding="utf-8")
    print(rule("WROTE output/RESULTS.md"))


# --------------------------------------------------------------------------- #
# LaTeX section (drops into the group report, matches the Auto MPG style)
# --------------------------------------------------------------------------- #
def _p(v: float) -> str:
    return "$<$0.001" if v < 1e-3 else f"{v:.3f}"


def _sci(v: float) -> str:
    if v == 0 or not np.isfinite(v):
        return "$<10^{-300}$"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10 ** exp
    return f"${mant:.2f} \\times 10^{{{exp}}}$"


def write_latex_section(df, summary, top2, results) -> None:
    n_rows = len(df)
    n_dup = int(df.duplicated().sum())
    n_unique = n_rows - n_dup
    n_missing = int(df.isna().sum().sum())
    p1, p2 = top2
    P1, P2 = FIG_STEM[p1], FIG_STEM[p2]
    r1 = next(r for r in results if r["predictor"] == p1)
    r2 = next(r for r in results if r["predictor"] == p2)

    label = {
        "cement": "Cement", "slag": "Blast Furnace Slag", "fly_ash": "Fly Ash",
        "water": "Water", "superplasticizer": "Superplasticizer",
        "coarse_agg": "Coarse Aggregate", "fine_agg": "Fine Aggregate",
        "age": "Age", "strength": "Compressive Strength",
    }

    # --- EDA figure block -------------------------------------------------
    eda_figs = "\n".join(
        "\\begin{figure}[H]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.8\\textwidth]{{{FIG_STEM[p]}EDA.png}}\n"
        f"    \\caption{{{label[p]} EDA}}\n"
        f"    \\label{{fig:concrete-{p}-eda}}\n"
        "\\end{figure}"
        for p in PREDICTORS
    )

    # --- summary-statistics table rows ---------------------------------
    srows = ""
    for v in PREDICTORS + [RESPONSE]:
        s = summary.loc[v]
        srows += (
            f"{label[v]} & {s['count']:.0f} & {s['mean']:.3f} & {s['std']:.3f} & "
            f"{s['min']:.3f} & {s['25%']:.3f} & {s['50%']:.3f} & {s['75%']:.3f} & "
            f"{s['max']:.3f} & {s['skew']:.3f} \\\\\n"
        )

    # --- correlation-with-target rows -------------------------------
    corr = df.corr(method="pearson")[RESPONSE].drop(RESPONSE)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    crows = "".join(
        f"{label[k]} & {val:+.3f} & {abs(val):.3f} \\\\\n" for k, val in corr.items()
    )

    # --- coefficient table rows (two separate simple regressions) ------
    def coef_rows(r, pred):
        return (
            f"strength $\\sim$ {label[pred]} & Intercept & {r['intercept']:.4f} & "
            f"{r['intercept_se']:.4f} & {r['intercept_t']:.3f} & {_p(r['intercept_p'])} & "
            f"[{r['intercept_ci_low']:.4f},\\,{r['intercept_ci_high']:.4f}] \\\\\n"
            f" & {label[pred]} & {r['slope']:.4f} & {r['slope_se']:.4f} & "
            f"{r['slope_t']:.3f} & {_p(r['slope_p_value'])} & "
            f"[{r['slope_ci_low']:.4f},\\,{r['slope_ci_high']:.4f}] \\\\\n"
        )

    r1_r = r1["r_squared"] ** 0.5 * np.sign(corr[p1])
    r2_r = r2["r_squared"] ** 0.5 * np.sign(corr[p2])
    p12 = float(df[[p1, p2]].corr().iloc[0, 1])
    best = "Cement" if r1["r_squared"] >= r2["r_squared"] else "Superplasticizer"

    template = r"""% ======================================================================
%  Concrete Compressive Strength -- generated by concrete_eda_regression.py
%  Copy every PNG from output/report_figures/ next to this .tex file, then
%  \input{concrete_section.tex} (or paste this block) into the group report.
% ======================================================================
\section{Concrete Compressive Strength Dataset}
\subsection{Statsmodels}

I began by confirming data quality. All nine columns (eight ingredient/age
predictors and the response, compressive strength in MPa) load as numeric, so
no string parsing or categorical encoding was required. There are @N_MISSING@
missing cells, which matches the dataset documentation, so no imputation or row
removal was needed on that basis. I also checked for exact duplicate rows:
@N_DUP@ of the @N_ROWS@ rows are full duplicates (@N_UNIQUE@ distinct rows).
These are identical mix designs measured at the same curing age, i.e.\
legitimate laboratory replicates rather than data-entry errors, so I kept them;
removing them would bias the summary statistics and discard real measurement
information.
\\

Next, I inspected the data for outliers by evaluating the EDA plots of $y$ vs
each feature.
\\

@EDA_FIGS@

The Tukey $1.5\times$IQR rule flags points on \texttt{age},
\texttt{superplasticizer}, \texttt{water}, \texttt{slag}, and
\texttt{fine\_agg}, but the EDA plots show these are not true anomalies:
\texttt{slag}, \texttt{fly\_ash}, and \texttt{superplasticizer} contain many
exact zeros because that ingredient was simply not used in the mix (structural
zeros), and \texttt{age} is right-skewed by design because strength is routinely
tested at 3, 7, 28, 56, 90, and 365 days. Every flagged value is a physically
plausible mix quantity, the sample is only @N_ROWS@ rows, and the data is raw
(unscaled) experimental data, so I did not remove any rows for outliers.
\\

Next, I evaluated the statistical summaries for every feature and the response.
\\

\begin{table}[H]
\centering
\caption{Statistical Summaries of the Concrete Features and Response}
\label{tab:concrete_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\hline
\textbf{Variable} & \textbf{Count} & \textbf{Mean} & \textbf{Std. Dev.} &
\textbf{Min} & \textbf{25\%} & \textbf{50\%} & \textbf{75\%} & \textbf{Max} &
\textbf{Skew} \\
\hline
@SROWS@\hline
\end{tabular}%
}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=1\textwidth]{ConcreteCorrelationHeatmap.png}
    \caption{Correlation Heatmap}
    \label{fig:concrete-correlation-heatmap}
\end{figure}

\begin{table}[H]
\centering
\caption{Pearson Correlation of Each Predictor with Compressive Strength}
\label{tab:concrete_target_corr}
\begin{tabular}{lrr}
\hline
\textbf{Predictor} & \textbf{Pearson $r$} & \textbf{$|r|$} \\
\hline
@CROWS@\hline
\end{tabular}
\end{table}

Using the correlation heatmap I selected the top two predictors as the features
with the largest absolute correlation with the target: \textbf{Cement}
($r = @R1_R@$) and \textbf{Superplasticizer} ($r = @R2_R@$). These two are also
nearly uncorrelated with each other ($r \approx @P12@$), so they carry largely
independent information. As required by the workflow, each predictor was then
fit in its \emph{own} simple linear regression rather than in a single multiple
regression.
\\

\begin{table}[H]
\centering
\caption{Simple Linear Regression Coefficients (statsmodels OLS)}
\label{tab:concrete_ols_coefficients}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrr}
\hline
\textbf{Model} & \textbf{Variable} & \textbf{Coefficient} & \textbf{Std. Error} &
\textbf{t} & \textbf{p-value} & \textbf{95\% CI} \\
\hline
@COEF1@\hline
@COEF2@\hline
\end{tabular}%
}
\end{table}

\begin{table}[H]
\centering
\caption{Simple Linear Regression Model Statistics}
\label{tab:concrete_ols_stats}
\begin{tabular}{lrr}
\hline
\textbf{Statistic} & \textbf{strength $\sim$ Cement} &
\textbf{strength $\sim$ Superplasticizer} \\
\hline
Observations        & @R1_N@ & @R2_N@ \\
$R^2$               & @R1_R2@ & @R2_R2@ \\
Adjusted $R^2$      & @R1_AR2@ & @R2_AR2@ \\
RMSE (MPa)          & @R1_RMSE@ & @R2_RMSE@ \\
F-statistic         & @R1_F@ & @R2_F@ \\
Prob. (F-statistic) & @R1_FP@ & @R2_FP@ \\
AIC                 & @R1_AIC@ & @R2_AIC@ \\
BIC                 & @R1_BIC@ & @R2_BIC@ \\
\hline
\end{tabular}
\end{table}

Both slopes are positive and highly significant ($p \ll 0.001$): a
$1\,\mathrm{kg/m^3}$ increase in cement is associated with about @R1_SLOPE@ MPa
more strength, and a $1\,\mathrm{kg/m^3}$ increase in superplasticizer with
about @R2_SLOPE@ MPa more strength. On its own, cement explains $R^2 = @R1_R2@$
of the variance in strength and superplasticizer $R^2 = @R2_R2@$, so
\textbf{@BEST@} is the stronger single predictor. Neither single-variable line
captures a large share of the variance, which is expected: compressive strength
is a nonlinear function of curing age and the full set of ingredients, and a
one-predictor straight line cannot represent that. The typical prediction error
(RMSE) is roughly @R1_RMSE1@--@R2_RMSE1@ MPa against a response that ranges
about @Y_MIN@--@Y_MAX@ MPa.
\\

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{y_vs_yhat_@P1@.png}
    \caption{y vs yhat ordered by Cement}
    \label{fig:concrete-y-vs-yhat-cement}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{y_vs_yhat_@P2@.png}
    \caption{y vs yhat ordered by Superplasticizer}
    \label{fig:concrete-y-vs-yhat-superplasticizer}
\end{figure}

Looking at these plots of $y$ vs $\hat{y}$, the fitted line follows the overall
upward trend for both predictors, but the actual strengths scatter widely
around it, consistent with the modest $R^2$ values above.
"""

    subs = {
        "@N_MISSING@": str(n_missing),
        "@N_DUP@": str(n_dup),
        "@N_ROWS@": str(n_rows),
        "@N_UNIQUE@": str(n_unique),
        "@EDA_FIGS@": eda_figs,
        "@SROWS@": srows,
        "@CROWS@": crows,
        "@R1_R@": f"{r1_r:+.3f}",
        "@R2_R@": f"{r2_r:+.3f}",
        "@P12@": f"{p12:+.2f}",
        "@COEF1@": coef_rows(r1, p1),
        "@COEF2@": coef_rows(r2, p2),
        "@R1_N@": str(r1["n"]), "@R2_N@": str(r2["n"]),
        "@R1_R2@": f"{r1['r_squared']:.3f}", "@R2_R2@": f"{r2['r_squared']:.3f}",
        "@R1_AR2@": f"{r1['adj_r_squared']:.3f}", "@R2_AR2@": f"{r2['adj_r_squared']:.3f}",
        "@R1_RMSE@": f"{r1['rmse']:.3f}", "@R2_RMSE@": f"{r2['rmse']:.3f}",
        "@R1_RMSE1@": f"{r1['rmse']:.1f}", "@R2_RMSE1@": f"{r2['rmse']:.1f}",
        "@R1_F@": f"{r1['fvalue']:.1f}", "@R2_F@": f"{r2['fvalue']:.1f}",
        "@R1_FP@": _sci(r1["f_p_value"]), "@R2_FP@": _sci(r2["f_p_value"]),
        "@R1_AIC@": f"{r1['aic']:.0f}", "@R2_AIC@": f"{r2['aic']:.0f}",
        "@R1_BIC@": f"{r1['bic']:.0f}", "@R2_BIC@": f"{r2['bic']:.0f}",
        "@R1_SLOPE@": f"{r1['slope']:.4f}", "@R2_SLOPE@": f"{r2['slope']:.3f}",
        "@BEST@": best,
        "@Y_MIN@": f"{df[RESPONSE].min():.0f}", "@Y_MAX@": f"{df[RESPONSE].max():.0f}",
        "@P1@": p1, "@P2@": p2,
    }
    tex = template
    for k, v in subs.items():
        tex = tex.replace(k, v)

    (REPORTFIG / "concrete_section.tex").write_text(tex, encoding="utf-8")
    print(rule("WROTE output/report_figures/concrete_section.tex  (+ matching PNGs)"))


# --------------------------------------------------------------------------- #
def main() -> None:
    xls_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLS
    if not xls_path.exists():
        sys.exit(f"Cannot find dataset: {xls_path}")
    print(f"Reading {xls_path}")
    df, raw = load_data(xls_path)

    clean = step1_prepare(df, raw)
    make_eda_plots(clean)
    summary = step2_summarize(clean)
    top2 = step3_explore(clean)
    reg, results = steps45_model(clean, top2)
    write_results_md(summary, top2, reg)
    write_latex_section(clean, summary, top2, results)

    print(rule("DONE"))
    print("All outputs written to:", OUT)
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
