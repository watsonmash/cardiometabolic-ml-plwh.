"""Small-sample machine learning on a cardiometabolic HIV cohort, done carefully.

Three analyses:
  A. Hypertension prediction: does adding glycaemic markers (HbA1c, GA%) and the
     Atherogenic Index of Plasma improve discrimination beyond age, sex and BMI?
     Logistic regression vs penalised logistic vs random forest vs XGBoost,
     evaluated by repeated stratified cross-validation (AUC, Brier, calibration).
  B. Marker discordance: which characteristics explain the signed difference between
     standardised GA% and HbA1c? Linear vs ridge vs XGBoost, with a permutation null.
  C. Exploratory clustering of cardiometabolic phenotypes with bootstrap stability.

Usage:
    python src/ml_analysis.py                          # synthetic data
    python src/ml_analysis.py --data path/to/real.xlsx # never commit real data
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import adjusted_rand_score, brier_score_loss, r2_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import derive, load

warnings.filterwarnings("ignore")
SEED = 2026
INK, MUTED, GRID = "#1a1a19", "#5f5f5c", "#d8d8d4"
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": GRID,
                     "font.size": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})


def save(fig, name, out, synthetic):
    if synthetic:
        fig.text(0.99, 0.005, "SYNTHETIC DATA – illustrative only", ha="right", va="bottom",
                 fontsize=8, color="#C0392B")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------------ A. hypertension
FEATURE_SETS = {
    "Clinical (age, sex, BMI)": ["age", "male", "bmi"],
    "+ HbA1c, GA%": ["age", "male", "bmi", "hba1c", "ga_pct"],
    "+ HbA1c, GA%, AIP": ["age", "male", "bmi", "hba1c", "ga_pct", "aip"],
}
CLASSIFIERS = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)),
    "Penalised logistic": make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000)),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=8, n_jobs=-1,
                                            max_features="sqrt", random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=5,
                             eval_metric="logloss", random_state=SEED, verbosity=0),
}


def calib_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]


def hypertension(d, out, synthetic, n_rep=10):
    y = d["htn"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    splits = list(cv.split(d, y))
    rows, oof_store, fold_auc = [], {}, {}
    for fs_name, cols in FEATURE_SETS.items():
        X = d[cols].values
        for m_name, model in CLASSIFIERS.items():
            aucs, briers = [], []
            oof = np.zeros((n_rep, len(y)))
            for k, (tr, te) in enumerate(splits):
                m = clone(model).fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                oof[k // 5, te] = p
                aucs.append(roc_auc_score(y[te], p))
                briers.append(brier_score_loss(y[te], p))
            slopes = [calib_slope(y, oof[r]) for r in range(n_rep)]
            oof_store[(fs_name, m_name)] = oof
            fold_auc[(fs_name, m_name)] = np.array(aucs)
            rows.append({"Feature set": fs_name, "Model": m_name,
                         "AUC mean": np.mean(aucs), "AUC 2.5%": np.percentile(aucs, 2.5),
                         "AUC 97.5%": np.percentile(aucs, 97.5),
                         "Brier": np.mean(briers), "Calibration slope": np.median(slopes)})
    res = pd.DataFrame(rows)

    # paired fold-level comparison: does adding markers improve AUC over clinical baseline?
    base = "Clinical (age, sex, BMI)"
    inc = []
    for fs_name in list(FEATURE_SETS)[1:]:
        for m_name in CLASSIFIERS:
            diff = fold_auc[(fs_name, m_name)] - fold_auc[(base, m_name)]
            inc.append({"Comparison": f"{fs_name} vs clinical", "Model": m_name,
                        "Mean ΔAUC": diff.mean(), "ΔAUC 2.5%": np.percentile(diff, 2.5),
                        "ΔAUC 97.5%": np.percentile(diff, 97.5),
                        "Folds improved, %": (diff > 0).mean() * 100})
    inc = pd.DataFrame(inc)

    # figure: AUC by model and feature set
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    models = list(CLASSIFIERS)
    for i, fs_name in enumerate(FEATURE_SETS):
        sub = res[res["Feature set"] == fs_name].set_index("Model").loc[models]
        xs = np.arange(len(models)) + (i - 1) * 0.22
        ax.errorbar(xs, sub["AUC mean"], yerr=[sub["AUC mean"] - sub["AUC 2.5%"],
                                               sub["AUC 97.5%"] - sub["AUC mean"]],
                    fmt="o", color=PAL[i], capsize=3, label=fs_name, ms=6)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("Cross-validated AUC (fold 2.5–97.5%)")
    ax.set_title("A. Hypertension: discrimination by model and feature set")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "a1_auc_comparison", out, synthetic)

    # figure: calibration (out-of-fold, first repeat) for logistic vs XGBoost, full set
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    full = list(FEATURE_SETS)[-1]
    for j, m_name in enumerate(["Logistic", "XGBoost"]):
        p = oof_store[(full, m_name)].mean(axis=0)
        fx, fy = calibration_curve(y, p, n_bins=5, strategy="quantile")
        ax.plot(fy, fx, "o-", color=PAL[j], label=m_name)
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":")
    ax.set_xlabel("Predicted probability (out-of-fold)")
    ax.set_ylabel("Observed proportion")
    ax.set_title("A. Calibration, full feature set")
    ax.legend()
    save(fig, "a2_calibration", out, synthetic)

    # SHAP for XGBoost on full feature set (descriptive; fitted on all data)
    try:
        import shap
        cols = FEATURE_SETS[full]
        m = clone(CLASSIFIERS["XGBoost"]).fit(d[cols].values, y)
        sv = shap.TreeExplainer(m).shap_values(d[cols].values)
        plt.figure(figsize=(6, 3.6))
        shap.summary_plot(sv, d[cols], show=False, plot_size=None)
        fig = plt.gcf()
        fig.axes[0].set_title("A. SHAP values, XGBoost (all participants)", loc="left", fontweight="bold")
        save(fig, "a3_shap", out, synthetic)
        shap_imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    except Exception as e:  # shap is optional
        print("SHAP skipped:", e)
        shap_imp = pd.Series(dtype=float)

    # logistic coefficients on standardised features (full data) for comparison with SHAP
    cols = FEATURE_SETS[full]
    lg = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)).fit(d[cols], y)
    coefs = pd.Series(lg[-1].coef_[0], index=cols)
    return res, inc, shap_imp, coefs


# ------------------------------------------------------------------ B. discordance
def discordance(d, out, synthetic, n_perm=200):
    cols = ["albumin", "bmi", "age", "male", "aip"]
    X, y = d[cols].values, d["discordance"].values
    cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    splits = list(cv.split(X))
    regs = {"Linear": make_pipeline(StandardScaler(), LinearRegression()),
            "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                    min_child_weight=5, reg_lambda=5, random_state=SEED, verbosity=0)}

    def cv_r2(model, yv):
        pred = np.zeros((20, len(yv)))
        for k, (tr, te) in enumerate(splits):
            pred[k // 5, te] = clone(model).fit(X[tr], yv[tr]).predict(X[te])
        return np.array([r2_score(yv, pred[r]) for r in range(20)])

    rows = []
    rng = np.random.default_rng(SEED)
    for name, model in regs.items():
        r2 = cv_r2(model, y)
        # permutation null for the linear model only (cheap); compares observed CV R² to chance
        if name == "Linear":
            null = np.array([cv_r2(model, rng.permutation(y)).mean() for _ in range(n_perm)])
            p_perm = (np.sum(null >= r2.mean()) + 1) / (n_perm + 1)
        else:
            p_perm = np.nan
        rows.append({"Model": name, "CV R² mean": r2.mean(), "CV R² 2.5%": np.percentile(r2, 2.5),
                     "CV R² 97.5%": np.percentile(r2, 97.5), "Permutation p": p_perm})
    res = pd.DataFrame(rows)

    # bootstrap CIs for standardised linear coefficients
    Xs = StandardScaler().fit_transform(X)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        boots.append(LinearRegression().fit(Xs[i], y[i]).coef_)
    boots = np.array(boots)
    coef = pd.DataFrame({"Feature": cols, "Std. coefficient": LinearRegression().fit(Xs, y).coef_,
                         "2.5%": np.percentile(boots, 2.5, 0), "97.5%": np.percentile(boots, 97.5, 0)})

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    c = coef.sort_values("Std. coefficient")
    ax.errorbar(c["Std. coefficient"], range(len(c)),
                xerr=[c["Std. coefficient"] - c["2.5%"], c["97.5%"] - c["Std. coefficient"]],
                fmt="o", color=PAL[0], capsize=3)
    ax.axvline(0, color=MUTED, ls=":")
    ax.set_yticks(range(len(c)), c["Feature"])
    ax.set_xlabel("Standardised coefficient (bootstrap 95% CI)")
    ax.set_title("B. Correlates of GA%–HbA1c discordance")
    save(fig, "b1_discordance_coefs", out, synthetic)
    return res, coef


# ------------------------------------------------------------------ C. clustering
def clustering(d, out, synthetic, n_boot=200):
    cols = ["hba1c", "ga_pct", "aip", "bmi", "sbp"]
    # rank-based normal scores: k-means on raw z-scores simply isolates a handful of extreme
    # values (e.g. very high HbA1c/GA%) rather than finding phenotypes
    Xs = QuantileTransformer(n_quantiles=len(d), output_distribution="normal",
                             random_state=SEED).fit_transform(d[cols])
    ks = range(2, 7)
    sil = [silhouette_score(Xs, KMeans(k, n_init=50, random_state=SEED).fit_predict(Xs)) for k in ks]
    bic = [GaussianMixture(k, n_init=5, random_state=SEED).fit(Xs).bic(Xs) for k in ks]
    k_best = list(ks)[int(np.argmax(sil))]
    ref = KMeans(k_best, n_init=50, random_state=SEED).fit(Xs)
    labels = ref.labels_

    # stability: refit on bootstrap samples, compare labels on the resampled individuals
    rng = np.random.default_rng(SEED)
    ari = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Xs), len(Xs))
        lb = KMeans(k_best, n_init=10, random_state=int(rng.integers(1e9))).fit(Xs[i]).predict(Xs[i])
        ari.append(adjusted_rand_score(labels[i], lb))
    stability = {"k (max silhouette)": k_best, "silhouette": max(sil),
                 "bootstrap ARI median": float(np.median(ari)),
                 "bootstrap ARI 2.5%": float(np.percentile(ari, 2.5))}

    g = d.assign(cluster=labels + 1).groupby("cluster")
    prof = g[cols + ["age"]].median().join(g["htn"].mean().rename("htn (proportion)"))
    prof.insert(0, "n", pd.Series(labels + 1).value_counts().sort_index().values)

    pcs = PCA(2, random_state=SEED).fit(Xs)
    P = pcs.transform(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].plot(list(ks), sil, "o-", color=PAL[0])
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette")
    axes[0].set_title("C. Choosing k")
    for c in range(k_best):
        m = labels == c
        axes[1].scatter(P[m, 0], P[m, 1], s=22, color=PAL[c % 5], alpha=0.75, label=f"Cluster {c+1}")
    axes[1].set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.0f}%)")
    axes[1].set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.0f}%)")
    axes[1].set_title(f"C. k-means clusters (ARI stability {stability['bootstrap ARI median']:.2f})")
    axes[1].legend(fontsize=8)
    save(fig, "c1_clusters", out, synthetic)
    return pd.DataFrame({"k": list(ks), "silhouette": sil, "GMM BIC": bic}), stability, prof


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_study_data.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--exclude-qc", action="store_true", help="drop physiologically implausible records")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    synthetic = "synthetic" in Path(a.data).name.lower()

    d = derive(load(a.data)).dropna(subset=["hba1c", "ga_pct", "aip", "bmi", "albumin"])
    if a.exclude_qc:
        d = d[~d["qc_flag"]]
    d = d.reset_index(drop=True)
    print(f"n = {len(d)}, hypertension events = {int(d['htn'].sum())}, "
          f"QC-flagged retained = {int(d['qc_flag'].sum())}")

    resA, incA, shap_imp, coefs = hypertension(d, out, synthetic)
    resB, coefB = discordance(d, out, synthetic)
    kTab, stab, prof = clustering(d, out, synthetic)

    with pd.ExcelWriter(out / "ml_results.xlsx") as xw:
        resA.round(3).to_excel(xw, sheet_name="A_discrimination", index=False)
        incA.round(3).to_excel(xw, sheet_name="A_incremental_value", index=False)
        pd.DataFrame({"mean |SHAP|": shap_imp.round(3), "logistic std coef": coefs.round(3)}) \
            .to_excel(xw, sheet_name="A_importance")
        resB.round(3).to_excel(xw, sheet_name="B_discordance_cv", index=False)
        coefB.round(3).to_excel(xw, sheet_name="B_discordance_coefs", index=False)
        kTab.round(3).to_excel(xw, sheet_name="C_choose_k", index=False)
        pd.Series(stab).to_frame("value").to_excel(xw, sheet_name="C_stability")
        prof.round(2).to_excel(xw, sheet_name="C_profiles")

    pd.set_option("display.width", 200)
    print("\nA. Hypertension discrimination\n", resA.round(3).to_string(index=False))
    print("\nA. Incremental value over clinical model\n", incA.round(3).to_string(index=False))
    print("\nA. Importance\n", pd.DataFrame({"|SHAP|": shap_imp, "logit coef": coefs}).round(3))
    print("\nB. Discordance\n", resB.round(3).to_string(index=False))
    print(coefB.round(3).to_string(index=False))
    print("\nC. Clustering\n", kTab.round(3).to_string(index=False), "\n", stab, "\n", prof.round(2))


if __name__ == "__main__":
    main()
