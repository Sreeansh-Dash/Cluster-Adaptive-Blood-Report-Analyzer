"""
Generator for notebooks_v2/03b_Cluster_Verification.ipynb
"""
import json, pathlib

def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

cells.append(cell("markdown", [
    "# 03b — Cluster Verification\n",
    "\n",
    "Verify that clusters are stable and reflect real patient variation, not preprocessing artefacts.  \n",
    "All checks use **train data only**."
]))

cells.append(cell("code", [
    "import os, warnings, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import adjusted_rand_score\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "\n",
    "PROCESSED = '../data/processed'\n",
    "MODELS    = '../models'\n",
    "PLOTS     = '../results/plots'\n",
    "print('Ready.')"
]))

cells.append(cell("markdown", "## Load Data"))

cells.append(cell("code", [
    "train_scaled   = pd.read_csv(f'{PROCESSED}/train_wide_scaled.csv')\n",
    "train_unscaled = pd.read_csv(f'{PROCESSED}/train_wide_unscaled.csv')\n",
    "train_clusters = pd.read_csv(f'{PROCESSED}/train_clusters.csv')\n",
    "\n",
    "pca    = joblib.load(f'{MODELS}/pca.pkl')\n",
    "kmeans = joblib.load(f'{MODELS}/kmeans.pkl')\n",
    "best_k = kmeans.n_clusters\n",
    "\n",
    "EXCLUDE = ['document_id','age','gender']\n",
    "feat_cols = [c for c in train_scaled.columns\n",
    "             if c not in EXCLUDE and not c.endswith('_missing')]\n",
    "\n",
    "X_train = train_scaled[feat_cols].values\n",
    "X_pca   = pca.transform(X_train)\n",
    "ref_labels = train_clusters.set_index('document_id')['cluster_id'].reindex(train_scaled['document_id']).values\n",
    "\n",
    "print(f'k={best_k}, train patients={len(train_scaled):,}')"
]))

# Check 1 — Bootstrap stability
cells.append(cell("markdown", [
    "## Check 1 — Bootstrap Stability (ARI)\n",
    "\n",
    "Resample train with replacement 10 times, refit K-Means on each bootstrap sample,  \n",
    "then assign clusters to the **original** train patients and compute Adjusted Rand Index vs. the reference solution.  \n",
    "Mean ARI > 0.6 indicates reasonable stability."
]))

cells.append(cell("code", [
    "N_BOOT = 10\n",
    "ari_scores = []\n",
    "n = len(X_pca)\n",
    "\n",
    "for seed in range(N_BOOT):\n",
    "    rng = np.random.default_rng(seed)\n",
    "    boot_idx = rng.integers(0, n, size=n)\n",
    "    X_boot   = X_pca[boot_idx]\n",
    "\n",
    "    km_boot = KMeans(n_clusters=best_k, random_state=seed, n_init=10)\n",
    "    km_boot.fit(X_boot)\n",
    "\n",
    "    boot_labels = km_boot.predict(X_pca)  # assign original patients\n",
    "    ari = adjusted_rand_score(ref_labels, boot_labels)\n",
    "    ari_scores.append(ari)\n",
    "    print(f'Bootstrap {seed:02d}: ARI = {ari:.4f}')\n",
    "\n",
    "print(f'\\nMean ARI : {np.mean(ari_scores):.4f}')\n",
    "print(f'Std  ARI : {np.std(ari_scores):.4f}')\n",
    "if np.mean(ari_scores) > 0.6:\n",
    "    print('✅ Cluster solution is stable (mean ARI > 0.6).')\n",
    "else:\n",
    "    print('⚠️  Cluster solution may be unstable (mean ARI ≤ 0.6).')"
]))

cells.append(cell("code", [
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "ax.boxplot(ari_scores, vert=True, patch_artist=True,\n",
    "           boxprops=dict(facecolor='steelblue', alpha=0.6))\n",
    "ax.axhline(0.6, color='red', linestyle='--', label='ARI = 0.6 threshold')\n",
    "ax.set_title(f'Bootstrap ARI Distribution (k={best_k}, n=10 bootstraps)')\n",
    "ax.set_ylabel('Adjusted Rand Index')\n",
    "ax.set_xticks([])\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/bootstrap_ari.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Check 2 — Seed sensitivity
cells.append(cell("markdown", "## Check 2 — Random Seed Sensitivity"))

cells.append(cell("code", [
    "SEEDS = [0, 1, 7, 13, 99]\n",
    "seed_aris = []\n",
    "\n",
    "for s in SEEDS:\n",
    "    km_s = KMeans(n_clusters=best_k, random_state=s, n_init=20)\n",
    "    labels_s = km_s.fit_predict(X_pca)\n",
    "    ari = adjusted_rand_score(ref_labels, labels_s)\n",
    "    seed_aris.append(ari)\n",
    "    print(f'seed={s:3d}: ARI = {ari:.4f}')\n",
    "\n",
    "print(f'\\nSeed ARI range: {min(seed_aris):.4f} – {max(seed_aris):.4f}')\n",
    "if max(seed_aris) - min(seed_aris) < 0.1:\n",
    "    print('✅ Low seed sensitivity — solution is robust.')\n",
    "else:\n",
    "    print('⚠️  High seed sensitivity — solution may not be robust.')"
]))

# Check 3 — Missingness per cluster
cells.append(cell("markdown", [
    "## Check 3 — Missingness-Driven Cluster Check\n",
    "\n",
    "Uses **unscaled** (pre-imputation NaN pattern) train data."
]))

cells.append(cell("code", [
    "feat_cols_us = [c for c in train_unscaled.columns\n",
    "                if c not in ['document_id','age','gender'] and not c.endswith('_missing')]\n",
    "\n",
    "train_unscaled['cluster_id'] = ref_labels\n",
    "overall_miss = train_unscaled[feat_cols_us].isnull().mean(axis=1).mean() * 100\n",
    "\n",
    "cluster_miss = {}\n",
    "for c in sorted(train_unscaled['cluster_id'].unique()):\n",
    "    grp  = train_unscaled[train_unscaled['cluster_id']==c]\n",
    "    miss = grp[feat_cols_us].isnull().mean(axis=1).mean() * 100\n",
    "    cluster_miss[c] = miss\n",
    "    diff = miss - overall_miss\n",
    "    flag = '  ⚠️  WARNING: possibly missingness-driven!' if diff > 20 else ''\n",
    "    print(f'Cluster {c}: {miss:.2f}%  (Δ {diff:+.2f}pp){flag}')\n",
    "\n",
    "miss_df = pd.DataFrame({'cluster': list(cluster_miss.keys()),\n",
    "                         'missing_pct': list(cluster_miss.values())})\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "sns.barplot(x='cluster', y='missing_pct', data=miss_df, palette='Blues_r', ax=ax)\n",
    "ax.axhline(overall_miss, color='red', linestyle='--', label=f'Overall ({overall_miss:.1f}%)')\n",
    "ax.axhline(overall_miss+20, color='orange', linestyle=':', label='+20pp threshold')\n",
    "ax.set_title('Missingness per Cluster (unscaled train data)')\n",
    "ax.set_xlabel('Cluster'); ax.set_ylabel('Mean missing %')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/verified_cluster_missingness.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Check 4 — Feature importance
cells.append(cell("markdown", [
    "## Check 4 — Feature Importance in Clustering\n",
    "\n",
    "Fit a shallow Decision Tree to predict `cluster_id` from unscaled features.  \n",
    "Top features reveal whether clusters are driven by lab values (good) or demographics alone."
]))

cells.append(cell("code", [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "dt = DecisionTreeClassifier(max_depth=4, random_state=42)\n",
    "X_dt = train_unscaled[feat_cols_us].fillna(train_unscaled[feat_cols_us].median())\n",
    "dt.fit(X_dt, ref_labels)\n",
    "\n",
    "fi = pd.Series(dt.feature_importances_, index=feat_cols_us)\\\n",
    "       .sort_values(ascending=False)\n",
    "\n",
    "print('Top 10 cluster-driving features:')\n",
    "print(fi.head(10).to_string())\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "sns.barplot(x=fi.head(10).values, y=fi.head(10).index, palette='viridis', ax=ax)\n",
    "ax.set_title('Top 10 Features Driving Cluster Separation')\n",
    "ax.set_xlabel('Feature Importance')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/cluster_feature_importance.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Check 5 — Summary
cells.append(cell("markdown", "## Check 5 — Summary"))

cells.append(cell("code", [
    "any_miss_warn = any(v - overall_miss > 20 for v in cluster_miss.values())\n",
    "top3_feats = fi.head(3).index.tolist()\n",
    "\n",
    "print('='*55)\n",
    "print('  CLUSTER VERIFICATION SUMMARY')\n",
    "print('='*55)\n",
    "print(f'  k (clusters)         : {best_k}')\n",
    "print(f'  Bootstrap mean ARI   : {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}')\n",
    "print(f'  Seed ARI range       : {min(seed_aris):.4f} – {max(seed_aris):.4f}')\n",
    "print(f'  Missingness warning  : {\"YES ⚠️\" if any_miss_warn else \"No ✅\"}')\n",
    "print(f'  Top cluster features : {top3_feats}')\n",
    "print('='*55)\n",
    "print('This summary will be quoted in the paper methods section.')"
]))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}
out = pathlib.Path(__file__).parent / '03b_Cluster_Verification.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
