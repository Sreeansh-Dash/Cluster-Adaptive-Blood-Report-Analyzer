"""
Generator for notebooks_v2/03_Clustering.ipynb
"""
import json, pathlib

def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

cells.append(cell("markdown", [
    "# 03 — Patient Clustering (Train-Only)\n",
    "\n",
    "**PCA** and **K-Means** are fit on **train data only**.  \n",
    "Test patients receive cluster assignments via `predict()` — no refitting."
]))

cells.append(cell("code", [
    "import os, warnings, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.manifold import TSNE\n",
    "from sklearn.metrics import silhouette_score, davies_bouldin_score\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "\n",
    "PROCESSED = '../data/processed'\n",
    "MODELS    = '../models'\n",
    "PLOTS     = '../results/plots'\n",
    "for d in [MODELS, PLOTS]:\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "print('Ready.')"
]))

# Step 1 — Load
cells.append(cell("markdown", "## Step 1 — Load Scaled Data"))

cells.append(cell("code", [
    "train_wide = pd.read_csv(f'{PROCESSED}/train_wide_scaled.csv')\n",
    "test_wide  = pd.read_csv(f'{PROCESSED}/test_wide_scaled.csv')\n",
    "print(f'Train: {train_wide.shape}  |  Test: {test_wide.shape}')\n",
    "\n",
    "# Identify numeric feature columns (exclude id, age, gender, _missing indicators)\n",
    "EXCLUDE = ['document_id', 'age', 'gender']\n",
    "feat_cols = [c for c in train_wide.columns\n",
    "             if c not in EXCLUDE and not c.endswith('_missing')]\n",
    "print(f'Feature columns for clustering: {len(feat_cols)}')\n",
    "\n",
    "X_train = train_wide[feat_cols].values\n",
    "X_test  = test_wide[feat_cols].values"
]))

# Step 2 — PCA
cells.append(cell("markdown", [
    "## Step 2 — PCA (Fit on Train Only)\n",
    "\n",
    "`n_components=0.95` retains 95% of explained variance."
]))

cells.append(cell("code", [
    "pca = PCA(n_components=0.95, random_state=42)\n",
    "X_train_pca = pca.fit_transform(X_train)\n",
    "X_test_pca  = pca.transform(X_test)\n",
    "\n",
    "joblib.dump(pca, f'{MODELS}/pca.pkl')\n",
    "print(f'PCA components kept: {pca.n_components_}')\n",
    "print(f'Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%')\n",
    "print('PCA saved → models/pca.pkl')"
]))

cells.append(cell("code", [
    "# Scree plot\n",
    "cum_var = np.cumsum(pca.explained_variance_ratio_)\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "ax.plot(cum_var, marker='o', markersize=3, color='steelblue')\n",
    "ax.axhline(0.95, color='red', linestyle='--', label='95% threshold')\n",
    "ax.set_title('PCA Explained Variance (cumulative)')\n",
    "ax.set_xlabel('Number of components'); ax.set_ylabel('Cumulative variance')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/pca_scree.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Step 3 — K-Means (elbow + silhouette)
cells.append(cell("markdown", [
    "## Step 3 — K-Means Clustering (Elbow + Silhouette)\n",
    "\n",
    "Fitted on **train PCA output** only. k tried: 3–8."
]))

cells.append(cell("code", [
    "inertias, sil_scores, db_scores = [], [], []\n",
    "K_RANGE = range(3, 9)\n",
    "\n",
    "for k in K_RANGE:\n",
    "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "    labels = km.fit_predict(X_train_pca)\n",
    "    inertias.append(km.inertia_)\n",
    "    sil = silhouette_score(X_train_pca, labels, sample_size=10000, random_state=42)\n",
    "    db  = davies_bouldin_score(X_train_pca, labels)\n",
    "    sil_scores.append(sil)\n",
    "    db_scores.append(db)\n",
    "    print(f'k={k}: inertia={km.inertia_:.0f}  silhouette={sil:.4f}  DB={db:.4f}')"
]))

cells.append(cell("code", [
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
    "\n",
    "axes[0].plot(list(K_RANGE), inertias, marker='o', color='steelblue')\n",
    "axes[0].set_title('Elbow Method'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')\n",
    "\n",
    "axes[1].plot(list(K_RANGE), sil_scores, marker='o', color='green')\n",
    "axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('k'); axes[1].set_ylabel('Score')\n",
    "\n",
    "axes[2].plot(list(K_RANGE), db_scores, marker='o', color='red')\n",
    "axes[2].set_title('Davies-Bouldin Index (lower=better)')\n",
    "axes[2].set_xlabel('k'); axes[2].set_ylabel('DB Index')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/clustering_selection.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Auto-select k: best silhouette score\n",
    "best_k = list(K_RANGE)[sil_scores.index(max(sil_scores))]\n",
    "print(f'\\nSelected k = {best_k} (highest silhouette score = {max(sil_scores):.4f})')"
]))

cells.append(cell("code", [
    "# Fit final model on train only\n",
    "kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)\n",
    "train_labels = kmeans.fit_predict(X_train_pca)\n",
    "test_labels  = kmeans.predict(X_test_pca)\n",
    "\n",
    "joblib.dump(kmeans, f'{MODELS}/kmeans.pkl')\n",
    "print(f'KMeans (k={best_k}) saved → models/kmeans.pkl')\n",
    "print(f'Train cluster distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}')\n",
    "print(f'Test  cluster distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}')"
]))

cells.append(cell("code", [
    "# Save cluster assignments\n",
    "pd.DataFrame({'document_id': train_wide['document_id'], 'cluster_id': train_labels})\\\n",
    "  .to_csv(f'{PROCESSED}/train_clusters.csv', index=False)\n",
    "pd.DataFrame({'document_id': test_wide['document_id'],  'cluster_id': test_labels})\\\n",
    "  .to_csv(f'{PROCESSED}/test_clusters.csv',  index=False)\n",
    "print('Cluster assignments saved.')"
]))

# Step 4 — tSNE
cells.append(cell("markdown", [
    "## Step 4 — tSNE Visualization (3000 Train Patients)\n",
    "\n",
    "Three plots: by **cluster_id**, by **age_group**, and by **gender**."
]))

cells.append(cell("code", [
    "TSNE_SAMPLE = 3000\n",
    "np.random.seed(42)\n",
    "idx = np.random.choice(len(X_train_pca), TSNE_SAMPLE, replace=False)\n",
    "X_tsne_in = X_train_pca[idx]\n",
    "\n",
    "print(f'Running tSNE on {TSNE_SAMPLE} train patients...')\n",
    "tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)\n",
    "X_2d = tsne.fit_transform(X_tsne_in)\n",
    "\n",
    "tsne_df = train_wide.iloc[idx][['document_id','age','gender']].copy()\n",
    "tsne_df['x']  = X_2d[:, 0]\n",
    "tsne_df['y']  = X_2d[:, 1]\n",
    "tsne_df['cluster'] = train_labels[idx]\n",
    "\n",
    "# Age group\n",
    "bins  = [0, 18, 35, 50, 65, 200]\n",
    "lbls  = ['<18','18-34','35-49','50-64','65+']\n",
    "tsne_df['age_group'] = pd.cut(pd.to_numeric(tsne_df['age'], errors='coerce'),\n",
    "                              bins=bins, labels=lbls, right=False)\n",
    "print('tSNE done.')"
]))

cells.append(cell("code", [
    "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
    "\n",
    "# Plot 1: by cluster\n",
    "palette = sns.color_palette('tab10', n_colors=best_k)\n",
    "for c in sorted(tsne_df['cluster'].unique()):\n",
    "    mask = tsne_df['cluster'] == c\n",
    "    axes[0].scatter(tsne_df[mask]['x'], tsne_df[mask]['y'],\n",
    "                    label=f'Cluster {c}', s=8, alpha=0.6, color=palette[c])\n",
    "axes[0].set_title('tSNE — by Cluster')\n",
    "axes[0].legend(markerscale=3, fontsize=9)\n",
    "\n",
    "# Plot 2: by age group\n",
    "age_palette = sns.color_palette('viridis', n_colors=len(lbls))\n",
    "for i, ag in enumerate(lbls):\n",
    "    mask = tsne_df['age_group'] == ag\n",
    "    axes[1].scatter(tsne_df[mask]['x'], tsne_df[mask]['y'],\n",
    "                    label=ag, s=8, alpha=0.6, color=age_palette[i])\n",
    "axes[1].set_title('tSNE — by Age Group')\n",
    "axes[1].legend(markerscale=3, fontsize=9)\n",
    "\n",
    "# Plot 3: by gender\n",
    "for g, col in [('male','steelblue'),('female','coral')]:\n",
    "    mask = tsne_df['gender'].str.lower() == g\n",
    "    axes[2].scatter(tsne_df[mask]['x'], tsne_df[mask]['y'],\n",
    "                    label=g, s=8, alpha=0.5, color=col)\n",
    "axes[2].set_title('tSNE — by Gender')\n",
    "axes[2].legend(markerscale=3, fontsize=9)\n",
    "\n",
    "for ax in axes:\n",
    "    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/tsne_clusters.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Step 5 — Cluster profiling
cells.append(cell("markdown", "## Step 5 — Cluster Profiling"))

cells.append(cell("code", [
    "# Load unscaled train for readable profiling\n",
    "train_unscaled = pd.read_csv(f'{PROCESSED}/train_wide_unscaled.csv')\n",
    "train_unscaled['cluster_id'] = train_labels\n",
    "\n",
    "feat_cols_us = [c for c in train_unscaled.columns\n",
    "                if c not in ['document_id','age','gender'] and not c.endswith('_missing')]\n",
    "\n",
    "# Top 10 most-covered tests\n",
    "coverage = train_unscaled[feat_cols_us].notna().mean().sort_values(ascending=False)\n",
    "top10 = coverage.head(10).index.tolist()\n",
    "\n",
    "print('=== Cluster Profiles ===')\n",
    "for c in sorted(train_unscaled['cluster_id'].unique()):\n",
    "    grp = train_unscaled[train_unscaled['cluster_id']==c]\n",
    "    print(f'\\n--- Cluster {c} (n={len(grp):,}) ---')\n",
    "    print(f'  Mean age : {grp[\"age\"].mean():.1f}')\n",
    "    print(f'  Gender   : {grp[\"gender\"].value_counts(normalize=True).round(3).to_dict()}')\n",
    "    print(f'  Top 10 lab means:')\n",
    "    for t in top10:\n",
    "        print(f'    {t:<40} {grp[t].mean():.2f}')"
]))

# Step 6 — Missingness check
cells.append(cell("markdown", [
    "## Step 6 — Missingness Check per Cluster\n",
    "\n",
    "A cluster with dramatically higher missingness than others may be a **data-sparsity artefact**, not a real patient group."
]))

cells.append(cell("code", [
    "overall_miss = train_unscaled[feat_cols_us].isnull().mean(axis=1).mean() * 100\n",
    "print(f'Overall mean missingness per patient: {overall_miss:.2f}%\\n')\n",
    "\n",
    "cluster_miss = []\n",
    "for c in sorted(train_unscaled['cluster_id'].unique()):\n",
    "    grp  = train_unscaled[train_unscaled['cluster_id']==c]\n",
    "    miss = grp[feat_cols_us].isnull().mean(axis=1).mean() * 100\n",
    "    cluster_miss.append({'cluster': c, 'mean_missing_pct': round(miss, 2)})\n",
    "    diff = miss - overall_miss\n",
    "    flag = '  ⚠️  WARNING: possibly missingness-driven!' if diff > 15 else ''\n",
    "    print(f'Cluster {c}: {miss:.2f}% missing  (Δ {diff:+.2f}pp){flag}')\n",
    "\n",
    "miss_df = pd.DataFrame(cluster_miss)\n",
    "fig, ax = plt.subplots(figsize=(8, 4))\n",
    "sns.barplot(x='cluster', y='mean_missing_pct', data=miss_df, palette='Blues_r', ax=ax)\n",
    "ax.axhline(overall_miss, color='red', linestyle='--', label=f'Overall mean ({overall_miss:.1f}%)')\n",
    "ax.axhline(overall_miss + 15, color='orange', linestyle=':', label='+15pp threshold')\n",
    "ax.set_title('Mean Missing % per Patient by Cluster')\n",
    "ax.set_xlabel('Cluster'); ax.set_ylabel('Mean missing %')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/cluster_missingness.png', bbox_inches='tight')\n",
    "plt.show()"
]))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}
out = pathlib.Path(__file__).parent / '03_Clustering.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
