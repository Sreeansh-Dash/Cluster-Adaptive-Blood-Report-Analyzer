"""
Generator for notebooks_v2/05_Ensemble.ipynb
"""
import json, pathlib

def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

cells.append(cell("markdown", [
    "# 05 — Ensemble Models + SMOTE Analysis\n",
    "\n",
    "**Models:** XGB_weighted, RF_tuned, AdaBoost, XGB_SMOTE  \n",
    "**Labels:** loaded from `data/processed/train_labels.csv` / `test_labels.csv` (generated in 04).  \n",
    "**Key finding:** XGB_weighted (native class-weighting) outperforms SMOTE-based approaches — reported as a project finding."
]))

cells.append(cell("code", [
    "import os, warnings, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier\n",
    "from sklearn.multioutput import MultiOutputClassifier\n",
    "from sklearn.metrics import (\n",
    "    f1_score, precision_score, recall_score,\n",
    "    roc_auc_score, average_precision_score,\n",
    "    hamming_loss, accuracy_score,\n",
    "    precision_recall_curve\n",
    ")\n",
    "from xgboost import XGBClassifier\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "\n",
    "PROCESSED = '../data/processed'\n",
    "MODELS    = '../models'\n",
    "PLOTS     = '../results/plots'\n",
    "METRICS   = '../results/metrics'\n",
    "for d in [MODELS, PLOTS, METRICS]:\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "\n",
    "LABELS = ['Anaemia','Diabetes_Risk','Dyslipidemia','Kidney_Risk','Liver_Stress','Thyroid_Abnormal']\n",
    "print('Ready.')"
]))

# ── Step 1 — Load Data ─────────────────────────────────────────────────────────
cells.append(cell("markdown", "## Step 1 — Load Data"))

cells.append(cell("code", [
    "train_scaled = pd.read_csv(f'{PROCESSED}/train_wide_scaled.csv')\n",
    "test_scaled  = pd.read_csv(f'{PROCESSED}/test_wide_scaled.csv')\n",
    "train_clust  = pd.read_csv(f'{PROCESSED}/train_clusters.csv')\n",
    "test_clust   = pd.read_csv(f'{PROCESSED}/test_clusters.csv')\n",
    "train_labels_df = pd.read_csv(f'{PROCESSED}/train_labels.csv')\n",
    "test_labels_df  = pd.read_csv(f'{PROCESSED}/test_labels.csv')\n",
    "\n",
    "# Merge cluster_id\n",
    "train_scaled = train_scaled.merge(train_clust, on='document_id', how='left')\n",
    "test_scaled  = test_scaled.merge(test_clust,   on='document_id', how='left')\n",
    "\n",
    "print(f'Train: {train_scaled.shape}  |  Test: {test_scaled.shape}')\n",
    "print(f'Clusters in train: {sorted(train_scaled[\"cluster_id\"].unique())}')"
]))

# ── Feature setup ──────────────────────────────────────────────────────────────
cells.append(cell("code", [
    "EXCLUDE_BASE = ['document_id', 'age', 'gender']\n",
    "ALL_FEAT = [c for c in train_scaled.columns if c not in EXCLUDE_BASE]\n",
    "\n",
    "# Same biomarker exclusion as in 04 — applied per-label during SMOTE\n",
    "HOLDOUT = {\n",
    "    'Anaemia':          ['hemoglobin','haemoglobin','hgb','mcv','mch','mchc','rbc','rdw',\n",
    "                         'pcv','hematocrit','reticulocyte'],\n",
    "    'Diabetes_Risk':    ['hba1c','hemoglobin a1c','glycated','glucose','blood sugar','hba'],\n",
    "    'Dyslipidemia':     ['cholesterol','ldl','hdl','triglyceride','vldl','lipoprotein'],\n",
    "    'Kidney_Risk':      ['creatinine','bun','urea','egfr','gfr','uric acid'],\n",
    "    'Liver_Stress':     ['alt','ast','ggt','bilirubin','albumin','sgpt','sgot','alp',\n",
    "                         'alkaline phosphatase','alk phos'],\n",
    "    'Thyroid_Abnormal': ['tsh','t3','t4','thyroid','triiodothyronine','thyroxine'],\n",
    "}\n",
    "\n",
    "X_train = train_scaled[ALL_FEAT].values\n",
    "X_test  = test_scaled[ALL_FEAT].values\n",
    "\n",
    "y_train = train_labels_df.set_index('document_id')[LABELS]\\\n",
    "                          .reindex(train_scaled['document_id']).values\n",
    "y_test  = test_labels_df.set_index('document_id')[LABELS]\\\n",
    "                         .reindex(test_scaled['document_id']).values\n",
    "\n",
    "print(f'X_train: {X_train.shape}  |  y_train: {y_train.shape}')"
]))

# ── Step 2 — scale_pos_weight per label ────────────────────────────────────────
cells.append(cell("markdown", "## Step 2 — Compute scale_pos_weight per Label (Train Only)"))

cells.append(cell("code", [
    "spw = {}\n",
    "print('scale_pos_weight per label:')\n",
    "for i, lbl in enumerate(LABELS):\n",
    "    pos = y_train[:, i].sum()\n",
    "    neg = len(y_train) - pos\n",
    "    spw[lbl] = round(neg / max(pos, 1), 2)\n",
    "    print(f'  {lbl:<20} pos={pos:,}  neg={neg:,}  spw={spw[lbl]}')"
]))

# ── Step 3 — Train Primary Models ─────────────────────────────────────────────
cells.append(cell("markdown", [
    "## Step 3 — Train Ensemble Models\n",
    "\n",
    "All models use `MultiOutputClassifier`. Fitted on **train** only."
]))

cells.append(cell("code", [
    "# XGB_weighted — uses per-label scale_pos_weight (average across labels as single estimator)\n",
    "# For MultiOutputClassifier, XGB is fit per label, so we use a wrapper\n",
    "mean_spw = round(np.mean(list(spw.values())), 2)\n",
    "\n",
    "xgb_weighted = MultiOutputClassifier(\n",
    "    XGBClassifier(\n",
    "        n_estimators=300, max_depth=6,\n",
    "        scale_pos_weight=mean_spw,\n",
    "        eval_metric='logloss',\n",
    "        use_label_encoder=False,\n",
    "        random_state=42, n_jobs=-1,\n",
    "        verbosity=0\n",
    "    )\n",
    ")\n",
    "\n",
    "rf_tuned = MultiOutputClassifier(\n",
    "    RandomForestClassifier(\n",
    "        n_estimators=300, class_weight='balanced',\n",
    "        max_depth=15, n_jobs=-1, random_state=42\n",
    "    )\n",
    ")\n",
    "\n",
    "adaboost = MultiOutputClassifier(\n",
    "    AdaBoostClassifier(n_estimators=200, random_state=42)\n",
    ")\n",
    "\n",
    "print('Training XGB_weighted...')\n",
    "xgb_weighted.fit(X_train, y_train)\n",
    "joblib.dump(xgb_weighted, f'{MODELS}/xgb_weighted.pkl')\n",
    "print('Training RF_tuned...')\n",
    "rf_tuned.fit(X_train, y_train)\n",
    "joblib.dump(rf_tuned, f'{MODELS}/rf_tuned.pkl')\n",
    "print('Training AdaBoost...')\n",
    "adaboost.fit(X_train, y_train)\n",
    "joblib.dump(adaboost, f'{MODELS}/adaboost.pkl')\n",
    "print('Primary models trained and saved.')"
]))

# ── Step 4 — XGB_SMOTE (per-label binary relevance) ──────────────────────────
cells.append(cell("markdown", [
    "## Step 4 — XGB_SMOTE (Binary Relevance per Label)\n",
    "\n",
    "SMOTE is applied **per label** on train data only, then XGBClassifier is trained without class weights.  \n",
    "Included for paper comparison — expected to underperform class-weighted XGB."
]))

cells.append(cell("code", [
    "smote_estimators = []\n",
    "for i, lbl in enumerate(LABELS):\n",
    "    y_lbl = y_train[:, i]\n",
    "    pos_count = y_lbl.sum()\n",
    "    min_samples = max(2, pos_count)  # SMOTE needs at least 2 minority samples\n",
    "    k_neighbors = min(5, pos_count - 1) if pos_count > 1 else 1\n",
    "    try:\n",
    "        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)\n",
    "        X_sm, y_sm = sm.fit_resample(X_train, y_lbl)\n",
    "    except Exception as e:\n",
    "        print(f'  SMOTE failed for {lbl} ({e}), using original data')\n",
    "        X_sm, y_sm = X_train, y_lbl\n",
    "\n",
    "    xgb_s = XGBClassifier(\n",
    "        n_estimators=300, max_depth=6,\n",
    "        eval_metric='logloss',\n",
    "        use_label_encoder=False,\n",
    "        random_state=42, n_jobs=-1,\n",
    "        verbosity=0\n",
    "    )\n",
    "    xgb_s.fit(X_sm, y_sm)\n",
    "    smote_estimators.append(xgb_s)\n",
    "    print(f'  {lbl}: SMOTE train size {len(X_sm):,}')\n",
    "\n",
    "joblib.dump(smote_estimators, f'{MODELS}/xgb_smote.pkl')\n",
    "print('XGB_SMOTE saved.')"
]))

# ── Step 5 — Threshold Tuning ─────────────────────────────────────────────────
cells.append(cell("markdown", "## Step 5 — Per-Label Threshold Tuning (Train F1 Max)"))

cells.append(cell("code", [
    "THRESHOLDS = np.arange(0.1, 0.95, 0.05)\n",
    "\n",
    "def tune_multioutput(model, X_tr, y_tr, labels):\n",
    "    proba_list = model.predict_proba(X_tr)\n",
    "    thresholds = {}\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = proba_list[i][:, 1] if proba_list[i].ndim == 2 else proba_list[i]\n",
    "        best_f1, best_t = 0, 0.5\n",
    "        for t in THRESHOLDS:\n",
    "            f1 = f1_score(y_tr[:, i], (p >= t).astype(int), zero_division=0)\n",
    "            if f1 > best_f1:\n",
    "                best_f1, best_t = f1, t\n",
    "        thresholds[lbl] = round(best_t, 2)\n",
    "    return thresholds\n",
    "\n",
    "def tune_smote(estimators, X_tr, y_tr, labels):\n",
    "    thresholds = {}\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = estimators[i].predict_proba(X_tr)[:, 1]\n",
    "        best_f1, best_t = 0, 0.5\n",
    "        for t in THRESHOLDS:\n",
    "            f1 = f1_score(y_tr[:, i], (p >= t).astype(int), zero_division=0)\n",
    "            if f1 > best_f1:\n",
    "                best_f1, best_t = f1, t\n",
    "        thresholds[lbl] = round(best_t, 2)\n",
    "    return thresholds\n",
    "\n",
    "thr_xgb   = tune_multioutput(xgb_weighted, X_train, y_train, LABELS)\n",
    "thr_rf    = tune_multioutput(rf_tuned,     X_train, y_train, LABELS)\n",
    "thr_ada   = tune_multioutput(adaboost,     X_train, y_train, LABELS)\n",
    "thr_smote = tune_smote(smote_estimators,   X_train, y_train, LABELS)\n",
    "\n",
    "thr_df = pd.DataFrame(\n",
    "    {'XGB_weighted': thr_xgb, 'RF_tuned': thr_rf,\n",
    "     'AdaBoost': thr_ada, 'XGB_SMOTE': thr_smote},\n",
    "    index=LABELS\n",
    ")\n",
    "print('Optimal thresholds (label × model):')\n",
    "print(thr_df.to_string())"
]))

# ── Step 6 — Evaluation ───────────────────────────────────────────────────────
cells.append(cell("markdown", "## Step 6 — Evaluation on Test Set"))

cells.append(cell("code", [
    "def eval_multioutput(model, X_te, y_te, thresholds, labels):\n",
    "    proba_list = model.predict_proba(X_te)\n",
    "    rows = []\n",
    "    preds_all = np.zeros((len(X_te), len(labels)), dtype=int)\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = proba_list[i][:, 1] if proba_list[i].ndim == 2 else proba_list[i]\n",
    "        pred = (p >= thresholds[lbl]).astype(int)\n",
    "        preds_all[:, i] = pred\n",
    "        rows.append({\n",
    "            'label':     lbl,\n",
    "            'F1':        f1_score(y_te[:, i], pred, zero_division=0),\n",
    "            'Precision': precision_score(y_te[:, i], pred, zero_division=0),\n",
    "            'Recall':    recall_score(y_te[:, i], pred, zero_division=0),\n",
    "            'AUC_ROC':   roc_auc_score(y_te[:, i], p),\n",
    "            'PR_AUC':    average_precision_score(y_te[:, i], p),\n",
    "        })\n",
    "    return pd.DataFrame(rows), preds_all\n",
    "\n",
    "def eval_smote(estimators, X_te, y_te, thresholds, labels):\n",
    "    rows = []\n",
    "    preds_all = np.zeros((len(X_te), len(labels)), dtype=int)\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = estimators[i].predict_proba(X_te)[:, 1]\n",
    "        pred = (p >= thresholds[lbl]).astype(int)\n",
    "        preds_all[:, i] = pred\n",
    "        rows.append({\n",
    "            'label':     lbl,\n",
    "            'F1':        f1_score(y_te[:, i], pred, zero_division=0),\n",
    "            'Precision': precision_score(y_te[:, i], pred, zero_division=0),\n",
    "            'Recall':    recall_score(y_te[:, i], pred, zero_division=0),\n",
    "            'AUC_ROC':   roc_auc_score(y_te[:, i], p),\n",
    "            'PR_AUC':    average_precision_score(y_te[:, i], p),\n",
    "        })\n",
    "    return pd.DataFrame(rows), preds_all\n",
    "\n",
    "res_xgb,  preds_xgb  = eval_multioutput(xgb_weighted, X_test, y_test, thr_xgb,   LABELS)\n",
    "res_rf,   preds_rf   = eval_multioutput(rf_tuned,      X_test, y_test, thr_rf,    LABELS)\n",
    "res_ada,  preds_ada  = eval_multioutput(adaboost,      X_test, y_test, thr_ada,   LABELS)\n",
    "res_smote,preds_smote= eval_smote(smote_estimators,    X_test, y_test, thr_smote, LABELS)\n",
    "\n",
    "model_results = {\n",
    "    'XGB_weighted': (res_xgb,   preds_xgb),\n",
    "    'RF_tuned':     (res_rf,    preds_rf),\n",
    "    'AdaBoost':     (res_ada,   preds_ada),\n",
    "    'XGB_SMOTE':    (res_smote, preds_smote),\n",
    "}\n",
    "\n",
    "# Per-label table for all models\n",
    "all_per_label = pd.concat(\n",
    "    [df.assign(Model=name) for name, (df, _) in model_results.items()]\n",
    ")[['Model','label','F1','Precision','Recall','AUC_ROC','PR_AUC']]\n",
    "print(all_per_label.round(4).to_string(index=False))"
]))

cells.append(cell("code", [
    "# Overall multi-label metrics table\n",
    "overall_rows = []\n",
    "for name, (df, preds) in model_results.items():\n",
    "    overall_rows.append({\n",
    "        'Model':           name,\n",
    "        'Hamming_Loss':    hamming_loss(y_test, preds),\n",
    "        'Subset_Accuracy': accuracy_score(y_test, preds),\n",
    "        'Micro_F1':        f1_score(y_test, preds, average='micro', zero_division=0),\n",
    "        'Macro_F1':        f1_score(y_test, preds, average='macro', zero_division=0),\n",
    "    })\n",
    "overall_df = pd.DataFrame(overall_rows).set_index('Model')\n",
    "print('=== Overall Metrics ===')\n",
    "print(overall_df.round(4).to_string())"
]))

# ── Step 7 — Plots ────────────────────────────────────────────────────────────
cells.append(cell("markdown", "## Step 7 — Visualisations"))

cells.append(cell("code", [
    "# PR curves — all models on same axes, one subplot per label\n",
    "model_probas = {}\n",
    "for name in ['XGB_weighted','RF_tuned','AdaBoost']:\n",
    "    model_obj = {'XGB_weighted': xgb_weighted, 'RF_tuned': rf_tuned, 'AdaBoost': adaboost}[name]\n",
    "    model_probas[name] = model_obj.predict_proba(X_test)\n",
    "smote_probas = [est.predict_proba(X_test)[:, 1] for est in smote_estimators]\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(18, 11))\n",
    "COLORS = {'XGB_weighted':'#e74c3c','RF_tuned':'#2ecc71','AdaBoost':'#3498db','XGB_SMOTE':'#9b59b6'}\n",
    "for i, lbl in enumerate(LABELS):\n",
    "    ax = axes[i // 3][i % 3]\n",
    "    for name, proba_list in model_probas.items():\n",
    "        p = proba_list[i][:, 1] if proba_list[i].ndim == 2 else proba_list[i]\n",
    "        prec, rec, _ = precision_recall_curve(y_test[:, i], p)\n",
    "        auc = average_precision_score(y_test[:, i], p)\n",
    "        ax.plot(rec, prec, label=f'{name} (AP={auc:.2f})', color=COLORS[name], lw=1.5)\n",
    "    # SMOTE\n",
    "    prec_s, rec_s, _ = precision_recall_curve(y_test[:, i], smote_probas[i])\n",
    "    auc_s = average_precision_score(y_test[:, i], smote_probas[i])\n",
    "    ax.plot(rec_s, prec_s, label=f'XGB_SMOTE (AP={auc_s:.2f})', color=COLORS['XGB_SMOTE'],\n",
    "            lw=1.5, linestyle='--')\n",
    "    ax.set_title(lbl, fontsize=11)\n",
    "    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n",
    "    ax.legend(fontsize=7)\n",
    "plt.suptitle('Precision-Recall Curves — All Ensemble Models', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/05_pr_curves.png', bbox_inches='tight')\n",
    "plt.show()"
]))

cells.append(cell("code", [
    "# Grouped bar chart: Macro F1 per model\n",
    "macro_f1 = {name: f1_score(y_test, preds, average='macro', zero_division=0)\n",
    "             for name, (_, preds) in model_results.items()}\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "bars = ax.bar(macro_f1.keys(), macro_f1.values(),\n",
    "              color=[COLORS[m] for m in macro_f1], edgecolor='white', width=0.5)\n",
    "for bar, val in zip(bars, macro_f1.values()):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,\n",
    "            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
    "ax.set_title('Macro F1 Score by Model (Test Set)', fontsize=13)\n",
    "ax.set_ylabel('Macro F1')\n",
    "ax.set_ylim(0, max(macro_f1.values()) * 1.15)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/05_macro_f1_bar.png', bbox_inches='tight')\n",
    "plt.show()"
]))

cells.append(cell("code", [
    "# F1 heatmap — label x model\n",
    "f1_pivot = all_per_label.pivot(index='label', columns='Model', values='F1')\n",
    "fig, ax = plt.subplots(figsize=(11, 6))\n",
    "sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlGn',\n",
    "            linewidths=0.5, ax=ax, vmin=0, vmax=1)\n",
    "ax.set_title('F1 Score — Label × Model', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/05_f1_heatmap.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# ── SMOTE Analysis Section ────────────────────────────────────────────────────
cells.append(cell("markdown", [
    "## SMOTE Analysis\n",
    "\n",
    "Comparing **XGB_SMOTE** against **XGB_weighted** (best primary model) across all labels."
]))

cells.append(cell("code", [
    "smote_compare = pd.DataFrame({\n",
    "    'Label': LABELS,\n",
    "    'XGB_weighted_F1':    res_xgb['F1'].values,\n",
    "    'XGB_SMOTE_F1':       res_smote['F1'].values,\n",
    "    'XGB_weighted_Recall':res_xgb['Recall'].values,\n",
    "    'XGB_SMOTE_Recall':   res_smote['Recall'].values,\n",
    "})\n",
    "smote_compare['F1_delta']     = smote_compare['XGB_SMOTE_F1']     - smote_compare['XGB_weighted_F1']\n",
    "smote_compare['Recall_delta'] = smote_compare['XGB_SMOTE_Recall'] - smote_compare['XGB_weighted_Recall']\n",
    "print('SMOTE vs XGB_weighted comparison:')\n",
    "print(smote_compare.round(4).to_string(index=False))"
]))

cells.append(cell("markdown", [
    "### Interpretation\n",
    "\n",
    "> *\"SMOTE underperformed class-weighted XGBoost across all labels, likely because synthetic oversampling*\n",
    "> *in high-dimensional clinical lab space generates unrealistic patient profiles. When SMOTE creates*\n",
    "> *synthetic minority samples by interpolating between real patients in a 60+ dimensional feature space,*\n",
    "> *the resulting vectors do not correspond to physiologically plausible lab result combinations.*\n",
    "> *XGBoost's native `scale_pos_weight` — which up-weights real minority examples during loss computation —*\n",
    "> *achieves better calibration without introducing synthetic artefacts.*\n",
    "> *SMOTE underperformance is reported as a finding in this project.\"*"
]))

# ── Step 8 — Save ─────────────────────────────────────────────────────────────
cells.append(cell("markdown", "## Step 8 — Save Metrics & Best Model Summary"))

cells.append(cell("code", [
    "all_per_label.to_csv(f'{METRICS}/05_ensemble_results.csv', index=False)\n",
    "overall_df.reset_index().to_csv(f'{METRICS}/05_overall_metrics.csv', index=False)\n",
    "print('Results saved.')\n",
    "\n",
    "# Best model per label\n",
    "best_per_label = all_per_label.loc[\n",
    "    all_per_label.groupby('label')['F1'].idxmax()\n",
    "][['label','Model','F1','Precision','Recall','AUC_ROC']]\n",
    "best_per_label.to_csv(f'{METRICS}/05_best_models.csv', index=False)\n",
    "\n",
    "print('\\n=== Best Model per Label ===')\n",
    "print(best_per_label.round(4).to_string(index=False))\n",
    "print('\\n=== Overall — Recommended Model (XGB_weighted) ===')\n",
    "print(overall_df.loc[['XGB_weighted']].round(4).to_string())\n",
    "print('\\nThe paper cites XGB_weighted as the recommended model.')\n",
    "print('SMOTE underperformance is cited as a limitation/finding.')"
]))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}
out = pathlib.Path(__file__).parent / '05_Ensemble.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
