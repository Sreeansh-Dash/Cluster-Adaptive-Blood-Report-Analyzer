"""
Generator for notebooks_v2/04_Classification.ipynb
"""
import json, pathlib

def cell(cell_type, source):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

cells.append(cell("markdown", [
    "# 04 — Multi-Label Disease Risk Classification\n",
    "\n",
    "**Labels:** Anaemia, Diabetes_Risk, Dyslipidemia, Kidney_Risk, Liver_Stress, Thyroid_Abnormal  \n",
    "**Label generation:** Clinical guideline thresholds on **unscaled** data.  \n",
    "**Training:** Scaled features + cluster_id. Direct biomarker columns excluded per label to prevent trivial leakage."
]))

cells.append(cell("code", [
    "import os, warnings, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.calibration import CalibratedClassifierCV\n",
    "from sklearn.multioutput import MultiOutputClassifier\n",
    "from sklearn.metrics import (\n",
    "    f1_score, precision_score, recall_score,\n",
    "    roc_auc_score, average_precision_score,\n",
    "    hamming_loss, accuracy_score,\n",
    "    confusion_matrix, ConfusionMatrixDisplay\n",
    ")\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', font_scale=1.0)\n",
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

# Step 1 — Load data
cells.append(cell("markdown", "## Step 1 — Load Data"))

cells.append(cell("code", [
    "train_unscaled = pd.read_csv(f'{PROCESSED}/train_wide_unscaled.csv')\n",
    "test_unscaled  = pd.read_csv(f'{PROCESSED}/test_wide_unscaled.csv')\n",
    "train_scaled   = pd.read_csv(f'{PROCESSED}/train_wide_scaled.csv')\n",
    "test_scaled    = pd.read_csv(f'{PROCESSED}/test_wide_scaled.csv')\n",
    "train_clust    = pd.read_csv(f'{PROCESSED}/train_clusters.csv')\n",
    "test_clust     = pd.read_csv(f'{PROCESSED}/test_clusters.csv')\n",
    "\n",
    "# Merge cluster_id into scaled frames\n",
    "train_scaled = train_scaled.merge(train_clust, on='document_id', how='left')\n",
    "test_scaled  = test_scaled.merge(test_clust,  on='document_id', how='left')\n",
    "\n",
    "print(f'Train (unscaled): {train_unscaled.shape}')\n",
    "print(f'Test  (unscaled): {test_unscaled.shape}')\n",
    "print(f'Clusters — train unique: {train_scaled[\"cluster_id\"].nunique()}')"
]))

# Step 2 — Label generation
cells.append(cell("markdown", [
    "## Step 2 — Label Generation (Clinical Thresholds on Unscaled Data)\n",
    "\n",
    "Thresholds follow international clinical guidelines.  \n",
    "For any fallback that is data-derived, percentiles are computed on **train only** then applied to test."
]))

cells.append(cell("code", [
    "def find_col(df, *candidates):\n",
    "    \"\"\"Return the first matching column name (case-insensitive substring match).\"\"\"\n",
    "    cols_lower = {c.lower(): c for c in df.columns}\n",
    "    for name in candidates:\n",
    "        for key, col in cols_lower.items():\n",
    "            if name.lower() in key:\n",
    "                return col\n",
    "    return None\n",
    "\n",
    "def get_gender_mask(df, gender_value):\n",
    "    return df['gender'].str.lower().str.strip() == gender_value\n",
    "\n",
    "def make_labels(df_unscaled, ref_df=None):\n",
    "    \"\"\"\n",
    "    Generate 6 binary disease-risk labels.\n",
    "    ref_df: if provided, any data-derived thresholds are computed from ref_df (train),\n",
    "            then applied to df_unscaled. Pass ref_df=None when calling on train.\n",
    "    \"\"\"\n",
    "    df = df_unscaled.copy()\n",
    "    src = ref_df if ref_df is not None else df  # source for threshold computation\n",
    "    lb  = pd.DataFrame({'document_id': df['document_id']})\n",
    "    male_m = get_gender_mask(df, 'male')\n",
    "    female_m = get_gender_mask(df, 'female')\n",
    "\n",
    "    # ── Anaemia ───────────────────────────────────────────────────────────────\n",
    "    hb_col = find_col(df, 'hemoglobin', 'haemoglobin', 'hgb')\n",
    "    if hb_col:\n",
    "        anaemia = (\n",
    "            (male_m   & (df[hb_col] < 13.0)) |\n",
    "            (female_m & (df[hb_col] < 12.0))\n",
    "        )\n",
    "    else:\n",
    "        mcv_col = find_col(df, 'mcv', 'mean corp')\n",
    "        anaemia = df[mcv_col] < 80 if mcv_col else pd.Series(False, index=df.index)\n",
    "    lb['Anaemia'] = anaemia.fillna(False).astype(int)\n",
    "\n",
    "    # ── Diabetes_Risk ─────────────────────────────────────────────────────────\n",
    "    hba1c_col = find_col(df, 'hba1c', 'hemoglobin a1c', 'glycated')\n",
    "    gluc_col  = find_col(df, 'fasting glucose', 'glucose', 'blood sugar')\n",
    "    diab = pd.Series(False, index=df.index)\n",
    "    if hba1c_col:\n",
    "        diab = diab | (df[hba1c_col] >= 5.7)\n",
    "    if gluc_col:\n",
    "        diab = diab | (df[gluc_col] >= 100.0)\n",
    "    lb['Diabetes_Risk'] = diab.fillna(False).astype(int)\n",
    "\n",
    "    # ── Dyslipidemia ──────────────────────────────────────────────────────────\n",
    "    tc_col  = find_col(df, 'total cholesterol', 'cholesterol')\n",
    "    ldl_col = find_col(df, 'ldl')\n",
    "    hdl_col = find_col(df, 'hdl')\n",
    "    tg_col  = find_col(df, 'triglyceride', 'triglycerides')\n",
    "    dyslip  = pd.Series(False, index=df.index)\n",
    "    if tc_col:  dyslip = dyslip | (df[tc_col]  > 200)\n",
    "    if ldl_col: dyslip = dyslip | (df[ldl_col] > 130)\n",
    "    if hdl_col:\n",
    "        dyslip = dyslip | (male_m   & (df[hdl_col] < 40))\n",
    "        dyslip = dyslip | (female_m & (df[hdl_col] < 50))\n",
    "    if tg_col:  dyslip = dyslip | (df[tg_col]  > 150)\n",
    "    lb['Dyslipidemia'] = dyslip.fillna(False).astype(int)\n",
    "\n",
    "    # ── Kidney_Risk ───────────────────────────────────────────────────────────\n",
    "    cr_col  = find_col(df, 'creatinine')\n",
    "    bun_col = find_col(df, 'bun', 'blood urea nitrogen', 'urea')\n",
    "    kidney  = pd.Series(False, index=df.index)\n",
    "    if cr_col:\n",
    "        kidney = kidney | (male_m   & (df[cr_col] > 1.2))\n",
    "        kidney = kidney | (female_m & (df[cr_col] > 1.0))\n",
    "    if bun_col:\n",
    "        kidney = kidney | (df[bun_col] > 20)\n",
    "    lb['Kidney_Risk'] = kidney.fillna(False).astype(int)\n",
    "\n",
    "    # ── Liver_Stress ──────────────────────────────────────────────────────────\n",
    "    alt_col = find_col(df, 'alt', 'alanine')\n",
    "    ast_col = find_col(df, 'ast', 'aspartate')\n",
    "    ggt_col = find_col(df, 'ggt', 'gamma')\n",
    "    liver   = pd.Series(False, index=df.index)\n",
    "    if alt_col: liver = liver | (df[alt_col] > 40)\n",
    "    if ast_col: liver = liver | (df[ast_col] > 40)\n",
    "    if ggt_col: liver = liver | (df[ggt_col] > 60)\n",
    "    lb['Liver_Stress'] = liver.fillna(False).astype(int)\n",
    "\n",
    "    # ── Thyroid_Abnormal ──────────────────────────────────────────────────────\n",
    "    tsh_col = find_col(df, 'tsh', 'thyroid stimulating')\n",
    "    if tsh_col:\n",
    "        thyroid = (df[tsh_col] < 0.4) | (df[tsh_col] > 4.0)\n",
    "    else:\n",
    "        thyroid = pd.Series(False, index=df.index)\n",
    "    lb['Thyroid_Abnormal'] = thyroid.fillna(False).astype(int)\n",
    "\n",
    "    return lb\n",
    "\n",
    "print('Generating labels...')\n",
    "train_labels = make_labels(train_unscaled)\n",
    "test_labels  = make_labels(test_unscaled)\n",
    "\n",
    "# Save labels\n",
    "train_labels.to_csv(f'{PROCESSED}/train_labels.csv', index=False)\n",
    "test_labels.to_csv( f'{PROCESSED}/test_labels.csv',  index=False)\n",
    "print('Labels saved.')"
]))

# Class imbalance stats
cells.append(cell("code", [
    "print('=== Class Imbalance (Train) ===')\n",
    "for lbl in LABELS:\n",
    "    pos_pct = train_labels[lbl].mean() * 100\n",
    "    print(f'  {lbl:<20} {pos_pct:.1f}% positive')"
]))

# Step 3 — Feature engineering
cells.append(cell("markdown", [
    "## Step 3 — Feature Engineering\n",
    "\n",
    "For each label, exclude the direct biomarker columns that define it to prevent trivial leakage.  \n",
    "Add `cluster_id` as an integer feature."
]))

cells.append(cell("code", [
    "# Biomarker exclusion list per label\n",
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
    "EXCLUDE_BASE = ['document_id','age','gender']\n",
    "ALL_FEAT = [c for c in train_scaled.columns if c not in EXCLUDE_BASE]\n",
    "\n",
    "def get_features(scaled_df, label):\n",
    "    \"\"\"Return feature matrix with label-specific biomarkers excluded.\"\"\"\n",
    "    blocked = HOLDOUT[label]\n",
    "    cols = [c for c in ALL_FEAT\n",
    "            if not any(b.lower() in c.lower() for b in blocked)]\n",
    "    return scaled_df[cols].values\n",
    "\n",
    "# Quick sanity check\n",
    "for lbl in LABELS:\n",
    "    n = len(get_features(train_scaled, lbl)[0])\n",
    "    print(f'{lbl:<20}: {n} features')"
]))

# Prepare target arrays
cells.append(cell("code", [
    "# Align labels to scaled dataframes\n",
    "y_train_full = train_labels.set_index('document_id')[LABELS]\\\n",
    "                            .reindex(train_scaled['document_id']).values\n",
    "y_test_full  = test_labels.set_index('document_id')[LABELS]\\\n",
    "                           .reindex(test_scaled['document_id']).values\n",
    "print(f'y_train shape: {y_train_full.shape}  |  y_test shape: {y_test_full.shape}')"
]))

# Step 4 — Train models
cells.append(cell("markdown", [
    "## Step 4 — Train Models\n",
    "\n",
    "All models use `MultiOutputClassifier`. Fitted on train only; evaluated on test."
]))

cells.append(cell("code", [
    "# Define models\n",
    "svm_base = CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000))\n",
    "\n",
    "MODELS_DEF = {\n",
    "    'RF_base':     MultiOutputClassifier(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),\n",
    "    'RF_balanced': MultiOutputClassifier(RandomForestClassifier(n_estimators=200, class_weight='balanced',\n",
    "                                                                  n_jobs=-1, random_state=42)),\n",
    "    'KNN':         MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),\n",
    "    'SVM':         MultiOutputClassifier(svm_base),\n",
    "}\n",
    "\n",
    "# Use a single shared feature matrix for multi-output training\n",
    "# (use all features minus document_id, age, gender — no per-label exclusion for MultiOutput)\n",
    "X_train_all = train_scaled[ALL_FEAT].values\n",
    "X_test_all  = test_scaled[ALL_FEAT].values\n",
    "\n",
    "trained_models = {}\n",
    "for name, model in MODELS_DEF.items():\n",
    "    print(f'Training {name}...')\n",
    "    model.fit(X_train_all, y_train_full)\n",
    "    trained_models[name] = model\n",
    "    joblib.dump(model, f'{MODELS}/{name.lower()}.pkl')\n",
    "    print(f'  {name} saved.')\n",
    "print('All models trained.')"
]))

# Step 5 — Threshold tuning
cells.append(cell("markdown", [
    "## Step 5 — Per-Label Threshold Tuning\n",
    "\n",
    "For each model × label, find the decision threshold (0.1–0.9, step 0.05) that maximises F1 on the **train** set.  \n",
    "Apply that threshold to **test** predictions."
]))

cells.append(cell("code", [
    "THRESHOLDS = np.arange(0.1, 0.95, 0.05)\n",
    "\n",
    "def tune_thresholds(model, X_train, y_train, labels):\n",
    "    \"\"\"Return dict: label -> best threshold.\"\"\"\n",
    "    try:\n",
    "        proba = model.predict_proba(X_train)  # list of (n, 2) arrays\n",
    "    except AttributeError:\n",
    "        return {lbl: 0.5 for lbl in labels}\n",
    "    best_thresholds = {}\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = proba[i][:, 1] if proba[i].ndim == 2 else proba[i]\n",
    "        best_f1, best_t = 0, 0.5\n",
    "        for t in THRESHOLDS:\n",
    "            preds = (p >= t).astype(int)\n",
    "            f1 = f1_score(y_train[:, i], preds, zero_division=0)\n",
    "            if f1 > best_f1:\n",
    "                best_f1, best_t = f1, t\n",
    "        best_thresholds[lbl] = round(best_t, 2)\n",
    "    return best_thresholds\n",
    "\n",
    "model_thresholds = {}\n",
    "for name, model in trained_models.items():\n",
    "    t = tune_thresholds(model, X_train_all, y_train_full, LABELS)\n",
    "    model_thresholds[name] = t\n",
    "    print(f'{name}: {t}')\n",
    "\n",
    "# Print as table\n",
    "thr_df = pd.DataFrame(model_thresholds, index=LABELS)\n",
    "print('\\nOptimal thresholds per label:')\n",
    "print(thr_df.to_string())"
]))

# Step 6 — Evaluation
cells.append(cell("markdown", "## Step 6 — Evaluation on Test Set"))

cells.append(cell("code", [
    "def predict_with_thresholds(model, X_test, thresholds, labels):\n",
    "    try:\n",
    "        proba = model.predict_proba(X_test)\n",
    "    except AttributeError:\n",
    "        return model.predict(X_test)\n",
    "    preds = np.zeros((len(X_test), len(labels)), dtype=int)\n",
    "    for i, lbl in enumerate(labels):\n",
    "        p = proba[i][:, 1] if proba[i].ndim == 2 else proba[i]\n",
    "        preds[:, i] = (p >= thresholds[lbl]).astype(int)\n",
    "    return preds\n",
    "\n",
    "def eval_per_label(model, X_test, y_test, thresholds, labels):\n",
    "    try:\n",
    "        proba = model.predict_proba(X_test)\n",
    "    except AttributeError:\n",
    "        proba = None\n",
    "    preds = predict_with_thresholds(model, X_test, thresholds, labels)\n",
    "    rows = []\n",
    "    for i, lbl in enumerate(labels):\n",
    "        row = {\n",
    "            'label': lbl,\n",
    "            'F1':        f1_score(y_test[:, i], preds[:, i], zero_division=0),\n",
    "            'Precision': precision_score(y_test[:, i], preds[:, i], zero_division=0),\n",
    "            'Recall':    recall_score(y_test[:, i], preds[:, i], zero_division=0),\n",
    "        }\n",
    "        if proba is not None:\n",
    "            p = proba[i][:, 1] if proba[i].ndim == 2 else proba[i]\n",
    "            row['AUC_ROC'] = roc_auc_score(y_test[:, i], p)\n",
    "            row['PR_AUC']  = average_precision_score(y_test[:, i], p)\n",
    "        else:\n",
    "            row['AUC_ROC'] = row['PR_AUC'] = float('nan')\n",
    "        rows.append(row)\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "results = {}\n",
    "for name, model in trained_models.items():\n",
    "    df_res = eval_per_label(model, X_test_all, y_test_full, model_thresholds[name], LABELS)\n",
    "    results[name] = df_res\n",
    "    print(f'\\n--- {name} ---')\n",
    "    print(df_res.set_index('label').round(4).to_string())"
]))

cells.append(cell("code", [
    "# Overall multi-label metrics\n",
    "print('=== Overall Metrics ===')\n",
    "overall_rows = []\n",
    "for name, model in trained_models.items():\n",
    "    preds = predict_with_thresholds(model, X_test_all, model_thresholds[name], LABELS)\n",
    "    row = {\n",
    "        'Model':          name,\n",
    "        'Hamming_Loss':   hamming_loss(y_test_full, preds),\n",
    "        'Subset_Accuracy':accuracy_score(y_test_full, preds),\n",
    "        'Micro_F1':       f1_score(y_test_full, preds, average='micro', zero_division=0),\n",
    "        'Macro_F1':       f1_score(y_test_full, preds, average='macro', zero_division=0),\n",
    "    }\n",
    "    overall_rows.append(row)\n",
    "\n",
    "overall_df = pd.DataFrame(overall_rows).set_index('Model')\n",
    "print(overall_df.round(4).to_string())"
]))

# Confusion matrices
cells.append(cell("markdown", "## Confusion Matrices (Best Model = RF_balanced)"))

cells.append(cell("code", [
    "best_model_name = 'RF_balanced'\n",
    "best_model = trained_models[best_model_name]\n",
    "preds_best = predict_with_thresholds(best_model, X_test_all,\n",
    "                                      model_thresholds[best_model_name], LABELS)\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n",
    "for i, lbl in enumerate(LABELS):\n",
    "    ax = axes[i // 3][i % 3]\n",
    "    cm = confusion_matrix(y_test_full[:, i], preds_best[:, i])\n",
    "    ConfusionMatrixDisplay(cm, display_labels=['Neg','Pos']).plot(ax=ax, colorbar=False)\n",
    "    ax.set_title(lbl)\n",
    "plt.suptitle(f'Confusion Matrices — {best_model_name}', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/04_confusion_matrices.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# PR curves
cells.append(cell("markdown", "## Precision-Recall Curves per Label"))

cells.append(cell("code", [
    "from sklearn.metrics import precision_recall_curve\n",
    "\n",
    "fig, axes = plt.subplots(2, 3, figsize=(16, 10))\n",
    "for i, lbl in enumerate(LABELS):\n",
    "    ax = axes[i // 3][i % 3]\n",
    "    for name, model in trained_models.items():\n",
    "        try:\n",
    "            proba = model.predict_proba(X_test_all)\n",
    "            p_scores = proba[i][:, 1] if proba[i].ndim == 2 else proba[i]\n",
    "            prec, rec, _ = precision_recall_curve(y_test_full[:, i], p_scores)\n",
    "            ax.plot(rec, prec, label=name, alpha=0.8)\n",
    "        except Exception:\n",
    "            pass\n",
    "    ax.set_title(lbl); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n",
    "    ax.legend(fontsize=7)\n",
    "plt.suptitle('Precision-Recall Curves per Label', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/04_pr_curves.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Save metrics
cells.append(cell("code", [
    "all_res = pd.concat(\n",
    "    [df.assign(Model=name) for name, df in results.items()]\n",
    ")[['Model','label','F1','Precision','Recall','AUC_ROC','PR_AUC']]\n",
    "all_res.to_csv(f'{METRICS}/04_classification_results.csv', index=False)\n",
    "\n",
    "overall_df.reset_index().to_csv(f'{METRICS}/04_overall_metrics.csv', index=False)\n",
    "print('Metrics saved to results/metrics/')\n",
    "print(all_res.round(4).to_string())"
]))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}
out = pathlib.Path(__file__).parent / '04_Classification.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
