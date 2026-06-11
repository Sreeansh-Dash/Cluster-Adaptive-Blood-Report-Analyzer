"""
Generator for notebooks_v2/02_Preprocessing.ipynb
"""
import json, pathlib

def cell(cell_type, source, **kwargs):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    else:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

# Title
cells.append(cell("markdown", [
    "# 02 — Preprocessing Pipeline (Leakage-Free)\n",
    "\n",
    "All `sklearn` `.fit()` calls use **train data only**.  \n",
    "Test data is only **transformed** — never fitted.  \n",
    "Outputs: 4 CSVs (train/test × scaled/unscaled) + 2 model artefacts."
]))

# Imports
cells.append(cell("code", [
    "import os, re, warnings, joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from datasets import load_dataset\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_theme(style='whitegrid', font_scale=1.0)\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "\n",
    "SPLITS    = '../data/splits'\n",
    "PROCESSED = '../data/processed'\n",
    "MODELS    = '../models'\n",
    "PLOTS     = '../results/plots'\n",
    "for d in [SPLITS, PROCESSED, MODELS, PLOTS]:\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "print('Dirs ready.')"
]))

# Step 1 — Load raw data and split IDs
cells.append(cell("markdown", "## Step 1 — Load Raw Data & Split by Patient IDs"))

cells.append(cell("code", [
    "print('Loading NidaanKosha...')\n",
    "ds = load_dataset('ekacare/NidaanKosha-100k-V1.0')\n",
    "df = ds['train'].to_pandas()\n",
    "print(f'Raw shape: {df.shape}')\n",
    "\n",
    "train_ids = pd.read_csv(f'{SPLITS}/train_ids.csv')['document_id'].tolist()\n",
    "test_ids  = pd.read_csv(f'{SPLITS}/test_ids.csv')['document_id'].tolist()\n",
    "print(f'Train patients: {len(train_ids):,}  |  Test patients: {len(test_ids):,}')\n",
    "\n",
    "df_train = df[df['document_id'].isin(train_ids)].copy()\n",
    "df_test  = df[df['document_id'].isin(test_ids)].copy()\n",
    "print(f'df_train rows: {len(df_train):,}  |  df_test rows: {len(df_test):,}')"
]))

# Step 2 — Specimen filter
cells.append(cell("markdown", "## Step 2 — Specimen Filter (Blood Only)"))

cells.append(cell("code", [
    "BLOOD_TYPES = {'blood', 'serum', 'plasma', 'whole blood'}\n",
    "\n",
    "def filter_blood(df):\n",
    "    mask = df['specimen'].str.lower().str.strip().isin(BLOOD_TYPES)\n",
    "    return df[mask].copy()\n",
    "\n",
    "print('Before filter:')\n",
    "print(df_train['specimen'].value_counts().head(10).to_string())\n",
    "\n",
    "df_train = filter_blood(df_train)\n",
    "df_test  = filter_blood(df_test)\n",
    "print(f'\\nAfter filter — train: {len(df_train):,}  test: {len(df_test):,}')"
]))

# Step 3 — LOINC-based column standardisation
cells.append(cell("markdown", [
    "## Step 3 — LOINC-Based Column Standardisation\n",
    "\n",
    "- Group rows by LOINC code; assign **canonical test name** = most common `test_name` for that LOINC in **train**.\n",
    "- Rows with no LOINC: keep `test_name` as-is.\n",
    "- Apply mapping to both splits."
]))

cells.append(cell("code", [
    "# Build mapping on TRAIN only\n",
    "loinc_map = (\n",
    "    df_train.dropna(subset=['loinc'])\n",
    "            .groupby('loinc')['test_name']\n",
    "            .agg(lambda x: x.value_counts().index[0])  # modal test_name\n",
    ")\n",
    "print(f'LOINC codes mapped: {len(loinc_map)}')\n",
    "print(loinc_map.head(10).to_string())\n",
    "\n",
    "def apply_loinc_map(df, mapping):\n",
    "    df = df.copy()\n",
    "    df['canonical_test'] = df.apply(\n",
    "        lambda r: mapping.get(r['loinc'], r['test_name']) if pd.notna(r['loinc']) else r['test_name'],\n",
    "        axis=1\n",
    "    )\n",
    "    return df\n",
    "\n",
    "df_train = apply_loinc_map(df_train, loinc_map)\n",
    "df_test  = apply_loinc_map(df_test,  loinc_map)\n",
    "print(f'\\nUnique canonical tests — train: {df_train[\"canonical_test\"].nunique()}')"
]))

# Step 3b — Select top tests by train coverage
cells.append(cell("code", [
    "# Keep only tests present in >= 20% of TRAIN patients\n",
    "n_train_pts = df_train['document_id'].nunique()\n",
    "cov = df_train.groupby('canonical_test')['document_id'].nunique()\n",
    "cov_pct = cov / n_train_pts\n",
    "selected_tests = cov_pct[cov_pct >= 0.20].index.tolist()\n",
    "print(f'Tests with >=20% patient coverage: {len(selected_tests)}')\n",
    "\n",
    "df_train = df_train[df_train['canonical_test'].isin(selected_tests)].copy()\n",
    "df_test  = df_test[df_test['canonical_test'].isin(selected_tests)].copy()\n",
    "print(f'Train rows after test selection: {len(df_train):,}')\n",
    "print(f'Test  rows after test selection: {len(df_test):,}')"
]))

# Step 4 — Unit normalisation
cells.append(cell("markdown", [
    "## Step 4 — Unit Normalisation (LOINC-Aware)\n",
    "\n",
    "For each canonical test, the **modal unit in training** is determined.  \n",
    "Known conversion factors are applied when a row uses a non-modal unit.\n",
    "**No value-threshold heuristics** — only actual unit string matching."
]))

cells.append(cell("code", [
    "# Known unit conversions: (from_unit, to_unit) -> multiply_factor\n",
    "UNIT_CONV = {\n",
    "    # Haematology — cell counts\n",
    "    ('/µL',     '10^3/µL'): 1e-3,\n",
    "    ('/ul',     '10^3/µL'): 1e-3,\n",
    "    ('cells/µL','10^3/µL'): 1e-3,\n",
    "    ('10^6/µL', '10^3/µL'): 1e3,\n",
    "    ('10^9/L',  '10^3/µL'): 1.0,\n",
    "    ('/mm3',    '10^3/µL'): 1e-3,\n",
    "    ('x10^3/µL','10^3/µL'): 1.0,\n",
    "    ('K/µL',    '10^3/µL'): 1.0,\n",
    "    # Haemoglobin\n",
    "    ('g/l',     'g/dL'): 0.1,\n",
    "    ('g/L',     'g/dL'): 0.1,\n",
    "    # Glucose / HbA1c\n",
    "    ('mmol/L',  'mg/dL'): 18.016,  # glucose\n",
    "    ('mg/dl',   'mg/dL'): 1.0,\n",
    "    # Lipids — cholesterol\n",
    "    ('mmol/l',  'mg/dL'): 38.67,\n",
    "    # Creatinine\n",
    "    ('µmol/L',  'mg/dL'): 0.01131,\n",
    "    ('umol/L',  'mg/dL'): 0.01131,\n",
    "    # Thyroid (TSH)\n",
    "    ('mU/L',    'mIU/L'): 1.0,\n",
    "    ('µU/mL',   'mIU/L'): 1.0,\n",
    "    ('uIU/mL',  'mIU/L'): 1.0,\n",
    "    # Liver enzymes\n",
    "    ('IU/L',    'U/L'): 1.0,\n",
    "    ('u/l',     'U/L'): 1.0,\n",
    "    # Iron\n",
    "    ('µg/dL',   'μg/dL'): 1.0,\n",
    "    ('ug/dL',   'μg/dL'): 1.0,\n",
    "}\n",
    "\n",
    "# Fit modal units on TRAIN only\n",
    "modal_unit = (\n",
    "    df_train.dropna(subset=['unit'])\n",
    "            .groupby('canonical_test')['unit']\n",
    "            .agg(lambda x: x.value_counts().index[0])\n",
    ")\n",
    "print(f'Modal units determined for {len(modal_unit)} tests')\n",
    "\n",
    "def normalise_units(df, modal_unit_map, conv_table):\n",
    "    df = df.copy()\n",
    "    df['value'] = pd.to_numeric(df['value'], errors='coerce')\n",
    "    for test, modal in modal_unit_map.items():\n",
    "        mask = (df['canonical_test'] == test) & (df['unit'] != modal) & df['unit'].notna()\n",
    "        for row_idx in df[mask].index:\n",
    "            from_u = df.at[row_idx, 'unit']\n",
    "            factor = conv_table.get((from_u, modal), None)\n",
    "            if factor is not None:\n",
    "                df.at[row_idx, 'value'] = df.at[row_idx, 'value'] * factor\n",
    "                df.at[row_idx, 'unit']  = modal\n",
    "    return df\n",
    "\n",
    "print('Normalising train units...')\n",
    "df_train = normalise_units(df_train, modal_unit, UNIT_CONV)\n",
    "print('Normalising test  units...')\n",
    "df_test  = normalise_units(df_test,  modal_unit, UNIT_CONV)\n",
    "print('Unit normalisation done.')"
]))

# Step 5 — Long to wide pivot
cells.append(cell("markdown", "## Step 5 — Long → Wide Pivot"))

cells.append(cell("code", [
    "def long_to_wide(df):\n",
    "    df = df.copy()\n",
    "    df['value'] = pd.to_numeric(df['value'], errors='coerce')\n",
    "    wide = df.pivot_table(\n",
    "        index='document_id',\n",
    "        columns='canonical_test',\n",
    "        values='value',\n",
    "        aggfunc='median'  # median across duplicate tests per patient\n",
    "    )\n",
    "    wide.columns.name = None\n",
    "    wide = wide.reset_index()\n",
    "    # Merge demographics\n",
    "    demo = df[['document_id','age','gender']].drop_duplicates('document_id')\n",
    "    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')\n",
    "    wide = wide.merge(demo, on='document_id', how='left')\n",
    "    return wide\n",
    "\n",
    "print('Pivoting train...')\n",
    "train_wide = long_to_wide(df_train)\n",
    "print('Pivoting test...')\n",
    "test_wide  = long_to_wide(df_test)\n",
    "\n",
    "# Align columns: test set must have SAME columns as train\n",
    "feat_cols = [c for c in train_wide.columns if c not in ['document_id','age','gender']]\n",
    "for col in feat_cols:\n",
    "    if col not in test_wide.columns:\n",
    "        test_wide[col] = np.nan\n",
    "test_wide = test_wide[train_wide.columns]  # same column order\n",
    "\n",
    "print(f'Train wide: {train_wide.shape}  |  Test wide: {test_wide.shape}')"
]))

# Step 6 — Missing value handling
cells.append(cell("markdown", [
    "## Step 6 — Missing Value Handling\n",
    "\n",
    "- Add binary missingness indicators for features with >20% missing **in train**.\n",
    "- Fit `SimpleImputer(median)` on **train** only."
]))

cells.append(cell("code", [
    "# Compute missingness rate on TRAIN\n",
    "miss_rate_train = train_wide[feat_cols].isnull().mean()\n",
    "high_miss = miss_rate_train[miss_rate_train > 0.20].index.tolist()\n",
    "print(f'Features with >20% missing in train: {len(high_miss)}')\n",
    "\n",
    "# Add binary missingness indicators (apply to both splits)\n",
    "for col in high_miss:\n",
    "    train_wide[col + '_missing'] = train_wide[col].isnull().astype(int)\n",
    "    test_wide[col  + '_missing'] = test_wide[col].isnull().astype(int)\n",
    "\n",
    "# Recompute feat_cols to include missingness indicators\n",
    "miss_indicator_cols = [c + '_missing' for c in high_miss]\n",
    "num_cols = feat_cols  # original numeric features\n",
    "\n",
    "# Fit imputer on TRAIN\n",
    "imputer = SimpleImputer(strategy='median')\n",
    "train_wide[num_cols] = imputer.fit_transform(train_wide[num_cols])\n",
    "test_wide[num_cols]  = imputer.transform(test_wide[num_cols])\n",
    "\n",
    "joblib.dump(imputer, f'{MODELS}/imputer.pkl')\n",
    "print('Imputer saved → models/imputer.pkl')\n",
    "print(f'Missing after imputation — train: {train_wide[num_cols].isnull().sum().sum()}')\n",
    "print(f'Missing after imputation — test : {test_wide[num_cols].isnull().sum().sum()}')"
]))

# Step 7 — Outlier handling (5-sigma, train stats)
cells.append(cell("markdown", [
    "## Step 7 — Outlier Handling (Medical-Safe)\n",
    "\n",
    "Cap at ±5 standard deviations from the **training mean** only.  \n",
    "5σ is chosen because extreme lab values are often real disease signals, not noise."
]))

cells.append(cell("code", [
    "train_means = train_wide[num_cols].mean()\n",
    "train_stds  = train_wide[num_cols].std()\n",
    "\n",
    "lo = train_means - 5 * train_stds\n",
    "hi = train_means + 5 * train_stds\n",
    "\n",
    "def cap_outliers(df, lo, hi, cols):\n",
    "    df = df.copy()\n",
    "    for col in cols:\n",
    "        df[col] = df[col].clip(lower=lo[col], upper=hi[col])\n",
    "    return df\n",
    "\n",
    "train_wide = cap_outliers(train_wide, lo, hi, num_cols)\n",
    "test_wide  = cap_outliers(test_wide,  lo, hi, num_cols)\n",
    "print('Outlier capping done (5σ from train mean).')\n",
    "\n",
    "# Save UNSCALED wide format (needed for clinical threshold labeling in 04)\n",
    "train_wide.to_csv(f'{PROCESSED}/train_wide_unscaled.csv', index=False)\n",
    "test_wide.to_csv( f'{PROCESSED}/test_wide_unscaled.csv',  index=False)\n",
    "print('Unscaled CSVs saved.')"
]))

# Step 8 — Scaling
cells.append(cell("markdown", [
    "## Step 8 — Standard Scaling\n",
    "\n",
    "Fit `StandardScaler` on **train** (numeric test columns only — exclude `age` and missingness indicators).  \n",
    "Transform both splits."
]))

cells.append(cell("code", [
    "scale_cols = num_cols  # age is in demo, not in feat_cols; missingness indicators stay binary\n",
    "\n",
    "scaler = StandardScaler()\n",
    "train_scaled = train_wide.copy()\n",
    "test_scaled  = test_wide.copy()\n",
    "\n",
    "train_scaled[scale_cols] = scaler.fit_transform(train_wide[scale_cols])\n",
    "test_scaled[scale_cols]  = scaler.transform(test_wide[scale_cols])\n",
    "\n",
    "joblib.dump(scaler, f'{MODELS}/scaler.pkl')\n",
    "print('Scaler saved → models/scaler.pkl')"
]))

# Step 9 — Save outputs
cells.append(cell("markdown", "## Step 9 — Save All Outputs"))

cells.append(cell("code", [
    "train_scaled.to_csv(f'{PROCESSED}/train_wide_scaled.csv', index=False)\n",
    "test_scaled.to_csv( f'{PROCESSED}/test_wide_scaled.csv',  index=False)\n",
    "print('Scaled CSVs saved.')\n",
    "\n",
    "print('\\n=== PREPROCESSING SUMMARY ===')\n",
    "print(f'Train shape (scaled): {train_scaled.shape}')\n",
    "print(f'Test  shape (scaled): {test_scaled.shape}')\n",
    "print(f'Numeric features     : {len(num_cols)}')\n",
    "print(f'Missingness indicators: {len(miss_indicator_cols)}')\n",
    "print(f'Missing % before imputation (train mean): {miss_rate_train.mean()*100:.1f}%')\n",
    "print(f'Missing % after  imputation (train)     : {train_wide[num_cols].isnull().mean().mean()*100:.1f}%')\n",
    "print('\\nFiles saved:')\n",
    "print('  data/processed/train_wide_scaled.csv')\n",
    "print('  data/processed/test_wide_scaled.csv')\n",
    "print('  data/processed/train_wide_unscaled.csv')\n",
    "print('  data/processed/test_wide_unscaled.csv')\n",
    "print('  models/imputer.pkl')\n",
    "print('  models/scaler.pkl')"
]))

# Missingness heatmap
cells.append(cell("markdown", "## Appendix — Missingness Heatmap (Sample)"))

cells.append(cell("code", [
    "# Visualise missingness on a 200-patient sample of train\n",
    "sample_ids = train_wide.sample(200, random_state=42)['document_id']\n",
    "sample_df  = train_wide[train_wide['document_id'].isin(sample_ids)][num_cols]\n",
    "\n",
    "# Before imputation we'd need the raw — we show the post-imputation heatmap of scaled values\n",
    "fig, ax = plt.subplots(figsize=(18, 6))\n",
    "sns.heatmap(sample_df.isnull(), cmap='YlOrRd', cbar=False,\n",
    "            xticklabels=False, yticklabels=False, ax=ax)\n",
    "ax.set_title('Missingness (post-imputation check) — 200 train patients')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/imputation_heatmap.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# Write notebook
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}
out = pathlib.Path(__file__).parent / '02_Preprocessing.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
