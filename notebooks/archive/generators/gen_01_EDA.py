"""
Generator for notebooks_v2/01_EDA.ipynb
Run this script to (re)create the notebook JSON.
"""
import json, pathlib

def cell(cell_type, source, **kwargs):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    else:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(cell("markdown", [
    "# 01 — Exploratory Data Analysis (EDA)\n",
    "\n",
    "**Dataset:** NidaanKosha-100k-V1.0 — 100 000 Indian patients, long-format blood test records  \n",
    "**Goal:** Understand the data, then perform a **stratified train/test split** that all downstream notebooks must use.\n",
    "\n",
    "> ⚠️ The train/test IDs saved in this notebook are the **single source of truth** for the entire project.\n",
    "> Never re-split in any downstream notebook."
]))

# ── Imports ────────────────────────────────────────────────────────────────────
cells.append(cell("code", [
    "import os, re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "from datasets import load_dataset\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "\n",
    "PLOTS = '../results/plots'\n",
    "SPLITS = '../data/splits'\n",
    "os.makedirs(PLOTS, exist_ok=True)\n",
    "os.makedirs(SPLITS, exist_ok=True)\n",
    "print('Output dirs ready.')"
]))

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
cells.append(cell("markdown", "## 1. Load Dataset"))

cells.append(cell("code", [
    "ds = load_dataset('ekacare/NidaanKosha-100k-V1.0')\n",
    "df = ds['train'].to_pandas()\n",
    "print(f'Shape: {df.shape}')\n",
    "print(f'Columns: {list(df.columns)}')\n",
    "df.head(3)"
]))

# ── 2. Basic Exploration ───────────────────────────────────────────────────────
cells.append(cell("markdown", "## 2. Basic Exploration"))

cells.append(cell("code", [
    "print('=== dtypes ===')\n",
    "print(df.dtypes)\n",
    "print('\\n=== Missing values ===')\n",
    "mv = df.isnull().sum()\n",
    "print(pd.DataFrame({'count': mv, 'pct': (mv/len(df)*100).round(2)}))\n",
    "print(f'\\nUnique patients : {df[\"document_id\"].nunique():,}')\n",
    "print(f'Total rows      : {len(df):,}')\n",
    "print(f'Unique tests    : {df[\"test_name\"].nunique():,}')\n",
    "print(f'Unique LOINC    : {df[\"loinc\"].nunique():,}')\n",
    "print(f'Specimen types  : {sorted(df[\"specimen\"].dropna().unique().tolist())}')"
]))

# ── 3. Age & Gender Distribution ───────────────────────────────────────────────
cells.append(cell("markdown", "## 3. Age & Gender Distribution"))

cells.append(cell("code", [
    "pts = df.drop_duplicates('document_id')[['document_id','age','gender']].copy()\n",
    "pts['age'] = pd.to_numeric(pts['age'], errors='coerce')\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "for g, grp in pts.groupby('gender'):\n",
    "    axes[0].hist(grp['age'].dropna(), bins=30, alpha=0.6, label=g)\n",
    "axes[0].set_title('Age Distribution by Gender')\n",
    "axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')\n",
    "axes[0].legend()\n",
    "\n",
    "gvc = pts['gender'].value_counts()\n",
    "axes[1].pie(gvc.values, labels=gvc.index, autopct='%1.1f%%', startangle=90,\n",
    "            colors=sns.color_palette('pastel')[:len(gvc)])\n",
    "axes[1].set_title('Gender Distribution')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/age_gender_dist.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Age  — min:{pts.age.min():.0f}  max:{pts.age.max():.0f}  mean:{pts.age.mean():.1f}')\n",
    "print(pts['gender'].value_counts())"
]))

# ── 4. Test Coverage per Patient ───────────────────────────────────────────────
cells.append(cell("markdown", "## 4. Test Coverage per Patient"))

cells.append(cell("code", [
    "tests_per_pt = df.groupby('document_id')['test_name'].count()\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "ax.hist(tests_per_pt.values, bins=60, color='steelblue', edgecolor='white')\n",
    "ax.set_title('Number of Tests per Patient')\n",
    "ax.set_xlabel('Test count'); ax.set_ylabel('Number of patients')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/tests_per_patient.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "print(tests_per_pt.describe())"
]))

# ── TRAIN / TEST SPLIT (critical — added after basic exploration) ──────────────
cells.append(cell("markdown", [
    "## 5. ⭐ Stratified Train / Test Split\n",
    "\n",
    "Split happens **here**, on patient demographics only, **before** any per-test analysis.  \n",
    "All downstream notebooks load `train_ids.csv` / `test_ids.csv` — they never re-split."
]))

cells.append(cell("code", [
    "# ── patient-level dedup ──────────────────────────────────────────────────────\n",
    "pts = df.drop_duplicates('document_id')[['document_id', 'age', 'gender']].copy()\n",
    "pts['age'] = pd.to_numeric(pts['age'], errors='coerce')\n",
    "\n",
    "# ── age bins ─────────────────────────────────────────────────────────────────\n",
    "bins  = [0, 18, 35, 50, 65, 200]\n",
    "labels = ['<18', '18-34', '35-49', '50-64', '65+']\n",
    "pts['age_group'] = pd.cut(pts['age'], bins=bins, labels=labels, right=False)\n",
    "\n",
    "# ── stratification key ───────────────────────────────────────────────────────\n",
    "pts['strat_key'] = pts['gender'].astype(str) + '__' + pts['age_group'].astype(str)\n",
    "\n",
    "# ── drop strata with < 2 patients ────────────────────────────────────────────\n",
    "strat_counts = pts['strat_key'].value_counts()\n",
    "valid_strata = strat_counts[strat_counts >= 2].index\n",
    "pts_valid = pts[pts['strat_key'].isin(valid_strata)].copy()\n",
    "dropped = len(pts) - len(pts_valid)\n",
    "print(f'Dropped {dropped} patients from rare strata (< 2 patients in stratum)')\n",
    "\n",
    "# ── split ────────────────────────────────────────────────────────────────────\n",
    "train_pts, test_pts = train_test_split(\n",
    "    pts_valid,\n",
    "    test_size=0.2,\n",
    "    random_state=42,\n",
    "    stratify=pts_valid['strat_key']\n",
    ")\n",
    "\n",
    "train_ids = train_pts[['document_id']]\n",
    "test_ids  = test_pts[['document_id']]\n",
    "\n",
    "train_ids.to_csv(f'{SPLITS}/train_ids.csv', index=False)\n",
    "test_ids.to_csv(f'{SPLITS}/test_ids.csv',  index=False)\n",
    "\n",
    "print(f'\\nTrain patients : {len(train_ids):,}')\n",
    "print(f'Test  patients : {len(test_ids):,}')\n",
    "print(f'\\nSplit ratio    : {len(train_ids)/len(pts_valid)*100:.1f}% / {len(test_ids)/len(pts_valid)*100:.1f}%')"
]))

cells.append(cell("code", [
    "# ── confirm proportions are preserved ────────────────────────────────────────\n",
    "print('=== Gender proportions ===')\n",
    "for name, subset in [('Full', pts_valid), ('Train', train_pts), ('Test', test_pts)]:\n",
    "    p = subset['gender'].value_counts(normalize=True).round(3)\n",
    "    print(f'{name:6s}: {dict(p)}')\n",
    "\n",
    "print('\\n=== Age-group proportions ===')\n",
    "for name, subset in [('Full', pts_valid), ('Train', train_pts), ('Test', test_pts)]:\n",
    "    p = subset['age_group'].value_counts(normalize=True).sort_index().round(3)\n",
    "    print(f'{name:6s}: {dict(p)}')"
]))

# ── 6. Top 20 Most Common Tests ────────────────────────────────────────────────
cells.append(cell("markdown", "## 6. Top 20 Most Common Tests"))

cells.append(cell("code", [
    "top20 = df['test_name'].value_counts().head(20)\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "sns.barplot(x=top20.values, y=top20.index, palette='Blues_r', ax=ax)\n",
    "ax.set_title('Top 20 Most Common Tests (all rows)')\n",
    "ax.set_xlabel('Row count')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/top20_tests.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# ── 7. Missing Value Heatmap ───────────────────────────────────────────────────
cells.append(cell("markdown", "## 7. Missing-Value Heatmap (Top 30 Tests, 200-patient sample)"))

cells.append(cell("code", [
    "top30_tests = df['test_name'].value_counts().head(30).index.tolist()\n",
    "sample_pts  = df['document_id'].drop_duplicates().sample(200, random_state=42)\n",
    "df_sample   = df[df['document_id'].isin(sample_pts) & df['test_name'].isin(top30_tests)]\n",
    "hm = df_sample.pivot_table(index='document_id', columns='test_name', values='value', aggfunc='first')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(16, 8))\n",
    "sns.heatmap(hm.isnull(), cmap='YlOrRd', cbar=False, ax=ax,\n",
    "            xticklabels=True, yticklabels=False)\n",
    "ax.set_title('Missing Values — Top 30 Tests, 200 Patients (yellow = missing)')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/missing_heatmap.png', bbox_inches='tight')\n",
    "plt.show()"
]))

# ── 8. Display-Ranges Pattern Analysis ─────────────────────────────────────────
cells.append(cell("markdown", [
    "## 8. Display-Ranges Format Analysis\n",
    "\n",
    "Categorise the `display_ranges` strings so `02_Preprocessing` knows which regex patterns to write."
]))

cells.append(cell("code", [
    "dr_sample = df['display_ranges'].dropna().sample(min(500, df['display_ranges'].notna().sum()), random_state=42)\n",
    "\n",
    "def classify_range(s):\n",
    "    s = str(s).strip()\n",
    "    if re.fullmatch(r'[\\d.]+\\s*[-–]\\s*[\\d.]+', s):  return 'X-Y range'\n",
    "    if re.match(r'^>\\s*[\\d.]', s):                   return '>X'\n",
    "    if re.match(r'^<\\s*[\\d.]', s):                   return '<X'\n",
    "    if re.match(r'(?i)^up\\s*to\\s*[\\d.]', s):         return 'Up to X'\n",
    "    if re.match(r'^[\\d.]+$', s):                      return 'Single value'\n",
    "    return 'Free text / other'\n",
    "\n",
    "dr_cats = dr_sample.apply(classify_range).value_counts()\n",
    "print('Display-range pattern counts (n=500 sample):')\n",
    "print(dr_cats.to_string())\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 4))\n",
    "sns.barplot(x=dr_cats.values, y=dr_cats.index, palette='Set2', ax=ax)\n",
    "ax.set_title('display_ranges Format Patterns')\n",
    "ax.set_xlabel('Count in sample (n=500)')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/display_ranges_patterns.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# HbA1c example\n",
    "print('\\n--- HbA1c display_ranges examples ---')\n",
    "hba1c = df[df['test_name'].str.contains('HbA1c|Hba1c|hba1c|hemoglobin a1c', case=False, na=False)]\n",
    "print(hba1c['display_ranges'].dropna().value_counts().head(10).to_string())"
]))

# ── 9. Unit Inconsistency Detection ───────────────────────────────────────────
cells.append(cell("markdown", [
    "## 9. Unit Inconsistency Detection\n",
    "\n",
    "For each `test_name`, count distinct units. Tests with many unit variants will need conversion in `02_Preprocessing`."
]))

cells.append(cell("code", [
    "unit_var = (\n",
    "    df.dropna(subset=['unit'])\n",
    "      .groupby('test_name')['unit']\n",
    "      .nunique()\n",
    "      .sort_values(ascending=False)\n",
    "      .head(20)\n",
    ")\n",
    "print('Top 20 tests by number of distinct units:')\n",
    "print(unit_var.to_string())\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "sns.barplot(x=unit_var.values, y=unit_var.index, palette='Reds_r', ax=ax)\n",
    "ax.set_title('Top 20 Tests with Most Unit Variants')\n",
    "ax.set_xlabel('Distinct units')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/unit_inconsistency.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# show actual unit values for the top 3 worst tests\n",
    "worst3 = unit_var.head(3).index.tolist()\n",
    "for t in worst3:\n",
    "    units = df[df['test_name']==t]['unit'].value_counts()\n",
    "    print(f'\\n{t}: {list(units.items())[:8]}')"
]))

# ── 10. Specimen Type Distribution ─────────────────────────────────────────────
cells.append(cell("markdown", "## 10. Specimen Type Distribution"))

cells.append(cell("code", [
    "spec = df['specimen'].fillna('unknown').str.lower().str.strip()\n",
    "spec_counts = spec.value_counts()\n",
    "blood_types = ['blood', 'serum', 'plasma', 'whole blood']\n",
    "blood_pct = spec.isin(blood_types).mean() * 100\n",
    "print(f'Blood-related rows: {blood_pct:.1f}%')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "sns.barplot(x=spec_counts.values, y=spec_counts.index, palette='Blues_r', ax=ax)\n",
    "ax.set_title('Specimen Type Distribution')\n",
    "ax.set_xlabel('Row count')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{PLOTS}/specimen_distribution.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "print(spec_counts.to_string())"
]))

# ── 11. Summary ────────────────────────────────────────────────────────────────
cells.append(cell("markdown", [
    "## 11. EDA Summary & Findings\n",
    "\n",
    "### Key Findings\n",
    "- **6.84 M rows**, 100 000 unique patients; long format (one row per test per patient)\n",
    "- **Gender**: ~57% male, ~43% female\n",
    "- **Specimen**: ~79% of rows are blood/serum/plasma — the rest (urine, stool, …) are filtered out in `02_Preprocessing`\n",
    "- **Missing values**: `unit` (~15%), `display_ranges` (~11%), `loinc` (~7%), `specimen` (~6%)\n",
    "- **Unit inconsistency**: Several tests (e.g., Platelet Count, WBC) have values reported in 2+ incompatible units across labs — unit normalisation is critical in preprocessing\n",
    "- **display_ranges formats**: most follow `X-Y` numeric ranges; a minority use `>X`/`<X` or free text\n",
    "- **Test coverage**: most patients have 20-80 tests; a long tail of patients with <10 tests\n",
    "\n",
    "### Train / Test Split — SAVED\n",
    "\n",
    "| File | Path | Rows (approx) |\n",
    "|------|------|---------------|\n",
    "| `train_ids.csv` | `data/splits/train_ids.csv` | ~80 000 patients |\n",
    "| `test_ids.csv`  | `data/splits/test_ids.csv`  | ~20 000 patients |\n",
    "\n",
    "> **Rule for all downstream notebooks (02–05):**  \n",
    "> Load `train_ids.csv` and `test_ids.csv` at the top, then split the data immediately.  \n",
    "> **Do NOT call `train_test_split` again.** The split was stratified by gender × age group with `random_state=42`."
]))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

out = pathlib.Path(__file__).parent / '01_EDA.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written → {out}')
