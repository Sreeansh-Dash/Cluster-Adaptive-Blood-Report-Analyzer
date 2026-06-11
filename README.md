# Cluster-Adaptive Blood Report Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-006600?style=for-the-badge)
![License](https://img.shields.io/badge/License-CC--BY--SA%204.0-lightgrey?style=for-the-badge)

**Personalized multi-label disease risk detection on 100,000 Indian patients**  
*Replacing static WHO reference ranges with cohort-specific, data-driven thresholds*

[📊 Live Showcase](./showcase/index.html) · [📓 Notebooks](./notebooks/) · [📈 Results](./results/)

</div>

---

## 🧠 Project Overview

Current blood report interpretation uses **one-size-fits-all** population-wide reference ranges. A 22-year-old athlete and a 60-year-old with hypertension share the same "normal" HbA1c threshold — despite radically different risk profiles.

This project builds a **cluster-adaptive system** that:

1. **Clusters** 100,000 Indian patients into demographic cohorts via K-Means + PCA
2. **Learns** cohort-specific 5th–95th percentile reference ranges from real Indian diagnostic lab data
3. **Detects** 6 simultaneous conditions (multi-label) per patient
4. **Scores** severity using ensemble methods (XGBoost, LightGBM, AdaBoost)
5. **Handles** rare pathological conditions via SMOTE oversampling and class weighting

> **Key Finding:** 70% of cluster-specific thresholds diverge >10% from global population norms — proving that one-size-fits-all diagnostics systematically mis-classify patients.

---

## 🏥 Disease Conditions Detected (Multi-Label)

| Condition | Biomarkers | Positive Rate |
|-----------|-----------|---------------|
| 🩸 **Anaemia** | Hemoglobin ↓, RBC ↓, MCV ↓, MCH ↓, MCHC ↓ | 15.45% |
| 🍬 **Diabetes Risk** | HbA1c ↑, Glucose Fasting ↑, eAG ↑ | 8.18% |
| 💧 **Dyslipidemia** | Total Cholesterol ↑, LDL ↑, Triglycerides ↑, HDL ↓ | 20.69% |
| 🫘 **Kidney Risk** | Creatinine ↑, BUN ↑, Urea ↑, Uric Acid ↑, eGFR ↓ | 19.24% |
| 🟤 **Liver Stress** | ALT ↑, AST ↑, GGT ↑, Bilirubin ↑, ALP ↑ | 26.78% |
| 🦋 **Thyroid Abnormal** | TSH ↕, T3 ↕, T4 ↕ (bidirectional) | 24.58% |

---

## 📊 Dataset

**[NidaanKosha-100k-V1.0](https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0) — Eka Care (HuggingFace)**

| Attribute | Value |
|-----------|-------|
| Patients | 99,992 unique Indian patients |
| Lab Readings | 6.84 million individual test values |
| Format | Long → pivoted to Wide (109 columns) |
| Unique Tests | 581 LOINC codes |
| License | CC-BY-SA 4.0 |
| Source | Real lab reports from Eka Care PHR app across Indian diagnostic facilities |

```python
from datasets import load_dataset
ds = load_dataset("ekacare/NidaanKosha-100k-V1.0")
df = ds["train"].to_pandas()
# Pivot long → wide: 99,992 rows × 109 columns
```

---

## 🏗️ Architecture & Pipeline

```
NidaanKosha Dataset (HuggingFace, 100K Indian patients)
        ↓
  01. EDA — Demographics, missingness, test coverage
        ↓
  02. Preprocessing — Long→Wide pivot, LOINC standardization,
      KNN imputation, unit normalization, z-score scaling
        ↓
  03. PCA (55 components, 95% variance) + K-Means (k=5)
      + t-SNE visualization + Bootstrap stability (ARI)
        ↓
  03b. Cluster Verification — Missingness patterns, feature importance
        ↓
  03c. Per-Cluster Reference Range Learning
       (5th–95th percentile per biomarker per cluster)
        ↓
  04. Multi-label Classification Baselines
      (Random Forest, KNN, SVM)
        ↓
  05. Ensemble Models + SMOTE
      (XGBoost, LightGBM, AdaBoost + per-label threshold tuning)
        ↓
  06. Final Evaluation — Model selection, novelty validation
        ↓
  07. Demo — Patient risk report generation
```

---

## 📈 Results

### Clustering
| Metric | Value |
|--------|-------|
| Optimal K | 5 clusters |
| Silhouette Score | 0.0491 |
| Davies-Bouldin | 2.77 (best at k=5) |
| PCA Components | 55 (95% variance) |

**Cluster populations:** Cluster 2 (32,740) · Cluster 1 (26,660) · Cluster 3 (19,534) · Cluster 0 (960) · Cluster 4 (99)

### Threshold Novelty
| Metric | Value |
|--------|-------|
| Total (test, cluster) pairs | 355 |
| Pairs diverging >10% from global | **70%** |
| Unique biomarkers with cluster ranges | 51 |

**Example:** Hemoglobin in Cluster 0 (rare/extreme patients) has a lower bound **29% below** the global threshold — patients who would be classified as "normal" globally are flagged as anaemic in their cohort.

### Baseline Models (Notebook 04)
| Model | Hamming Loss ↓ | Subset Accuracy ↑ | Micro F1 ↑ | Macro F1 ↑ |
|-------|----------------|-------------------|------------|------------|
| **RF_base** | **0.2454** | **0.3047** | **0.4845** | **0.4694** |
| RF_balanced | 0.2772 | 0.2326 | 0.4817 | 0.4715 |
| KNN | 0.4313 | 0.0527 | 0.3946 | 0.3756 |
| SVM | 0.3509 | 0.1047 | 0.4066 | 0.3872 |

### Ensemble Models (Notebook 05) — AUC-ROC per label
| Model | Anaemia | Diabetes | Dyslipidemia | Kidney | Liver | Thyroid | Macro F1 |
|-------|---------|----------|--------------|--------|-------|---------|----------|
| **XGB_weighted ★** | 0.866 | 0.866 | 0.744 | 0.758 | 0.772 | 0.793 | 0.444 |
| RF_tuned | 0.867 | 0.860 | 0.751 | 0.761 | 0.782 | 0.804 | 0.446 |
| LGBM_weighted | 0.862 | 0.862 | 0.742 | 0.756 | 0.773 | 0.792 | 0.450 |

> ★ **XGB_weighted** selected as best overall model for deployment. AUC-ROC 0.731 on real test data, Macro F1 0.439, Hamming Loss 0.292.

---

## 🗂️ Repository Structure

```
📁 Cluster-Adaptive-Blood-Report-Analyzer/
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Pivot, imputation, normalization
│   ├── 03_Clustering.ipynb           # K-Means, PCA, t-SNE, bootstrap
│   ├── 03b_Cluster_Verification.ipynb# Cluster stability & feature importance
│   ├── 03c_ClusterRangeLearning.ipynb# Per-cluster reference range computation
│   ├── 04_Classification.ipynb       # Multi-label baseline classifiers
│   ├── 05_Ensemble.ipynb             # XGBoost, LGBM, AdaBoost + SMOTE
│   ├── 06_Evaluation.ipynb           # Final metrics & model comparison
│   └── 07_Demo.ipynb                 # Patient risk report demo
│
├── 📊 results/
│   ├── plots/                        # All visualization outputs (21 plots)
│   └── metrics/                      # CSV/JSON metric outputs
│
├── 🌐 showcase/
│   ├── index.html                    # Interactive project showcase (this site)
│   ├── style.css                     # Dark-mode premium UI
│   └── app.js                        # Chart.js + scroll animations
│
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Purpose | Libraries |
|---------|-----------|
| Data handling | `pandas`, `numpy` |
| Dataset loading | `datasets` (HuggingFace) |
| Clustering | `scikit-learn` (KMeans, PCA, TSNE) |
| ML models | `scikit-learn`, `xgboost`, `lightgbm` |
| Class imbalance | `imbalanced-learn` (SMOTE) |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Notebook | Jupyter Notebook |

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn \
            matplotlib seaborn plotly jupyter datasets
```

---

## 📚 Syllabus Coverage (BCSE209L Machine Learning)

| Module | Topic | Application |
|--------|-------|-------------|
| Module 2 | Multi-label Classification | 6-condition simultaneous detection |
| Module 3 | KNN, SVM | Baseline classifiers for comparison |
| Module 4 | K-Means, PCA, t-SNE | Patient cohort clustering + visualization |
| Module 5 | Random Forest, XGBoost, AdaBoost | Ensemble severity scoring |
| Module 6 | SMOTE, Hyperparameter Tuning | Rare pathology class imbalance handling |

---

## 🔬 Novelty & Research Angle

> *"Cohort-specific adaptive reference range generation for multi-label blood abnormality detection in Indian patient populations"*

- **Gap in literature:** Most ML papers on lab reports use global (WHO/Western) thresholds
- **This project** learns from real Indian lab `display_ranges` — what Indian diagnostic centres actually consider normal
- **Multi-label + clustering** applied to Indian population data at this scale is novel
- **100,000 patient scale** with geographic diversity across India gives statistical strength
- Potential submission to: *Expert Systems with Applications*, *Computers in Biology and Medicine*, *IEEE Access*

---

## 🚀 Getting Started

1. **Clone this repo**
   ```bash
   git clone https://github.com/Sreeansh-Dash/Cluster-Adaptive-Blood-Report-Analyzer.git
   cd Cluster-Adaptive-Blood-Report-Analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```python
   from datasets import load_dataset
   ds = load_dataset("ekacare/NidaanKosha-100k-V1.0")
   df = ds["train"].to_pandas()
   df.to_parquet("data/raw/nidaankosha.parquet")
   ```

4. **Run notebooks in order:** `01_EDA → 02 → 03 → 03b → 03c → 04 → 05 → 06 → 07`

5. **View the showcase:** Open `showcase/index.html` in a browser

---

## 📝 Citation

Dataset:
```
@dataset{nidaankosha2024,
  title     = {NidaanKosha-100k-V1.0},
  author    = {Eka Care},
  year      = {2024},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0},
  license   = {CC-BY-SA 4.0}
}
```

---

<div align="center">

**BCSE209L Machine Learning · VIT Chennai**  
Built for Amazon ML Summer School Application 2025

</div>
