# Academic Report Data Collection

## 1. Dataset Overview

**Total Patient Count (Train + Test):** 99,992

**All 109 Column Names Exactly:**
`['document_id', '% TRANSFERRIN SATURATION', '25-OH VITAMIN D (TOTAL)', 'A/G Ratio', 'ALKALINE PHOSPHATASE', 'ASPARTATE AMINOTRANSFERASE (SGOT )', 'AST/ALT Ratio', 'Absolute Basophil Count', 'Absolute Eosinophil Count', 'Absolute Lymphocyte Count', 'Absolute Monocyte Count', 'Absolute Neutrophil Count', 'Albumin', 'BLOOD UREA NITROGEN (BUN)', 'BUN / SR.CREATININE RATIO', 'Basophils', 'Bilirubin Direct', 'Bilirubin Indirect', 'Bilirubin Total', 'CALCIUM', 'Chloride', 'Creatinine', 'ESR Erythrocyte Sedimentation Rate', 'EST. GLOMERULAR FILTRATION RATE (eGFR)', 'Eosinophils', 'Estimated Average Glucose (eAG)', 'GAMMA GLUTAMYL TRANSFERASE (GGT)', 'Globulin', 'Glucose Fasting', 'HDL Cholesterol', 'HbA1c', 'Hemoglobin', 'Iron', 'LDL / HDL RATIO', 'LDL Cholesterol', 'Lymphocytes', 'MCH', 'MCHC', 'MCV', 'MPV', 'Monocytes', 'Neutrophils', 'Non HDL Cholesterol', 'PCT', 'PCV', 'PDW', 'PLATELET TO LARGE CELL RATIO(PLCR)', 'Phosphorous', 'Platelet Count', 'Platelet distribution width-CV', 'Potassium', 'RBC Count', 'RDW-CV', 'RDW-SD', 'SGPT (ALT)', 'Sodium', 'TC/ HDL CHOLESTEROL RATIO', 'THYROID STIMULATING HORMONE (TSH)', 'TOTAL CHOLESTEROL', 'TOTAL IRON BINDING CAPACITY (TIBC)', 'TOTAL THYROXINE (T4)', 'TOTAL TRIIODOTHYRONINE (T3)', 'Total Protein', 'Total WBC Count', 'Triglycerides', 'Urea', 'Uric Acid', 'VITAMIN B-12', 'VLDL CHOLESTEROL', 'age', 'gender', '% TRANSFERRIN SATURATION_missing', '25-OH VITAMIN D (TOTAL)_missing', 'A/G Ratio_missing', 'ALKALINE PHOSPHATASE_missing', 'ASPARTATE AMINOTRANSFERASE (SGOT )_missing', 'AST/ALT Ratio_missing', 'Absolute Basophil Count_missing', 'Albumin_missing', 'BLOOD UREA NITROGEN (BUN)_missing', 'BUN / SR.CREATININE RATIO_missing', 'CALCIUM_missing', 'Chloride_missing', 'Creatinine_missing', 'ESR Erythrocyte Sedimentation Rate_missing', 'EST. GLOMERULAR FILTRATION RATE (eGFR)_missing', 'Estimated Average Glucose (eAG)_missing', 'GAMMA GLUTAMYL TRANSFERASE (GGT)_missing', 'Globulin_missing', 'Glucose Fasting_missing', 'HbA1c_missing', 'Iron_missing', 'LDL / HDL RATIO_missing', 'Non HDL Cholesterol_missing', 'PCT_missing', 'PDW_missing', 'PLATELET TO LARGE CELL RATIO(PLCR)_missing', 'Phosphorous_missing', 'Platelet distribution width-CV_missing', 'Potassium_missing', 'RDW-SD_missing', 'Sodium_missing', 'THYROID STIMULATING HORMONE (TSH)_missing', 'TOTAL IRON BINDING CAPACITY (TIBC)_missing', 'TOTAL THYROXINE (T4)_missing', 'TOTAL TRIIODOTHYRONINE (T3)_missing', 'Urea_missing', 'Uric Acid_missing', 'VITAMIN B-12_missing']`

**Train Labels Statistics:**
* Anaemia: Pos=12355, Neg=67638, Rate=15.45%
* Diabetes_Risk: Pos=6546, Neg=73447, Rate=8.18%
* Dyslipidemia: Pos=16551, Neg=63442, Rate=20.69%
* Kidney_Risk: Pos=15390, Neg=64603, Rate=19.24%
* Liver_Stress: Pos=21419, Neg=58574, Rate=26.78%
* Thyroid_Abnormal: Pos=19663, Neg=60330, Rate=24.58%

**Cluster Counts (train_clusters.csv):**
* Cluster 2: 32740
* Cluster 1: 26660
* Cluster 3: 19534
* Cluster 0: 960
* Cluster 4: 99

---

## 2. Clustering Details

**Full Silhouette Sweep Table:**
```
k=3: inertia=4785364  silhouette=0.0472  DB=3.6878
k=4: inertia=4684195  silhouette=0.0488  DB=3.2173
k=5: inertia=4582368  silhouette=0.0491  DB=2.7666
k=6: inertia=4506626  silhouette=0.0307  DB=3.1574
k=7: inertia=4412956  silhouette=0.0424  DB=2.7707
k=8: inertia=4365171  silhouette=0.0420  DB=2.4569
```

**Number of PCA Components:** 
Retained `n_components=0.95` (explains 95% variance) which results in `55` components.

**tSNE Visualization Cell Description:**
The tSNE plot shows 3,000 randomly sampled train patients across 3 subplots:
1. `tSNE - by Cluster`: Displays points scattered based on the 5 predicted clusters. The clusters overlap significantly with each other. Uses the tab10 color palette with small marker sizes (`s=8`) and transparency (`alpha=0.6`).
2. `tSNE - by Age Group`: Shows the same 2D points, colored sequentially using the viridis color palette for age groups: `<18`, `18-34`, `35-49`, `50-64`, `65+`.
3. `tSNE - by Gender`: Displays points coded by male (`steelblue`) and female (`coral`).

**Exact Feature Exclusion Logic:**
```python
# Identify numeric feature columns (exclude id, age, gender, _missing indicators)
EXCLUDE = ['document_id', 'age', 'gender']
feat_cols = [c for c in train_wide.columns
             if c not in EXCLUDE and not c.endswith('_missing')]
```

**results/metrics/03_clustering_summary.json (Full Content):**
```json
{"K": 5, "silhouette": 0.0491, "davies_bouldin": 2.7666}
```

---

## 3. Cluster vs Global Threshold Novelty

**results/metrics/cluster_thresholds.csv:**
* Shape: 355 rows × 4 columns
* First 10 rows:
```
         test  cluster  lower_5th  upper_95th
0  Hemoglobin        0     7.1000      14.400
1  Hemoglobin        1    11.8000      16.700
2  Hemoglobin        2     9.2000      15.100
3  Hemoglobin        3    10.8000      16.300
4  Hemoglobin        4    11.0900      15.700
5   RBC Count        0     2.4395       5.150
6   RBC Count        1     4.0800       5.860
7   RBC Count        2     3.6000       5.590
8   RBC Count        3     4.0000       5.760
9   RBC Count        4     3.6000       5.682
```
* Unique test names present: 51
* Unique clusters present: 5

**results/metrics/global_thresholds.csv:**
* Shape: 51 rows × 3 columns
* First 10 rows:
```
                                      test  lower_5th  upper_95th
0                               Hemoglobin     10.000       16.20
1                                RBC Count      3.780        5.75
2                                      MCV     71.772       99.60
3                                      MCH     24.400       34.50
4                                     MCHC     24.400       34.50
5                                    HbA1c      0.000        8.10
6                            HbA1c_missing      0.000        8.10
7          Estimated Average Glucose (eAG)      0.000      171.42
8                          Glucose Fasting      0.000      171.42
9  Estimated Average Glucose (eAG)_missing      0.000      171.42
```

**Manual Cluster Difference Computation (5 examples vs global):**
1. **Hemoglobin, Cluster 0:** 
   * Cluster Thresholds: [7.10, 14.40] 
   * Global Thresholds: [10.00, 16.20] 
   * % Difference: Cluster lower bound is **29.0% lower** than global; upper bound is **11.1% lower**.
2. **Hemoglobin, Cluster 1:** 
   * Cluster Thresholds: [11.80, 16.70] 
   * Global Thresholds: [10.00, 16.20] 
   * % Difference: Cluster lower bound is **18.0% higher** than global; upper bound is **3.1% higher**.
3. **Hemoglobin, Cluster 4:** 
   * Cluster Thresholds: [11.09, 15.70] 
   * Global Thresholds: [10.00, 16.20] 
   * % Difference: Cluster lower bound is **10.9% higher** than global; upper bound is **3.1% lower**.
4. **RBC Count, Cluster 0:** 
   * Cluster Thresholds: [2.4395, 5.150] 
   * Global Thresholds: [3.780, 5.750] 
   * % Difference: Cluster lower bound is **35.5% lower** than global; upper bound is **10.4% lower**.
5. **RBC Count, Cluster 1:** 
   * Cluster Thresholds: [4.08, 5.86] 
   * Global Thresholds: [3.780, 5.750] 
   * % Difference: Cluster lower bound is **7.9% higher** than global; upper bound is **1.9% higher**.

**Divergence Reporting:**
* Total (test, cluster) pairs: 355
* % Diverging > 10% from global on either bound: 70.0%

---

## 4. Baseline Model Results (Notebook 04)

**Per-label F1, Precision, Recall, AUC_ROC:**
```
     Model             label        F1  Precision    Recall   AUC_ROC
   RF_base           Anaemia  0.447544   0.880468  0.299595  0.864749
   RF_base     Diabetes_Risk  0.408705   0.725916  0.282921  0.854093
   RF_base      Dyslipidemia  0.432145   0.702758  0.313460  0.749503
   RF_base       Kidney_Risk  0.449675   0.806659  0.312151  0.760233
   RF_base      Liver_Stress  0.562848   0.778842  0.441860  0.781878
   RF_base  Thyroid_Abnormal  0.515578   0.811808  0.377317  0.803738

RF_balanced          Anaemia  0.449587   0.852445  0.306073  0.864506
RF_balanced    Diabetes_Risk  0.417028   0.672808  0.301564  0.854336
RF_balanced     Dyslipidemia  0.433941   0.670557  0.320743  0.750424
RF_balanced      Kidney_Risk  0.445982   0.760086  0.314693  0.759902
RF_balanced     Liver_Stress  0.563938   0.765666  0.446864  0.781825
RF_balanced Thyroid_Abnormal  0.518539   0.793798  0.385558  0.803135

       KNN           Anaemia  0.324954   0.584346  0.224291  0.640960
       KNN     Diabetes_Risk  0.253831   0.457816  0.174820  0.621455
       KNN      Dyslipidemia  0.371625   0.509748  0.293444  0.612080
       KNN       Kidney_Risk  0.377594   0.590525  0.279093  0.644917
       KNN      Liver_Stress  0.479421   0.594776  0.401131  0.662244
       KNN  Thyroid_Abnormal  0.446230   0.619048  0.348612  0.669818

       SVM           Anaemia  0.330451   0.898864  0.203239  0.599723
       SVM     Diabetes_Risk  0.334426   0.854962  0.207868  0.597519
       SVM      Dyslipidemia  0.356590   0.806657  0.228224  0.598288
       SVM       Kidney_Risk  0.365648   0.871025  0.231922  0.605230
       SVM      Liver_Stress  0.490715   0.829141  0.348651  0.644682
       SVM  Thyroid_Abnormal  0.445646   0.843641  0.303102  0.627694
```

**Aggregate Metrics for Base Models:**
```
      Model  Hamming_Loss  Subset_Accuracy  Micro_F1  Macro_F1
    RF_base      0.245412         0.304715  0.484454  0.469416
RF_balanced      0.277189         0.232562  0.481666  0.471503
        KNN      0.431313         0.052653  0.394629  0.375609
        SVM      0.350909         0.104705  0.406601  0.387246
```

**feature_cols_for_label Definition (Verbatim):**
```python
def feature_cols_for_label(label, columns):
    """
    Returns all columns EXCEPT other labels, plus the cluster_id.
    """
    other_labels = [l for l in LABELS if l != label]
    return [c for c in columns if c not in other_labels]
```

---

## 5. Ensemble Model Results (Notebook 05)

**Per-label Metrics (from 05_ensemble_results.csv / nb06):**
```
         Model             label        F1  Precision    Recall   AUC_ROC
  XGB_weighted           Anaemia  0.420198   0.505703  0.359919  0.865910
  XGB_weighted     Diabetes_Risk  0.409229   0.399596  0.419342  0.866038
  XGB_weighted      Dyslipidemia  0.399869   0.420658  0.381048  0.744158
  XGB_weighted       Kidney_Risk  0.425695   0.476483  0.384661  0.758229
  XGB_weighted      Liver_Stress  0.535274   0.539798  0.530825  0.772023
  XGB_weighted  Thyroid_Abnormal  0.472232   0.520803  0.431872  0.793390

      RF_tuned           Anaemia  0.403337   0.551745  0.317409  0.866504
      RF_tuned     Diabetes_Risk  0.392497   0.457816  0.344033  0.860161
      RF_tuned      Dyslipidemia  0.410013   0.528437  0.335260  0.751167
      RF_tuned       Kidney_Risk  0.431918   0.612140  0.332304  0.760777
      RF_tuned      Liver_Stress  0.544428   0.640954  0.472856  0.781845
      RF_tuned  Thyroid_Abnormal  0.495738   0.663185  0.395725  0.803852

 LGBM_weighted           Anaemia  0.422904   0.531201  0.351417  0.862211
 LGBM_weighted     Diabetes_Risk  0.422045   0.458284  0.391216  0.862369
 LGBM_weighted      Dyslipidemia  0.409484   0.437976  0.384534  0.742337
 LGBM_weighted       Kidney_Risk  0.425648   0.485294  0.379261  0.756262
 LGBM_weighted      Liver_Stress  0.535478   0.538575  0.532415  0.772522
 LGBM_weighted  Thyroid_Abnormal  0.481923   0.538462  0.435987  0.791535

      AdaBoost           Anaemia  0.799316   0.941320  0.694332  0.970220
      AdaBoost     Diabetes_Risk  0.960105   0.985651  0.935832  0.996160
      AdaBoost      Dyslipidemia  0.748811   0.941916  0.621370  0.957580
      AdaBoost       Kidney_Risk  0.801849   0.912858  0.714911  0.981968
      AdaBoost      Liver_Stress  0.847968   0.963173  0.757379  0.968381
      AdaBoost  Thyroid_Abnormal  0.904377   0.981954  0.838160  0.994727

     XGB_SMOTE           Anaemia  0.997665   0.997665  0.997665  0.999942
     XGB_SMOTE     Diabetes_Risk  0.993928   0.996156  0.991709  0.999962
     XGB_SMOTE      Dyslipidemia  0.992084   0.990516  0.993657  0.999810
     XGB_SMOTE       Kidney_Risk  0.994324   0.996123  0.992532  0.999915
     XGB_SMOTE      Liver_Stress  0.991432   0.994767  0.988120  0.999672
     XGB_SMOTE  Thyroid_Abnormal  0.996606   0.998146  0.995071  0.999629
```

**Aggregate Ensemble Metrics:**
```
         Model  Hamming_Loss  Subset_Accuracy  Micro_F1  Macro_F1
  XGB_weighted      0.265638         0.192810  0.454354  0.443750
      RF_tuned      0.270039         0.189459  0.460678  0.446322
 LGBM_weighted      0.280414         0.162708  0.460666  0.449580
      AdaBoost      0.055603         0.711686  0.837435  0.843738
     XGB_SMOTE      0.002225         0.987649  0.994138  0.994340
```

**Exact `scale_pos_weight` Values Computed:**
* Anaemia: `spw=5.46`
* Diabetes_Risk: `spw=11.29`
* Dyslipidemia: `spw=3.83`
* Kidney_Risk: `spw=4.19`
* Liver_Stress: `spw=2.73`
* Thyroid_Abnormal: `spw=3.06`

**results/metrics/05_threshold_tuning.csv:**
```
Model,label,Threshold,Val_F1
XGB_weighted,Anaemia,0.5,0.4333868378812199
XGB_weighted,Diabetes_Risk,0.45,0.412590799031477
XGB_weighted,Dyslipidemia,0.35,0.41787770770496296
XGB_weighted,Kidney_Risk,0.4,0.42825456406756524
XGB_weighted,Liver_Stress,0.4,0.5407523510971787
XGB_weighted,Thyroid_Abnormal,0.35,0.47843421529539687
RF_tuned,Anaemia,0.45,0.408130081300813
RF_tuned,Diabetes_Risk,0.4,0.3857729138166895
RF_tuned,Dyslipidemia,0.4,0.4272478141719953
RF_tuned,Kidney_Risk,0.4,0.4368098159509202
RF_tuned,Liver_Stress,0.45,0.5562810133194045
RF_tuned,Thyroid_Abnormal,0.45,0.4977709845879506
LGBM_weighted,Anaemia,0.55,0.4312360109425516
LGBM_weighted,Diabetes_Risk,0.65,0.42555190230155004
LGBM_weighted,Dyslipidemia,0.45,0.4237530928506316
LGBM_weighted,Kidney_Risk,0.5,0.42926356589147285
LGBM_weighted,Liver_Stress,0.45,0.5421288647628039
LGBM_weighted,Thyroid_Abnormal,0.45,0.48272716907992247
```

---

## 6. Final Evaluation (Notebook 06)

**results/metrics/06_final_summary.json (Full Content):**
```json
{
    "Best overall model": "XGB_weighted",
    "Best per-label models": {
        "Anaemia": "XGB_weighted",
        "Diabetes_Risk": "XGB_weighted",
        "Dyslipidemia": "XGB_weighted",
        "Kidney_Risk": "XGB_weighted",
        "Liver_Stress": "XGB_weighted",
        "Thyroid_Abnormal": "RF_tuned"
    },
    "AUC_ROC": 0.731402593287192,
    "Macro_F1": 0.4390782689832999,
    "Hamming_Loss": 0.29219794323049486,
    "Silhouette Score": 0.0491,
    "% cluster thresholds diverging from global (>10%)": 70.0,
    "Labels ordered by F1": [
        "Anaemia",
        "Diabetes_Risk",
        "Dyslipidemia",
        "Kidney_Risk",
        "Liver_Stress",
        "Thyroid_Abnormal"
    ]
}
```

**Novelty Comparison Table (Output from nb06):**
```
Count of matched tests: 20
% of matched cluster-thresholds that diverge >10% from global: 70.0%

Top 5 largest divergences:
```
*Note: The table itself does not print due to truncation in the Jupyter notebook's standard output.*

**Thresholds Applied for XGB_weighted:**
* Anaemia: `0.5`
* Diabetes_Risk: `0.45`
* Dyslipidemia: `0.35`
* Kidney_Risk: `0.4`
* Liver_Stress: `0.4`
* Thyroid_Abnormal: `0.35`

**Final Per-Label Metrics for XGB_weighted and RF_tuned:**
```
               Anaemia  Diabetes_Risk  Dyslipidemia  Kidney_Risk  Liver_Stress  Thyroid_Abnormal
Model                                                                                               
RF_tuned       0.403337       0.392497      0.410013     0.431918      0.544428          0.495738   
XGB_weighted   0.420198       0.409229      0.399869     0.425695      0.535274          0.472232 
```

---

## 7. Saved Plots and Visualizations

*All files presently inside `results/plots/` (after cleanup) and their likely contents based on their names:*

1. `04_confusion_matrices.png`: Subplots of confusion matrices for evaluating baseline classification models on multiple labels.
2. `04_pr_curves.png`: Precision-Recall curves evaluating performance of base classification models.
3. `05_f1_heatmap.png`: A heatmap displaying the Macro-F1 scores across different ensemble models and condition labels.
4. `05_macro_f1_bar.png`: A bar chart comparing the overall Macro-F1 score across different ensemble models.
5. `05_pr_curves.png`: Precision-Recall curves comparing the performance of the tuned ensemble models.
6. `06_novelty_validation.png`: A visualization highlighting the divergence of cluster-specific thresholds vs global rules for clinical validation.
7. `age_gender_dist.png`: Bar/Histogram plots of the age and gender demographics in the dataset.
8. `bootstrap_ari.png`: Boxplots/histograms showing Adjusted Rand Index distributions across bootstrapped clustering iterations for stability analysis.
9. `cluster_feature_importance.png`: Bar chart indicating the features that contribute the most variance to determining the KMeans clusters.
10. `cluster_missingness.png`: Heatmap/bars detailing missing value rates broken down across the derived clusters.
11. `clustering_selection.png`: Elbow curve indicating KMeans inertia and Silhouette Scores to choose optimal K.
12. `display_ranges_patterns.png`: Visualization of distributions of lab test values.
13. `imputation_heatmap.png`: Heatmap demonstrating the distribution and location of missing values vs imputed values.
14. `missing_heatmap.png`: Heatmap illustrating null sparsity and missingness density globally in the raw dataset.
15. `pca_scree.png`: A scree plot showing the cumulative explained variance percentage mapped against the number of PCA components.
16. `specimen_distribution.png`: Bar chart summarizing the most common specimen types collected (e.g. serum, whole blood).
17. `tests_per_patient.png`: Histogram documenting the distribution of the total number of tests conducted per individual patient.
18. `top20_tests.png`: Horizontal bar chart listing the 20 most frequently occurring laboratory tests in the dataset.
19. `tsne_clusters.png`: A 2D t-SNE scatterplot highlighting the clusters, age-groups, and genders of patients.
20. `unit_inconsistency.png`: Visualization checking for disparate metrics or unstandardized units across identical test types.
21. `verified_cluster_missingness.png`: Missingness distribution across fully verified valid data subsets/clusters.

---

## 8. Notebook 03c — Label Generation Logic

**Exact Percentile Values Used:**
```python
LOW_PCT  = 5               # percentile for low-abnormal thresholds
HIGH_PCT = 95              # percentile for high-abnormal thresholds
```

**Biomarker Groups & Condition Label logic (Verbatim):**
```python
# direction: 'high' | 'low' | 'both'
BIOMARKER_GROUPS = {
    'Anaemia': [
        {'terms': ['haemoglobin','hemoglobin','hgb'], 'direction': 'low'},
        {'terms': ['rbc'],                            'direction': 'low'},
        {'terms': ['mcv'],                            'direction': 'low'},
        {'terms': ['mch'],                            'direction': 'low'},
        {'terms': ['mchc'],                           'direction': 'low'},
    ],
    'Diabetes_Risk': [
        {'terms': ['hba1c','hemoglobin a1c','glycated'], 'direction': 'high'},
        {'terms': ['glucose','fasting glucose','blood sugar'], 'direction': 'high'},
    ],
    'Dyslipidemia': [
        {'terms': ['total cholesterol','cholesterol'], 'direction': 'high'},
        {'terms': ['ldl'],                             'direction': 'high'},
        {'terms': ['triglyceride','triglycerides'],    'direction': 'high'},
        {'terms': ['vldl'],                            'direction': 'high'},
        {'terms': ['hdl'],                             'direction': 'low'},
    ],
    'Kidney_Risk': [
        {'terms': ['creatinine'],                      'direction': 'high'},
        {'terms': ['bun','blood urea nitrogen'],        'direction': 'high'},
        {'terms': ['urea'],                             'direction': 'high'},
        {'terms': ['uric acid'],                        'direction': 'high'},
        {'terms': ['egfr','gfr'],                       'direction': 'low'},
    ],
    'Liver_Stress': [
        {'terms': ['alt','sgpt'],                       'direction': 'high'},
        {'terms': ['ast','sgot'],                       'direction': 'high'},
        {'terms': ['ggt'],                              'direction': 'high'},
        {'terms': ['bilirubin'],                        'direction': 'high'},
        {'terms': ['alp','alkaline phosphatase'],        'direction': 'high'},
    ],
    'Thyroid_Abnormal': [
        {'terms': ['tsh'],         'direction': 'both'},
        {'terms': ['t3','ft3'],    'direction': 'both'},
        {'terms': ['t4','ft4'],    'direction': 'both'},
        {'terms': ['thyroid'],     'direction': 'both'},
    ],
}
```

**Global Threshold Computation Logic (Verbatim):**
*The global thresholds are computed from train data only.*
```python
            # global fallback (computed once across all train rows)
            global_vals = train_df[cols].stack().dropna()
            if len(global_vals) == 0:
                skipped.append({'label': label, 'terms': str(terms)})
                continue
            g_low  = float(np.percentile(global_vals, LOW_PCT))
            g_high = float(np.percentile(global_vals, HIGH_PCT))
```
