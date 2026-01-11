# <img src="https://github.com/user-attachments/assets/660efd96-ca6c-47f7-b98f-42cb866e241f" width="40" style="vertical-align: text-bottom;"/> Decision Tree–Based Trade Pattern Mining Pipeline

## <img src="https://github.com/user-attachments/assets/dcdcffb4-c4e2-40ee-84cc-aca8612d257e" width="30px">Summary

This repository implements an end-to-end quantitative machine learning pipeline developed during the Xcitium–NuFintech Internship. 
The project focuses on transforming raw financial features into interpretable, statistically validated, and production-ready trading patterns using leakage-safe feature engineering, decision tree–based clustering, and rigorous multi-stage validation. The core philosophy of this project is robustness over complexity — prioritizing explainability, out-of-sample reliability, and temporal stability rather than black-box performance.


## <img src="https://github.com/user-attachments/assets/f03f321f-8340-452b-b82a-33f487bb52a4" width="28" style="vertical-align: text-bottom;"/> MODULE 1: FEATURE ENGINEERING & SELECTION

### <img src="https://github.com/user-attachments/assets/6672ee8c-15ed-4fb5-9cd5-63c04ac747c1" height="20px" style="vertical-align:bottom;"> Working:
Four-stage feature reduction process applied to prevent data leakage and optimize model performance:  

- 1️⃣ Analyzed **190 raw features** for redundancy via correlation analysis, low-variance detection, and one-hot group identification.
- 2️⃣ Performed **80/20 stratified train-test split BEFORE any feature engineering** to ensure zero information leakage, created target-encoded features using only training data statistics.
- 3️⃣ Reduced features from **81 to 50** through time-window consolidation (kept 5d, dropped other variants), domain-based filtering with Random Forest importance verification, and final importance-based ranking.
- 4️⃣ Validated that feature reduction maintained test performance while reducing overfitting.

### 📦 Output Files:

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 1.1a (Feature Cleanup Analysis):
- `features_to_drop.txt`  – List of redundant features identified for removal.  
- `cleaned_features.txt`  – Initial cleaned feature list after redundancy removal.  
- `feature_analysis_report.json`  – Detailed redundancy analysis with correlation and variance statistics.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 1.1b (Dataset Preparation):
- `train_prepared.parquet` – Training dataset with **13,600 samples** and engineered features.  
- `test_prepared.parquet`  – Test dataset with **3,400 samples** using same transformations.  
- `prepared_features.txt`  – List of **81 features** after cleanup and encoding.  
- `feature_engineering_artifacts.json`  – Target encoding maps learned from training data only.  
- `dataset_summary.json` – Train/test split statistics and class distributions.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 1.2 (Feature Reduction):
- `final_features.txt` – Final **50 optimized features** selected for modeling.  
- `reduction_summary.json`  – Step-by-step reduction statistics showing **81 → 50** feature reduction.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 1.3 (Feature Validation):
- `validation_comparison.png`  – Train vs test accuracy comparison chart demonstrating overfitting reduction.  
- `performance_comparison.csv` – Detailed metrics including train accuracy, test accuracy, and overfitting gap comparison.  



## <img src="https://github.com/user-attachments/assets/8f09f616-c687-48a1-8a30-76dffda656b9" width="28" style="position: relative; top: 3px;"/> MODULE 2: MODEL TRAINING (DECISION TREE CLUSTERING)

### <img src="https://github.com/user-attachments/assets/6672ee8c-15ed-4fb5-9cd5-63c04ac747c1" height="20px" style="vertical-align:bottom;"> Working:
Trained decision tree classifier on **50 optimized features** using training data only with hyperparameters:  
`max_depth=10`, `min_samples_split=30`, `min_samples_leaf=20`.  
resulting in **195 leaf nodes** where each leaf represents a distinct pattern cluster defined by specific feature threshold conditions from root to leaf.

### 📦 Output Files:
- `decision_tree_model.pkl` – Trained decision tree model with **195 leaf nodes**  
- `model_metadata.json` – Model hyperparameters, tree depth, number of leaves, and feature names.  
- `cluster_rules.json`  – Complete path conditions from root to each leaf for all **195 clusters**  
- `decision_tree.png`   – Tree visualization showing top 3 levels of split logic.  
- `confusion_matrix.png`  – Test set performance visualization.  
- `feature_importance.png` – Bar chart of top 15 most important features for splitting decisions.  



## <img src="https://github.com/user-attachments/assets/39198c0b-6b23-4755-9fe9-86135085a06b" width="28" style="position: relative; top: 3px;"/> MODULE 3: CLUSTER ANALYSIS & QUALITY FILTERING

### <img src="https://github.com/user-attachments/assets/6672ee8c-15ed-4fb5-9cd5-63c04ac747c1" height="20px" style="vertical-align:bottom;"> Working:
Analyzed all **195 decision tree leaf node clusters** on combined train and test data, calculated cluster purity (dominant class percentage), size, and top 5 distinguishing features per cluster, then applied actionable criteria filtering for **purity ≥70%** and **size ≥50 samples** to identify high-quality patterns, resulting in **42 actionable clusters** meeting quality thresholds.

### 📦 Output Files:
- `cluster_statistics.json`  – Complete statistics for all **195 clusters** including size, purity, target distribution, and top 5 distinguishing features with percentage differences from global means.  
- `actionable_clusters.json` – **42 patterns** meeting quality criteria with cluster ID, size, purity, dominant class, and confidence score.  
- `cluster_distribution.png` – Four-panel visualization showing cluster sizes, target distribution by cluster, purity histogram, and size distribution.  



## <img src="https://github.com/user-attachments/assets/da81abd6-53fd-49a0-814f-d2b8d0062730" width="28" style="position: relative; top: 3px;"/> MODULE 4: RULE EXTRACTION & ANALYSIS


### <img src="https://github.com/user-attachments/assets/6672ee8c-15ed-4fb5-9cd5-63c04ac747c1" height="20px" style="vertical-align:bottom;">  Working:
Extracted interpretable decision rules by tracing root-to-leaf paths for **42 actionable clusters**, converting tree split conditions into readable if-then rules, organized rules into **3 confidence tiers** based on purity.  
(Tier 1: ≥95%, Tier 2: 85–95%, Tier 3: 70–85%),

Calculated signal strength scores (0–10) based on confidence and sample size, then analyzed patterns across rules to identify most important features, common feature combinations, signal direction balance, and potential rule conflicts.

### 📦 Output Files:

### <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 4.1 (Rule Extraction):

- `rule_catalog.json` – All **42 extracted rules** with full conditions, tier assignments, confidence scores, signal strength, and predicted classes.  
- `rules_summary.csv` – Tabular summary with cluster ID, signal direction, confidence percentage, sample size, strength score, and condition count.  
- `rules_detailed.txt`– Human-readable format with complete conditions and trading interpretations for all **42 rules**.  
- `rules_quick_reference.txt` – Condensed trading desk guide showing only **Tier 1 rules (≥95% confidence)**.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 4.2 (Rule Analysis & Pattern Detection):
- `feature_importance_analysis.png`  – Dual-panel chart showing overall feature frequency and tier-based breakdown.  
- `rule_analysis.json` – Comprehensive analysis data including feature frequencies by tier, signal direction statistics, top features by direction, and identified conflicts.  
- `key_insights.txt`   – Actionable summary with top 5 critical features, signal bias assessment, and improvement recommendations.  



## <img src="https://github.com/user-attachments/assets/6467a42f-afae-4dc7-8ede-46f12a087c6f" width="20" /> MODULE 5: PATTERN VALIDATION (BACKTESTING & WALK-FORWARD)

### <img src="https://github.com/user-attachments/assets/6672ee8c-15ed-4fb5-9cd5-63c04ac747c1" height="20px" style="vertical-align:bottom;">  Working:
Applied rigorous multi-period validation to **42 actionable patterns** through three stages:  

- 1️⃣ Backtest validation on **20% held-out test set** measuring out-of-sample accuracy per cluster.  
- 2️⃣ Walk-forward validation testing each pattern across **5 independent time periods** to assess temporal stability via mean accuracy, standard deviation, and coefficient of variation. 
- 3️⃣ Comprehensive quality analysis calculating reliability scores (0–100) based on sample size adequacy, stability, accuracy consistency, and train-test degradation, then filtering for patterns with **reliability ≥70** and no critical issues, resulting in **16 production-quality patterns**.

### 📦 Output Files:

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 5.1 (Backtesting):
- `cluster_performance.csv` – Per-cluster test set accuracy, sample counts, true positives, false positives, true negatives, false negatives.  
- `backtest_summary.json`  – Overall performance statistics across all **42 patterns**.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;"> Module 5.2 (Walk-Forward Validation):
- `walkforward_stability.csv`  – Stability metrics per pattern: mean accuracy across 5 periods, standard deviation, coefficient of variation, minimum and maximum accuracy.  
- `walkforward_results_detailed.csv`  – Individual period performance showing accuracy in each of the **5 time windows** per pattern.  
- `walkforward_degrading.csv`  – Patterns showing significant performance degradation over time periods.  
- `stability_analysis.png`  – Visualization of pattern consistency across validation periods.  

###  <img src="https://github.com/user-attachments/assets/f3dcee8e-e008-457a-97fb-d3848b425713" height="20px" style="vertical-align:text-bottom;">  Module 5.3 (Quality Analysis & Final Filtering):
- `pattern_quality_analysis.csv`  – All **42 patterns** with quality grades (A+ to F), reliability scores, tier assignments, accuracy metrics, stability CV, test sample counts, degradation percentages, and flagged issues.  
- `usable_patterns_only.csv`  – **16 patterns** passing final criteria (reliability ≥70, issues=CLEAN) ready for consideration. 
- `grade_a_patterns.csv`  – **14 highest-quality patterns** graded A+ or A representing **87.5% of validated set**.


