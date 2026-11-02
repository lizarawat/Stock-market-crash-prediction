# Stock Market Crash Prediction and Investor Risk Profiling Using Historical Market Data and Machine Learning

**Course:** INT234: Predictive Analytics  
**Academic Session:** 2026–2027  
**Student:** Liza Rawat  
**Institution:** Lovely Professional University, Punjab, India  

---

## 1. Project Overview

This research project develops a multi-algorithm machine learning framework to analyze historical market dynamics, evaluate cross-asset macroeconomic contagion channels, forecast elevated market crash risk on the **NIFTY 50**, and perform unsupervised downside risk profiling across 17 major Indian equities.

The project is structured strictly in alignment with the **INT234 Predictive Analytics** course curriculum, implementing models across:
1. **Supervised Regression:** OLS Multiple Linear Regression, Polynomial Regression (Degree 2 + Ridge).
2. **Supervised Classification:** Logistic Regression (L2), Gaussian Naive Bayes, K-Nearest Neighbors, Decision Trees (CART), Support Vector Machines (RBF), Random Forest Classifier, Multi-Layer Perceptrons (Neural Networks), and Extreme Gradient Boosting (XGBoost).
3. **Unsupervised Learning:** K-Means Clustering, Agglomerative Hierarchical Linkage Clustering (Single, Complete, Average, Ward), and Apriori Association Rule Mining.
4. **Dimensionality Reduction:** Principal Component Analysis (PCA).

---

## 2. Repository Architecture

```
stock_crash_risk_analytics/
│
├── INT234_Predictive_Analytics_Project.ipynb  # Master executable Jupyter Notebook (All 16 Sections)
├── app.py                                    # Interactive Streamlit Web Demonstration Dashboard
├── requirements.txt                          # Python library dependencies
├── README.md                                 # Project setup & execution guide
├── DATASET_DOCUMENTATION.md                  # Comprehensive Data Dictionary & Pipeline Docs
│
├── data/
│   ├── raw/                                  # 23 cached Parquet and CSV historical datasets (2010–2025)
│   └── processed/                            # Engineered analytical matrices and metric tables
│
├── figures/                                  # Publication-quality academic visualizations (300 DPI)
│   ├── fig1_historical_drawdowns_and_targets.png
│   ├── fig2_roc_pr_curves.png
│   ├── fig3_features_and_pca.png
│   ├── fig4_stock_clustering_profiles.png
│   └── fig5_confusion_matrices.png
│
├── models/                                   # Serialized scikit-learn models & scalers (.joblib)
│
├── src/
│   ├── run_experiments.py                    # End-to-end Python experimental script
│   └── generate_notebook.py                  # Automated notebook generator and runner
│
└── research_paper/
    └── INT234_Research_Paper.md              # Complete academic research paper (IEEE format)
```

---

## 3. Key Experimental Results (Out-of-Sample Test: 2022–2025)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Majority Class)** | 0.9513 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0487 | 0.0487 |
| **Logistic Regression (L2)** | 0.6925 | 0.0690 | 0.4255 | 0.1187 | 0.6067 | 0.0676 | 0.2003 |
| **Gaussian Naive Bayes** | 0.9058 | 0.2105 | 0.3404 | 0.2602 | 0.7916 | 0.1685 | 0.0731 |
| **K-Nearest Neighbors (k=7)** | 0.9451 | 0.2000 | 0.0426 | 0.0702 | 0.6358 | 0.0814 | 0.0532 |
| **Decision Tree (Pruned CART)** | 0.6480 | 0.1072 | **0.8511** | 0.1905 | 0.7831 | 0.1248 | 0.1745 |
| **Support Vector Machine (RBF)**| 0.9130 | 0.1967 | 0.2553 | 0.2222 | 0.7729 | 0.1448 | **0.0497** |
| **Random Forest (Bagging)** | 0.9130 | **0.2394** | 0.3617 | **0.2881** | **0.8148** | **0.1980** | 0.1107 |
| **Multi-Layer Perceptron (MLP)**| 0.9513 | 0.0000 | 0.0000 | 0.0000 | 0.4769 | 0.0495 | 0.1044 |
| **Extreme Gradient Boosting** | 0.8468 | 0.0826 | 0.2128 | 0.1190 | 0.7633 | 0.1186 | 0.0952 |

### Continuous Drawdown Regression Performance
- **OLS Multiple Regression:** $MAE = 0.0193$, $RMSE = 0.0238$, $R^2 = -0.0297$
- **Polynomial Regression (Degree 2 + Ridge):** $MAE = 0.0181$, $RMSE = 0.0220$, $R^2 = +0.1198$

### Constituent Equities Downside Risk Clusters
- **Tier 1 (Defensive / Low Downside Risk):** `HDFCBANK`, `HINDUNILVR`, `ITC`, `TCS` ($\text{Beta} < 0.75$, Max Drawdown $< 42\%$)
- **Tier 2 (Moderate Risk / Market Core):** `BHARTIARTL`, `INFY`, `KOTAKBANK`, `LT`, `MARUTI`, `RELIANCE`, `SUNPHARMA`, `WIPRO`
- **Tier 3 (High Downside Risk):** `AXISBANK`, `ICICIBANK`, `ONGC`, `SBIN`, `TATASTEEL` ($\text{Beta} > 1.30$, Max Drawdown up to $76.8\%$)

---

## 4. Setup and Execution Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End Experimental Script
```bash
python src/run_experiments.py
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook INT234_Predictive_Analytics_Project.ipynb
```

### Step 4: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
