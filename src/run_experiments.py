import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Modeling libraries
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve, silhouette_score
)
from xgboost import XGBClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = r'C:\Users\ADMIN\.gemini\antigravity\scratch\stock_crash_risk_analytics'
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

for d in [PROC_DIR, FIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

print("--- 1. Loading and Preprocessing Primary Datasets ---")
# Load NIFTY 50
df_nifty = pd.read_parquet(os.path.join(RAW_DIR, 'NIFTY_50.parquet'))
df_vix = pd.read_parquet(os.path.join(RAW_DIR, 'INDIA_VIX.parquet'))
df_sp500 = pd.read_parquet(os.path.join(RAW_DIR, 'SP500.parquet'))
df_usdinr = pd.read_parquet(os.path.join(RAW_DIR, 'USD_INR.parquet'))
df_crude = pd.read_parquet(os.path.join(RAW_DIR, 'CRUDE_OIL.parquet'))
df_gold = pd.read_parquet(os.path.join(RAW_DIR, 'GOLD.parquet'))

# Standardize index to tz-naive UTC date
def clean_index(df):
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()

df_nifty = clean_index(df_nifty)
df_vix = clean_index(df_vix)
df_sp500 = clean_index(df_sp500)
df_usdinr = clean_index(df_usdinr)
df_crude = clean_index(df_crude)
df_gold = clean_index(df_gold)

# Use NIFTY trading dates as the reference calendar
calendar = df_nifty.index

# Align all exogenous series to NIFTY calendar using forward-fill (strictly avoiding future leaks)
df_merged = pd.DataFrame(index=calendar)
df_merged['NIFTY_Open'] = df_nifty['Open']
df_merged['NIFTY_High'] = df_nifty['High']
df_merged['NIFTY_Low'] = df_nifty['Low']
df_merged['NIFTY_Close'] = df_nifty['Close']
df_merged['NIFTY_Volume'] = df_nifty['Volume']

df_merged['INDIA_VIX'] = df_vix['Close'].reindex(calendar).ffill()
df_merged['SP500_Close'] = df_sp500['Close'].reindex(calendar).ffill()
df_merged['USD_INR'] = df_usdinr['Close'].reindex(calendar).ffill()
df_merged['CRUDE_Close'] = df_crude['Close'].reindex(calendar).ffill()
df_merged['GOLD_Close'] = df_gold['Close'].reindex(calendar).ffill()

# Initial dropna if any leading na
df_merged = df_merged.ffill().bfill()
print("Merged raw dataset shape:", df_merged.shape)

print("--- 2. Feature Engineering ---")
df_feat = pd.DataFrame(index=df_merged.index)
close = df_merged['NIFTY_Close']
high = df_merged['NIFTY_High']
low = df_merged['NIFTY_Low']
vol = df_merged['NIFTY_Volume']

# 1. Technical Indicators (NIFTY 50)
df_feat['NIFTY_Return_1d'] = close.pct_change(1)
df_feat['NIFTY_Return_5d'] = close.pct_change(5)
df_feat['NIFTY_Return_21d'] = close.pct_change(21)
df_feat['NIFTY_Return_63d'] = close.pct_change(63)

# Realized Volatility (Annualized)
df_feat['NIFTY_Vol_21d'] = df_feat['NIFTY_Return_1d'].rolling(21).std() * np.sqrt(252)
df_feat['NIFTY_Vol_63d'] = df_feat['NIFTY_Return_1d'].rolling(63).std() * np.sqrt(252)

# Moving Average Ratios
df_feat['SMA_Ratio_20'] = close / close.rolling(20).mean()
df_feat['SMA_Ratio_50'] = close / close.rolling(50).mean()
df_feat['SMA_Ratio_200'] = close / close.rolling(200).mean()

# Distance from 252-day Rolling High (Drawdown from Peak)
rolling_peak_252 = close.rolling(252, min_periods=63).max()
df_feat['Dist_From_52w_High'] = (close - rolling_peak_252) / rolling_peak_252

# RSI-14
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / (avg_loss + 1e-9)
df_feat['RSI_14'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = close.ewm(span=12, adjust=False).mean()
ema_26 = close.ewm(span=26, adjust=False).mean()
macd_line = ema_12 - ema_26
macd_signal = macd_line.ewm(span=9, adjust=False).mean()
df_feat['MACD_Hist'] = (macd_line - macd_signal) / close

# Volume Trend
df_feat['Volume_Ratio_21d'] = vol / (vol.rolling(21).mean() + 1e-5)

# 2. Implied Volatility (India VIX)
vix = df_merged['INDIA_VIX']
df_feat['VIX_Level'] = vix
df_feat['VIX_Return_5d'] = vix.pct_change(5)
df_feat['VIX_Return_21d'] = vix.pct_change(21)
df_feat['VIX_Ratio_63d'] = vix / vix.rolling(63).mean()

# 3. Macro & Global Contagion Features (Lagged by 1 day to strictly prevent look-ahead bias)
df_feat['SP500_Return_1d'] = df_merged['SP500_Close'].pct_change(1).shift(1)
df_feat['SP500_Return_5d'] = df_merged['SP500_Close'].pct_change(5).shift(1)
df_feat['USD_INR_Return_5d'] = df_merged['USD_INR'].pct_change(5).shift(1)
df_feat['USD_INR_Return_21d'] = df_merged['USD_INR'].pct_change(21).shift(1)
df_feat['CRUDE_Return_21d'] = df_merged['CRUDE_Close'].pct_change(21).shift(1)
df_feat['GOLD_Return_21d'] = df_merged['GOLD_Close'].pct_change(21).shift(1)

print("--- 3. Target Construction (Forward 21-Day Horizon) ---")
# Forward 21-day minimum price
horizon = 21
forward_min = close.iloc[::-1].rolling(window=horizon, min_periods=horizon).min().iloc[::-1].shift(-1)
forward_drawdown = (forward_min - close) / close

df_feat['Target_Forward_MDD'] = forward_drawdown
# Threshold: -7% decline over next 21 trading days constitutes an Elevated Crash Risk event
CRASH_THRESHOLD = -0.07
df_feat['Target_Crash_Binary'] = (forward_drawdown <= CRASH_THRESHOLD).astype(int)

# Drop NaN values generated by 252-day lookback and forward 21-day horizon
df_clean = df_feat.dropna().copy()
print(f"Clean analytical dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
crash_count = df_clean['Target_Crash_Binary'].sum()
crash_rate = df_clean['Target_Crash_Binary'].mean() * 100
print(f"Total Crash Events (MDD <= {CRASH_THRESHOLD*100:.1f}% in next 21d): {crash_count} / {len(df_clean)} ({crash_rate:.2f}%)")

# Save processed dataset
df_clean.to_parquet(os.path.join(PROC_DIR, 'nifty_crash_features.parquet'))
df_clean.to_csv(os.path.join(PROC_DIR, 'nifty_crash_features.csv'))

print("--- 4. Chronological Train, Validation, and Test Splitting ---")
# Train: 2010 to 2019
# Validation: 2020 to 2021 (includes COVID crash)
# Test: 2022 to 2025 (recent rate hike & modern regime)
feature_cols = [c for c in df_clean.columns if not c.startswith('Target_')]

train_mask = (df_clean.index < '2020-01-01')
val_mask = (df_clean.index >= '2020-01-01') & (df_clean.index < '2022-01-01')
test_mask = (df_clean.index >= '2022-01-01')

# Note: apply embargo of 21 days between splits to prevent overlapping target leakage
# Embargo: eliminate last 21 days of train, and last 21 days of val
train_dates = df_clean[train_mask].index[:-21]
val_dates = df_clean[val_mask].index[:-21]
test_dates = df_clean[test_mask].index

X_train, y_train_cls, y_train_reg = df_clean.loc[train_dates, feature_cols], df_clean.loc[train_dates, 'Target_Crash_Binary'], df_clean.loc[train_dates, 'Target_Forward_MDD']
X_val, y_val_cls, y_val_reg = df_clean.loc[val_dates, feature_cols], df_clean.loc[val_dates, 'Target_Crash_Binary'], df_clean.loc[val_dates, 'Target_Forward_MDD']
X_test, y_test_cls, y_test_reg = df_clean.loc[test_dates, feature_cols], df_clean.loc[test_dates, 'Target_Crash_Binary'], df_clean.loc[test_dates, 'Target_Forward_MDD']

print(f"Train set: {X_train.shape[0]} days (Crash rate: {y_train_cls.mean()*100:.2f}%)")
print(f"Val set:   {X_val.shape[0]} days (Crash rate: {y_val_cls.mean()*100:.2f}%)")
print(f"Test set:  {X_test.shape[0]} days (Crash rate: {y_test_cls.mean()*100:.2f}%)")

# Preprocessing: Fit Standard Scaler strictly on Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

print("--- 5. Training and Evaluating Supervised Regression Models ---")
# 1. OLS Multiple Linear Regression
ols_reg = LinearRegression()
ols_reg.fit(X_train_scaled, y_train_reg)
y_pred_reg_ols = ols_reg.predict(X_test_scaled)

# 2. Polynomial Regression (Degree 2 with Ridge Regularization)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled[:, :6]) # Use top 6 core features to avoid curse of dimensionality
X_test_poly = poly.transform(X_test_scaled[:, :6])
ridge_poly = Ridge(alpha=100.0)
ridge_poly.fit(X_train_poly, y_train_reg)
y_pred_reg_poly = ridge_poly.predict(X_test_poly)

reg_results = {
    'OLS Multiple Regression': {
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg_ols),
        'MSE': mean_squared_error(y_test_reg, y_pred_reg_ols),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_ols)),
        'R2': r2_score(y_test_reg, y_pred_reg_ols)
    },
    'Polynomial Regression (Deg 2 + Ridge)': {
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg_poly),
        'MSE': mean_squared_error(y_test_reg, y_pred_reg_poly),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_poly)),
        'R2': r2_score(y_test_reg, y_pred_reg_poly)
    }
}
df_reg_metrics = pd.DataFrame(reg_results).T
print("\n=== Supervised Regression Performance (Predicting Forward Continuous Drawdown) ===")
print(df_reg_metrics.to_string())

print("--- 6. Training and Evaluating Supervised Classification Models ---")
classifiers = {
    'Baseline (Majority Class)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression (L2)': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
    'Gaussian Naive Bayes': GaussianNB(var_smoothing=1e-2),
    'K-Nearest Neighbors (k=7)': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Decision Tree (Pruned CART)': DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight='balanced', random_state=42),
    'Support Vector Machine (RBF)': SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    'Random Forest (Bagging Ensemble)': RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=42),
    'Multi-Layer Perceptron (Neural Net)': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=400, early_stopping=True, random_state=42),
    'Extreme Gradient Boosting (XGBoost)': XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.03, scale_pos_weight=(len(y_train_cls) - sum(y_train_cls))/sum(y_train_cls), eval_metric='logloss', random_state=42)
}

cls_results = []
pred_probs = {}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train_cls)
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.split()[0].lower()}.joblib"))
    
    y_pred = clf.predict(X_test_scaled)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = np.zeros(len(y_test_cls))
    pred_probs[name] = y_prob
    
    acc = accuracy_score(y_test_cls, y_pred)
    prec = precision_score(y_test_cls, y_pred, zero_division=0)
    rec = recall_score(y_test_cls, y_pred, zero_division=0)
    f1 = f1_score(y_test_cls, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_cls, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
    pr_auc = average_precision_score(y_test_cls, y_prob) if len(np.unique(y_prob)) > 1 else y_test_cls.mean()
    brier = brier_score_loss(y_test_cls, y_prob)
    
    cls_results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'Brier Score': brier
    })

df_cls_metrics = pd.DataFrame(cls_results).set_index('Model')
print("\n=== Out-of-Sample Classification Performance on NIFTY 50 (2022–2025) ===")
print(df_cls_metrics.round(4).to_string())
df_cls_metrics.to_csv(os.path.join(PROC_DIR, 'classification_metrics.csv'))

print("--- 7. Feature Importance & Interpretability ---")
rf_model = classifiers['Random Forest (Bagging Ensemble)']
feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feat_imp.to_csv(os.path.join(PROC_DIR, 'feature_importances.csv'))
print("\nTop 10 Most Predictive Features for Market Crashes (Random Forest):")
print(feat_imp.head(10).round(4).to_string())

print("--- 8. Principal Component Analysis (Dimensionality Reduction) ---")
pca = PCA(n_components=10)
pca.fit(X_train_scaled)
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)
print(f"\nPCA: 10 components explain {cum_var[-1]*100:.2f}% of total feature variance")
for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
    print(f"  PC{i:2d}: {ev*100:5.2f}% (Cumulative: {cv*100:5.2f}%)")

print("--- 9. Unsupervised Learning: Stock Downside Risk Profiling ---")
stock_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.parquet') and f not in ['NIFTY_50.parquet', 'INDIA_VIX.parquet', 'SP500.parquet', 'USD_INR.parquet', 'CRUDE_OIL.parquet', 'GOLD.parquet']]

stock_profiles = []
nifty_ret = df_merged['NIFTY_Close'].pct_change().dropna()

for sf in stock_files:
    s_name = sf.replace('.parquet', '')
    sdf = pd.read_parquet(os.path.join(RAW_DIR, sf))
    sdf.index = pd.to_datetime(sdf.index).tz_localize(None).normalize()
    sc = sdf['Close'].reindex(calendar).ffill().dropna()
    s_ret = sc.pct_change().dropna()
    
    # 1. Maximum Drawdown (Full History)
    cum_ret = (1 + s_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_mdd = abs(drawdown.min())
    
    # 2. Downside Deviation (Semi-variance below 0)
    neg_ret = s_ret[s_ret < 0]
    downside_dev = np.sqrt((neg_ret**2).mean()) * np.sqrt(252)
    
    # 3. 99% Daily Value at Risk (VaR)
    var_99 = abs(np.percentile(s_ret, 1))
    
    # 4. Beta relative to NIFTY 50
    aligned = pd.concat([s_ret, nifty_ret], axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    nifty_var = np.var(aligned.iloc[:, 1])
    beta = cov / (nifty_var + 1e-9)
    
    # 5. Annualized Volatility
    ann_vol = s_ret.std() * np.sqrt(252)
    
    stock_profiles.append({
        'Stock': s_name,
        'Max_Drawdown': max_mdd,
        'Downside_Dev': downside_dev,
        'VaR_99': var_99,
        'Market_Beta': beta,
        'Annual_Vol': ann_vol
    })

df_stocks = pd.DataFrame(stock_profiles).set_index('Stock')
print("\n=== Equity Downside Risk Feature Matrix ===")
print(df_stocks.round(4).to_string())

# Cluster Stocks into 3 Downside Risk Profiles using K-Means and Hierarchical Clustering
stock_scaler = StandardScaler()
X_stocks_scaled = stock_scaler.fit_transform(df_stocks)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
df_stocks['KMeans_Cluster'] = kmeans.fit_predict(X_stocks_scaled)
sil_kmeans = silhouette_score(X_stocks_scaled, df_stocks['KMeans_Cluster'])
print(f"\nK-Means Clustering Silhouette Score: {sil_kmeans:.4f}")

# Agglomerative Hierarchical Linkage
hier_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
df_stocks['Hierarchical_Cluster'] = hier_ward.fit_predict(X_stocks_scaled)
sil_hier = silhouette_score(X_stocks_scaled, df_stocks['Hierarchical_Cluster'])
print(f"Hierarchical Clustering (Ward) Silhouette Score: {sil_hier:.4f}")

# Map clusters to economic risk tiers based on average beta and max drawdown
cluster_risk_rank = df_stocks.groupby('KMeans_Cluster')[['Market_Beta', 'Max_Drawdown']].mean().sum(axis=1).sort_values().index
tier_map = {cluster_risk_rank[0]: 'Tier 1: Defensive / Low Risk',
            cluster_risk_rank[1]: 'Tier 2: Moderate Risk',
            cluster_risk_rank[2]: 'Tier 3: High Downside Risk'}
df_stocks['Risk_Tier'] = df_stocks['KMeans_Cluster'].map(tier_map)
df_stocks.to_csv(os.path.join(PROC_DIR, 'stock_downside_clusters.csv'))
print("\n=== Stock Downside Risk Profiling Results ===")
print(df_stocks[['Market_Beta', 'Max_Drawdown', 'Downside_Dev', 'Risk_Tier']].to_string())

print("--- 10. Association Rule Mining (Pre-Crash Pattern Detection) ---")
# Discretize continuous market stress indicators into boolean transaction items
df_trans = pd.DataFrame(index=df_clean.index)
df_trans['High_India_VIX'] = (df_clean['VIX_Level'] > df_clean['VIX_Level'].quantile(0.80)).astype(int)
df_trans['VIX_Spike_5d'] = (df_clean['VIX_Return_5d'] > 0.15).astype(int)
df_trans['NIFTY_Oversold_RSI'] = (df_clean['RSI_14'] < 35).astype(int)
df_trans['Below_SMA_50'] = (df_clean['SMA_Ratio_50'] < 0.98).astype(int)
df_trans['USD_INR_Depreciation'] = (df_clean['USD_INR_Return_5d'] > 0.008).astype(int)
df_trans['SP500_Selloff'] = (df_clean['SP500_Return_5d'] < -0.02).astype(int)
df_trans['Severe_Crash_Next_21d'] = df_clean['Target_Crash_Binary']

frequent_itemsets = apriori(df_trans, min_support=0.03, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.35)
crash_rules = rules[rules['consequents'].apply(lambda x: 'Severe_Crash_Next_21d' in x)].sort_values(by='lift', ascending=False)
print("\n=== Top Discovered Pre-Crash Association Rules ===")
for idx, r in crash_rules.head(5).iterrows():
    ants = list(r['antecedents'])
    print(f"Rule: {ants} => Crash (Support: {r['support']:.3f}, Confidence: {r['confidence']:.3f}, Lift: {r['lift']:.3f})")
crash_rules.to_csv(os.path.join(PROC_DIR, 'association_rules.csv'))

print("--- 11. Generating Comprehensive Academic Visualizations ---")
# Figure 1: Historical NIFTY 50 Price, Drawdowns, and Identified Crash Regimes
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(df_clean.index, df_clean['Target_Forward_MDD']*100, color='#1f77b4', lw=1.2, label='NIFTY 50 Forward 21-Day MDD (%)')
axes[0].axhline(CRASH_THRESHOLD*100, color='#d62728', linestyle='--', lw=1.5, label=f'Crash Threshold ({CRASH_THRESHOLD*100}%)')
axes[0].set_title('A. NIFTY 50 Forward 21-Day Drawdown & Defined Crash Episodes (2010–2025)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Forward Drawdown (%)', fontsize=10)
axes[0].legend(loc='lower left', frameon=True)

axes[1].plot(df_clean.index, df_clean['VIX_Level'], color='#ff7f0e', lw=1.2, label='India VIX Index')
axes[1].axhline(df_clean['VIX_Level'].quantile(0.90), color='#7f7f7f', linestyle=':', label='90th Percentile VIX Stress')
axes[1].set_title('B. Domestic Market Implied Volatility (India VIX)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('India VIX Level', fontsize=10)
axes[1].legend(loc='upper right', frameon=True)

# Highlight identified crash periods
axes[2].fill_between(df_clean.index, 0, df_clean['Target_Crash_Binary'], color='#d62728', alpha=0.6, label='Crash Event Active (Y=1)')
axes[2].axvline(pd.to_datetime('2020-01-01'), color='black', linestyle='--', label='Train / Validation Split')
axes[2].axvline(pd.to_datetime('2022-01-01'), color='green', linestyle='--', label='Validation / Test Split')
axes[2].set_title('C. Binary Target Regime with Chronological Walk-Forward Split Boundaries', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Binary State', fontsize=10)
axes[2].set_ylim(-0.1, 1.2)
axes[2].legend(loc='upper right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_historical_drawdowns_and_targets.png'), dpi=300)
plt.close()

# Figure 2: ROC and Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
for (name, prob), color in zip(pred_probs.items(), colors):
    if len(np.unique(prob)) > 1:
        fpr, tpr, _ = roc_curve(y_test_cls, prob)
        roc_score = roc_auc_score(y_test_cls, prob)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.3f})", color=color, lw=1.5)
        
        prec, rec, _ = precision_recall_curve(y_test_cls, prob)
        pr_score = average_precision_score(y_test_cls, prob)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.3f})", color=color, lw=1.5)

ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_title('A. Out-of-Sample ROC Curves (2022–2025)', fontsize=12, fontweight='bold')
ax1.set_xlabel('False Positive Rate', fontsize=10)
ax1.set_ylabel('True Positive Rate', fontsize=10)
ax1.legend(loc='lower right', fontsize=8, frameon=True)

baseline_rate = y_test_cls.mean()
ax2.axhline(baseline_rate, color='k', linestyle='--', label=f'Baseline Rate ({baseline_rate:.3f})')
ax2.set_title('B. Out-of-Sample Precision-Recall Curves (Rare Event Benchmark)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Recall (Sensitivity)', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.legend(loc='upper right', fontsize=8, frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_roc_pr_curves.png'), dpi=300)
plt.close()

# Figure 3: Top Feature Importance & PCA Scree Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
feat_imp.head(10).plot(kind='barh', ax=ax1, color='#2ca02c', edgecolor='black')
ax1.set_title('A. Random Forest Top 10 Predictor Importances (MDI)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Gini Impurity Importance', fontsize=10)
ax1.invert_yaxis()

ax2.bar(range(1, 11), exp_var*100, color='#1f77b4', alpha=0.7, label='Individual Explained Variance (%)')
ax2.plot(range(1, 11), cum_var*100, color='#d62728', marker='o', lw=2, label='Cumulative Explained Variance (%)')
ax2.axhline(85, color='gray', linestyle=':', label='85% Variance Threshold')
ax2.set_title('B. PCA Scree Plot & Cumulative Variance', fontsize=12, fontweight='bold')
ax2.set_xlabel('Principal Component', fontsize=10)
ax2.set_ylabel('Variance Explained (%)', fontsize=10)
ax2.set_xticks(range(1, 11))
ax2.legend(loc='center right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_features_and_pca.png'), dpi=300)
plt.close()

# Figure 4: Hierarchical Dendrogram and Stock Risk Clustering Scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Dendrogram
Z = linkage(X_stocks_scaled, method='ward')
dendrogram(Z, labels=df_stocks.index.tolist(), orientation='top', ax=ax1, leaf_font_size=9)
ax1.set_title('A. Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cophentic Distance', fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# Cluster scatter plot
tier_palette = {'Tier 1: Defensive / Low Risk': '#2ca02c', 'Tier 2: Moderate Risk': '#1f77b4', 'Tier 3: High Downside Risk': '#d62728'}
for tier, group in df_stocks.groupby('Risk_Tier'):
    ax2.scatter(group['Market_Beta'], group['Max_Drawdown']*100, label=tier, s=120, color=tier_palette[tier], edgecolors='black', alpha=0.85)
    for stock, row in group.iterrows():
        ax2.annotate(stock, (row['Market_Beta']+0.015, row['Max_Drawdown']*100+0.5), fontsize=8)

ax2.set_title('B. Constituent Equities Downside Risk Map', fontsize=12, fontweight='bold')
ax2.set_xlabel('Systematic Market Beta (vs NIFTY 50)', fontsize=10)
ax2.set_ylabel('Historical Maximum Drawdown (%)', fontsize=10)
ax2.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_stock_clustering_profiles.png'), dpi=300)
plt.close()

# Figure 5: Confusion Matrices for Top Models
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
eval_models = ['Logistic Regression (L2)', 'Support Vector Machine (RBF)', 'Random Forest (Bagging Ensemble)']
for i, m_name in enumerate(eval_models):
    y_pred = classifiers[m_name].predict(X_test_scaled)
    cm = confusion_matrix(y_test_cls, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Normal', 'Crash'], yticklabels=['Normal', 'Crash'])
    axes[i].set_title(f"{m_name}\nPrecision: {precision_score(y_test_cls, y_pred):.2f} | Recall: {recall_score(y_test_cls, y_pred):.2f}", fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Predicted Label', fontsize=9)
    axes[i].set_ylabel('Actual Label', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_confusion_matrices.png'), dpi=300)
plt.close()

print("\n--- ALL EXPERIMENTAL RUNS, METRICS, MODELS, AND FIGURES COMPLETED SUCCESSFULLY! ---")
