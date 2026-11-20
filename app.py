import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================================================
# PAGE CONFIG & CUSTOM CSS STYLING
# =========================================================
st.set_page_config(
    page_title='NIFTY 50 Crash Prediction & Risk Profiling',
    page_icon='📉',
    layout='wide',
    initial_sidebar_state='expanded'
)



st.markdown('''
<style>
    .main-header {
        background: linear-gradient(135deg, #1B365D 0%, #2E5B88 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(27, 54, 93, 0.18);
    }
    .main-header h1 {
        color: white;
        font-size: 1.85rem;
        margin: 0 0 0.3rem 0;
        font-weight: 700;
    }
    .main-header p {
        color: #E2E8F0;
        font-size: 0.95rem;
        margin: 0;
    }

    .section-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2E5B88;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #1E293B;
    }
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #38BDF8;
        padding: 0.85rem 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .section-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.93rem;
        color: #E2E8F0;
    }
</style>
''', unsafe_allow_html=True)

sns.set_style('whitegrid')
plt.rcParams.update({'axes.edgecolor': '#CBD5E1', 'axes.linewidth': 0.8})

# =========================================================
# LOAD DATA & TRAINED MODELS BUNDLE
# =========================================================
bundle = joblib.load('crash_bundle.pkl')
models = bundle['models']
scaler = bundle['scaler']
feature_cols = bundle['feature_cols']
medians = bundle['medians']
X_test_scaled = bundle['X_test_scaled']
y_test_class = bundle['y_test_class']
cls_results = bundle['cls_results']
reg_results = bundle['reg_results']
stock_df = bundle['stock_df']
rules_df = bundle['rules']
df_full = bundle['df_full']

# =========================================================
# LEFT SIDEBAR NAVIGATION & CONTROLS
# =========================================================
with st.sidebar:
    st.markdown('### 📉 NIFTY 50 Risk Analytics')
    st.caption('INT234 Predictive Analytics Project')
    st.divider()

    page = st.radio(
        'Navigation Menu',
        [
            '1. Live Crash Predictor',
            '2. Market Trends & EDA',
            '3. Model Benchmarks',
            '4. Stock Risk Clusters & PCA',
            '5. Pre-Crash Rules (Apriori)'
        ]
    )

    st.divider()
    st.markdown('**Date Range Filter**')
    years = sorted(df_full.index.year.unique())
    yr_range = st.slider(
        'Select Analysis Years',
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=(int(years[0]), int(years[-1]))
    )

    st.divider()
    st.markdown('**Project Information**')
    st.info(
        '**Student:** Liza Rawat\n\n'
        '**Reg. No:** 12409771\n\n'
        '**Course:** INT234 (Predictive Analytics)\n\n'
        '**University:** Lovely Professional University'
    )

# =========================================================
# TOP HEADER BANNER & KPI METRICS
# =========================================================
st.markdown('''
<div class="main-header">
    <h1>Stock Market Crash Prediction & Investor Risk Profiling</h1>
    <p>Multi-Algorithm Machine Learning Framework on Indian Equities (NIFTY 50 & India VIX)</p>
</div>
''', unsafe_allow_html=True)

best_f1_row = cls_results.sort_values(by='F1_Score', ascending=False).iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric('Total Trading Sessions', f'{len(df_full):,}', f'{years[0]}–{years[-1]}')
k2.metric('Crash Warning Rate', f"{df_full['Crash'].mean()*100:.1f}%", 'Rare Event Class')
k3.metric('Top F1-Score Classifier', f"{best_f1_row['Algorithm']}", f"F1: {best_f1_row['F1_Score']:.4f}")
k4.metric('Profiled NIFTY Stocks', f'{len(stock_df)} Equities', '3 Risk Tiers')

st.markdown('---')

# =========================================================
# PAGE 1: LIVE CRASH RISK PREDICTOR
# =========================================================
if page == '1. Live Crash Predictor':
    st.subheader('Live NIFTY 50 Market Crash Risk Simulator')
    st.markdown(
        '<div class="section-card">Test custom market conditions or select a one-click stress scenario below '
        'to evaluate forward market crash risk in real time.</div>',
        unsafe_allow_html=True
    )

    # Preset scenarios for easy viva demonstration
    preset = st.radio(
        'Quick Market Scenario Preset',
        ['Normal Market (Median)', 'Elevated Fear (VIX Spike)', 'Severe Crash Stress'],
        horizontal=True
    )

    if preset == 'Normal Market (Median)':
        def_vals = (
            round(medians['Daily_Return']*100, 2),
            round(medians['Return_5d']*100, 2),
            round(medians['Return_21d']*100, 2),
            round(medians['Volatility_20d']*100, 2),
            round(medians['SMA_50_Ratio'], 3),
            round(medians['VIX'], 2)
        )
    elif preset == 'Elevated Fear (VIX Spike)':
        def_vals = (-1.80, -4.20, -6.50, 1.85, 0.955, 26.50)
    else:
        def_vals = (-4.50, -9.50, -16.00, 3.40, 0.860, 48.00)

    prob_models = [name for name, m in models.items() if hasattr(m, 'predict_proba')]
    default_idx = prob_models.index('Random Forest') if 'Random Forest' in prob_models else 0
    selected_model_name = st.selectbox('Select Prediction Model', prob_models, index=default_idx)

    c1, c2, c3 = st.columns(3)
    with c1:
        daily_ret = st.number_input('Daily Return (%)', min_value=-15.0, max_value=15.0, value=float(def_vals[0]), step=0.1) / 100.0
        ret_5d = st.number_input('5-Day Return (%)', min_value=-30.0, max_value=30.0, value=float(def_vals[1]), step=0.2) / 100.0
    with c2:
        ret_21d = st.number_input('21-Day Return (%)', min_value=-50.0, max_value=50.0, value=float(def_vals[2]), step=0.5) / 100.0
        vol_20d = st.number_input('20-Day Volatility (%)', min_value=0.1, max_value=10.0, value=float(def_vals[3]), step=0.1) / 100.0
    with c3:
        sma_ratio = st.number_input('Price / 50-Day SMA Ratio', min_value=0.60, max_value=1.40, value=float(def_vals[4]), step=0.01)
        vix_val = st.number_input('India VIX (Fear Index)', min_value=8.0, max_value=90.0, value=float(def_vals[5]), step=0.5)

    if st.button('Run Crash Risk Prediction', type='primary', use_container_width=True):
        input_df = pd.DataFrame([{
            'Daily_Return': daily_ret,
            'Return_5d': ret_5d,
            'Return_21d': ret_21d,
            'Volatility_20d': vol_20d,
            'SMA_50_Ratio': sma_ratio,
            'VIX': vix_val
        }])[feature_cols]

        input_scaled = scaler.transform(input_df)
        clf = models[selected_model_name]
        prob = clf.predict_proba(input_scaled)[0][1]
        pred = clf.predict(input_scaled)[0]

        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.metric(f'Crash Risk Probability ({selected_model_name})', f'{prob*100:.2f}%')
            st.progress(min(float(prob), 1.0))
        with res_c2:
            if pred == 1 or prob >= 0.40:
                st.error(
                    '**ELEVATED CRASH RISK WARNING**\n\n'
                    'Actionable Guidance: Hedge index exposure and rotate capital from High-Risk Cyclicals '
                    '(e.g., TATASTEEL) into Tier 1 Defensive stocks (e.g., HINDUNILVR, TCS, HDFCBANK).'
                )
            else:
                st.success(
                    '**NORMAL MARKET REGIME**\n\n'
                    'Actionable Guidance: Volatility and trend indicators remain within historical normal bounds. '
                    'Standard diversified allocation can be maintained.'
                )

# =========================================================
# PAGE 2: MARKET TRENDS & EXPLORATORY DATA ANALYSIS
# =========================================================
elif page == '2. Market Trends & EDA':
    st.subheader(f'NIFTY 50 Price Trajectory & India VIX Fear Index ({yr_range[0]}–{yr_range[1]})')
    st.markdown(
        '<div class="section-card">Use the Year Range slider in the left sidebar to zoom into specific market cycles '
        'such as the 2020 COVID-19 crash or post-2022 inflation volatility.</div>',
        unsafe_allow_html=True
    )

    df_view = df_full[(df_full.index.year >= yr_range[0]) & (df_full.index.year <= yr_range[1])]

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    ax1.plot(df_view.index, df_view['Nifty_Close'], label='NIFTY 50 Close', color='#1B365D', lw=1.6)
    ax1.plot(df_view.index, df_view['Nifty_Close'].rolling(50).mean(), label='50-Day SMA', color='#FF7F0E', linestyle='--', lw=1.2)
    ax1.plot(df_view.index, df_view['Nifty_Close'].rolling(200).mean(), label='200-Day SMA', color='#2CA02C', linestyle=':', lw=1.4)
    ax1.set_title('NIFTY 50 Closing Level with 50-Day & 200-Day Moving Averages', fontweight='bold', color='#1B365D')
    ax1.set_ylabel('Index Price (INR)')
    ax1.legend(loc='upper left', frameon=True)

    ax2.fill_between(df_view.index, df_view['VIX'], color='#D62728', alpha=0.18)
    ax2.plot(df_view.index, df_view['VIX'], label='India VIX (Fear Index)', color='#D62728', lw=1.3)
    ax2.axhline(df_view['VIX'].mean(), color='#1E293B', linestyle='--', lw=1.1, label=f"Period Mean VIX ({df_view['VIX'].mean():.1f})")
    ax2.set_title('India VIX Volatility Spikes (Market Panic Gauge)', fontweight='bold', color='#1B365D')
    ax2.set_ylabel('VIX Level')
    ax2.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown('---')
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.subheader('Daily Returns Distribution (Fat Tails)')
        fig2, ax = plt.subplots(figsize=(6.2, 4.6))
        sns.histplot(df_view['Daily_Return'], bins=60, kde=True, color='#2E5B88', stat='density', ax=ax)
        ax.axvline(0, color='#1E293B', linestyle='--', lw=1)
        ax.axvline(df_view['Daily_Return'].mean() - 2*df_view['Daily_Return'].std(), color='#D62728', linestyle='--', lw=1.4, label='2-Std Dev Extreme Dip')
        ax.set_xlabel('Daily Percentage Return')
        ax.set_ylabel('Density')
        ax.set_title('Leptokurtic Fat-Tail Distribution', fontweight='bold', color='#1B365D')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with col_eda2:
        st.subheader('Feature Correlation Heatmap')
        corr_cols = ['Daily_Return', 'Return_5d', 'Return_21d', 'Volatility_20d', 'SMA_50_Ratio', 'VIX', 'Crash']
        fig3, ax = plt.subplots(figsize=(6.2, 4.6))
        sns.heatmap(df_view[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax, cbar=True, linewidths=0.5)
        ax.set_title('Cross-Feature Correlation Matrix', fontweight='bold', color='#1B365D')
        plt.tight_layout()
        st.pyplot(fig3)

# =========================================================
# PAGE 3: MODEL BENCHMARKS & DIAGNOSTICS
# =========================================================
elif page == '3. Model Benchmarks':
    st.subheader('Supervised Classification Benchmark (7 Algorithms)')
    st.dataframe(cls_results, use_container_width=True, hide_index=True)

    fig4, ax = plt.subplots(figsize=(11.5, 4.3))
    x = np.arange(len(cls_results))
    w = 0.25
    ax.bar(x - w, cls_results['Accuracy'], w, label='Accuracy', color='#1B365D')
    ax.bar(x, cls_results['Recall'], w, label='Recall (Crisis Sensitivity)', color='#2CA02C')
    ax.bar(x + w, cls_results['F1_Score'], w, label='F1-Score', color='#FF7F0E')
    ax.set_xticks(x)
    ax.set_xticklabels(cls_results['Algorithm'], rotation=18, ha='right')
    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_ylim(0, 1.08)
    ax.set_title('Algorithm Comparison: Accuracy vs. Recall vs. F1-Score', fontweight='bold', color='#1B365D')
    ax.legend(frameon=True)
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown('---')
    st.subheader('Interactive Confusion Matrix & Feature Attribution')
    cm_model_name = st.selectbox('Select Classifier to Inspect Confusion Matrix', list(models.keys()), index=list(models.keys()).index('Random Forest'))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        clf_cm = models[cm_model_name]
        if cm_model_name == 'Naive Bayes':
            preds_cm = (clf_cm.predict_proba(X_test_scaled)[:, 1] > 0.15).astype(int)
        else:
            preds_cm = clf_cm.predict(X_test_scaled)
        cm = confusion_matrix(y_test_class, preds_cm)

        fig5, ax = plt.subplots(figsize=(5.6, 4.3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 13, 'weight': 'bold'},
                    xticklabels=['Pred Normal', 'Pred Crash'],
                    yticklabels=['Actual Normal', 'Actual Crash'], ax=ax)
        ax.set_title(f'{cm_model_name}: Out-of-Sample Confusion Matrix', fontweight='bold', color='#1B365D')
        plt.tight_layout()
        st.pyplot(fig5)

    with col_m2:
        rf_model = models['Random Forest']
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
        fig6, ax = plt.subplots(figsize=(5.6, 4.3))
        importances.plot(kind='barh', color='#2E5B88', ax=ax)
        ax.set_title('Random Forest: Predictor Importance Ranking', fontweight='bold', color='#1B365D')
        ax.set_xlabel('Gini Importance Weight')
        plt.tight_layout()
        st.pyplot(fig6)

    st.markdown('---')
    st.subheader('Continuous Forward Return Regression Benchmark')
    st.dataframe(reg_results, use_container_width=True, hide_index=True)

# =========================================================
# PAGE 4: STOCK RISK CLUSTERING & PCA
# =========================================================
elif page == '4. Stock Risk Clusters & PCA':
    st.subheader('Unsupervised Constituent Equity Downside Risk Profiling')
    st.markdown(
        '<div class="section-card">Bridging macro crash warnings with micro portfolio allocation: '
        'K-Means and Ward Hierarchical Clustering segment blue-chip NSE stocks into Defensive, Moderate, and High-Risk cohorts.</div>',
        unsafe_allow_html=True
    )
    st.dataframe(stock_df.reset_index(), use_container_width=True, hide_index=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig7, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.scatter(stock_df['Annual_Volatility'], stock_df['Annual_Return'],
                   c=stock_df['Cluster'], cmap='viridis', s=200, edgecolors='#1E293B', lw=1.2)
        for ticker, row in stock_df.iterrows():
            ax.annotate(ticker.replace('.NS', ''), (row['Annual_Volatility'] + 0.004, row['Annual_Return'] + 0.004), fontsize=9, fontweight='bold')
        ax.set_title('K-Means Stock Risk Tiers (Volatility vs. Return)', fontweight='bold', color='#1B365D')
        ax.set_xlabel('Annualized Volatility (Risk)')
        ax.set_ylabel('Annualized Return')
        plt.tight_layout()
        st.pyplot(fig7)

    with col_c2:
        fig8, ax = plt.subplots(figsize=(6.2, 4.6))
        Z = linkage(stock_df[['Annual_Return', 'Annual_Volatility']], method='ward')
        dendrogram(Z, labels=[t.replace('.NS', '') for t in stock_df.index], ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontweight='bold', color='#1B365D')
        ax.set_ylabel('Euclidean Distance')
        plt.tight_layout()
        st.pyplot(fig8)

    st.markdown('---')
    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.subheader('K-Means Elbow Curve')
        inertias = []
        K_range = range(1, min(7, len(stock_df) + 1))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(stock_df[['Annual_Return', 'Annual_Volatility']])
            inertias.append(km.inertia_)
        fig9, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.plot(list(K_range), inertias, marker='o', color='#1B365D', lw=2.2, markersize=7)
        ax.set_title('Elbow Method for Optimal Cluster Count (k = 3)', fontweight='bold', color='#1B365D')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Within-Cluster Inertia')
        plt.tight_layout()
        st.pyplot(fig9)

    with col_c4:
        st.subheader('PCA 2D Market Regime Projection')
        X_all_scaled = scaler.transform(df_full[feature_cols])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_scaled)
        var_pct = pca.explained_variance_ratio_.sum() * 100

        fig10, ax = plt.subplots(figsize=(6.2, 4.3))
        normal_mask = df_full['Crash'] == 0
        crash_mask = df_full['Crash'] == 1
        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], c='#2E5B88', alpha=0.45, s=18, label='Normal Days')
        ax.scatter(X_pca[crash_mask, 0], X_pca[crash_mask, 1], c='#D62728', alpha=0.85, s=28, label='Crash Warning Days')
        ax.set_title(f'PCA 2D Projection ({var_pct:.1f}% Total Variance Retained)', fontweight='bold', color='#1B365D')
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig10)

# =========================================================
# PAGE 5: PRE-CRASH ASSOCIATION RULES (APRIORI)
# =========================================================
elif page == '5. Pre-Crash Rules (Apriori)':
    st.subheader('Explainable Pre-Crash Association Rules (Apriori Algorithm)')
    st.markdown(
        '<div class="section-card">Unlike black-box classifiers, Apriori uncovers transparent IF-THEN risk rules. '
        'A Lift greater than 1.0 confirms that co-occurring market stress factors significantly multiply crash probability.</div>',
        unsafe_allow_html=True
    )
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    if len(rules_df) > 0:
        plot_rules = rules_df.head(6).copy()
        plot_rules['Rule'] = plot_rules['antecedents'] + '  ==>  ' + plot_rules['consequents']
        fig11, ax = plt.subplots(figsize=(10.5, 4.3))
        ax.barh(plot_rules['Rule'][::-1], plot_rules['lift'][::-1], color='#D62728', alpha=0.88)
        ax.axvline(1.0, color='#1E293B', linestyle='--', lw=1.3, label='Random Baseline (Lift = 1.0)')
        ax.set_xlabel('Lift (Crash Risk Multiplier Over Baseline)')
        ax.set_title('Top Pre-Crash Stress Patterns Ranked by Lift', fontweight='bold', color='#1B365D')
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig11)
