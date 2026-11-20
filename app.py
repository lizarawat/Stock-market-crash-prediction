import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title='NIFTY 50 Crash Prediction & Risk Analytics',
    page_icon='📉',
    layout='wide',
    initial_sidebar_state='expanded'
)

# =========================================================
# MODERN FINTECH DASHBOARD CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Header */
    .hero-banner {
        background: radial-gradient(120% 120% at 50% 10%, #1E293B 0%, #0F172A 70%, #020617 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.6);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        color: #F8FAFC;
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.96rem;
        font-weight: 500;
        margin: 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.6rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 8px #34D399;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 3px solid #38BDF8 !important;
        padding: 1rem 1.2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    /* Glass Cards */
    .glass-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .card-title {
        color: #F8FAFC;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-caption {
        color: #94A3B8;
        font-size: 0.86rem;
        margin-bottom: 0.8rem;
    }

    /* Action Decision Badges */
    .alert-box {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.8rem;
        font-size: 0.94rem;
        line-height: 1.5;
    }
    .alert-safe {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #A7F3D0;
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FDE68A;
    }
    .alert-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.45);
        color: #FECACA;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
        background-color: rgba(56, 189, 248, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to apply dark fintech styling cleanly to any figure
def apply_plot_theme(fig, height=350, **kwargs):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#CBD5E1', size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        **kwargs
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)')
    return fig

# =========================================================
# LOAD DATA & MODELS BUNDLE
# =========================================================
@st.cache_resource
def load_bundle():
    return joblib.load('crash_bundle.pkl')

bundle = load_bundle()
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
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.markdown("### ⚡ **Control Deck**")
    st.caption("Quantitative Risk Engineering Lab")
    
    view_mode = st.radio(
        "Dashboard Presentation Mode",
        ["Unified All-in-One Dashboard", "Tabbed Deep-Dive View"],
        index=0
    )
    st.divider()

    st.markdown("**Historical Date Filter**")
    years = sorted(df_full.index.year.unique())
    yr_range = st.slider(
        "Analysis Period",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=(int(years[0]), int(years[-1]))
    )

    st.divider()
    st.markdown("**Stress Scenario Presets**")
    preset = st.selectbox(
        "One-Click Market Simulation",
        [
            "Tranquil Market (Historical Median)",
            "Rising Fear (VIX Volatility Surge)",
            "Black Swan Crash (COVID / Lehman Shock)"
        ]
    )

    prob_models = [name for name, m in models.items() if hasattr(m, 'predict_proba')]
    default_clf_idx = prob_models.index('Random Forest') if 'Random Forest' in prob_models else 0
    selected_model_name = st.selectbox("Predictor Architecture", prob_models, index=default_clf_idx)

    cm_model_name = st.selectbox(
        "Diagnostic Model (Confusion Matrix)",
        list(models.keys()),
        index=list(models.keys()).index('Random Forest')
    )

    st.divider()
    st.markdown("**Researcher & University Profile**")
    st.info(
        "**Researcher:** Liza Rawat\n\n"
        "**Reg. No:** 12409771\n\n"
        "**Course:** INT234 (Predictive Analytics)\n\n"
        "**School:** Lovely Professional University"
    )

# =========================================================
# TOP HERO BANNER & KPI ROW
# =========================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">NIFTY 50 Market Crash Prediction & Downside Risk Profiling</div>
    <div class="hero-subtitle">Dual-Scale Machine Learning Architecture on Indian Benchmark Equities & Implied Volatility (India VIX)</div>
    <div class="status-pill"><span class="status-dot"></span> Quantitative Model Suite Active | 10-Year In-Sample History</div>
</div>
""", unsafe_allow_html=True)

best_f1_row = cls_results.sort_values(by='F1_Score', ascending=False).iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Trading Sessions", f"{len(df_full):,}", f"{years[0]}–{years[-1]}")
k2.metric("Crisis Exceedance Rate", f"{df_full['Crash'].mean()*100:.1f}%", "Rare Event Class (<5%)")
k3.metric("Top F1 Classifier", f"{best_f1_row['Algorithm']}", f"F1: {best_f1_row['F1_Score']:.4f}")
k4.metric("Constituent Tiers", f"{len(stock_df)} Equities", "3 Downside Clusters")

st.markdown("<br>", unsafe_allow_html=True)

# Preset Values
if preset == "Tranquil Market (Historical Median)":
    def_vals = (
        round(medians['Daily_Return']*100, 2),
        round(medians['Return_5d']*100, 2),
        round(medians['Return_21d']*100, 2),
        round(medians['Volatility_20d']*100, 2),
        round(medians['SMA_50_Ratio'], 3),
        round(medians['VIX'], 2)
    )
elif preset == "Rising Fear (VIX Volatility Surge)":
    def_vals = (-1.80, -4.20, -6.50, 1.85, 0.955, 26.50)
else:
    def_vals = (-4.50, -9.50, -16.00, 3.40, 0.860, 48.00)

df_view = df_full[(df_full.index.year >= yr_range[0]) & (df_full.index.year <= yr_range[1])]

# =========================================================
# REUSABLE SECTION FUNCTIONS
# =========================================================
def render_live_predictor():
    st.markdown('<div class="card-title">🎯 1. Real-Time Market Crash Risk Simulator & Speedometer Gauge</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Adjust technical market indicators or choose a preset in the sidebar to simulate forward tactical crash probability.</div>', unsafe_allow_html=True)

    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        daily_ret = st.number_input('Daily Return (%)', min_value=-15.0, max_value=15.0, value=float(def_vals[0]), step=0.1) / 100.0
        ret_5d = st.number_input('5-Day Return (%)', min_value=-30.0, max_value=30.0, value=float(def_vals[1]), step=0.2) / 100.0
    with c_in2:
        ret_21d = st.number_input('21-Day Return (%)', min_value=-50.0, max_value=50.0, value=float(def_vals[2]), step=0.5) / 100.0
        vol_20d = st.number_input('20-Day Volatility (%)', min_value=0.1, max_value=10.0, value=float(def_vals[3]), step=0.1) / 100.0
    with c_in3:
        sma_ratio = st.number_input('Price / 50-Day SMA Ratio', min_value=0.60, max_value=1.40, value=float(def_vals[4]), step=0.01)
        vix_val = st.number_input('India VIX (Fear Gauge)', min_value=8.0, max_value=90.0, value=float(def_vals[5]), step=0.5)

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
    prob_pct = prob * 100

    col_gauge, col_decision = st.columns([1.1, 1.4])
    with col_gauge:
        bar_color = "#10B981" if prob_pct < 25 else ("#F59E0B" if prob_pct < 45 else "#EF4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': "%", 'font': {'size': 44, 'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'}},
            title={'text': f"Predicted Crash Risk ({selected_model_name})", 'font': {'size': 14, 'color': '#94A3B8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': bar_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.18)"},
                    {'range': [25, 45], 'color': "rgba(245, 158, 11, 0.18)"},
                    {'range': [45, 75], 'color': "rgba(249, 115, 22, 0.22)"},
                    {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        apply_plot_theme(fig_gauge, height=260)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_decision:
        if prob_pct < 25:
            st.markdown(f"""
            <div class="alert-box alert-safe">
                <b>🟢 TRANQUIL REGIME (Normal Market Conditions)</b><br>
                <b>System Diagnostic:</b> Crash probability is low at <b>{prob_pct:.1f}%</b>. Market momentum and options-implied fear (India VIX) are well within historical safety thresholds.<br>
                <b>Portfolio Directive:</b> Maintain diversified benchmark weights. Capital can remain deployed across Tier 1 Defensive (TCS, HINDUNILVR) and Tier 2 Moderate Growth (INFY, SBIN) holdings without active index hedging.
            </div>
            """, unsafe_allow_html=True)
        elif prob_pct < 45:
            st.markdown(f"""
            <div class="alert-box alert-warning">
                <b>🟡 GUARDED REGIME (Elevated Latent Volatility)</b><br>
                <b>System Diagnostic:</b> Crash probability is elevated at <b>{prob_pct:.1f}%</b>. Volatility clustering or technical price deceleration detected.<br>
                <b>Portfolio Directive:</b> Tighten trailing stop-loss limits on cyclical momentum assets. Suspend aggressive new long positions until India VIX subsides below moving baseline.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box alert-danger">
                <b>🔴 CRITICAL CRASH WARNING (Tactical Defense Triggered)</b><br>
                <b>System Diagnostic:</b> Crash probability has surged to <b>{prob_pct:.1f}%</b>! High-risk confluence of trend exhaustion and acute volatility spikes.<br>
                <b>Portfolio Directive:</b> Immediate capital preservation protocol activated. Liquidate or hedge High-Beta Tier 3 Cyclicals (TATASTEEL), shift 40%+ into Tier 1 Defensive assets (HDFCBANK, HINDUNILVR), or acquire out-of-the-money NIFTY put options.
            </div>
            """, unsafe_allow_html=True)

def render_market_trends():
    st.markdown('<div class="card-title">📈 2. Interactive NIFTY 50 Trajectory & India VIX Volatility Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Dual-axis interactive time series showing index momentum against live options-implied panic. Hover to inspect prices.</div>', unsafe_allow_html=True)

    fig_trends = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('NIFTY 50 Benchmark Close with 50-Day & 200-Day Moving Averages', 'India VIX Fear Index (Options-Implied Market Panic)'),
        row_heights=[0.65, 0.35]
    )

    fig_trends.add_trace(go.Scatter(
        x=df_view.index, y=df_view['Nifty_Close'],
        name='NIFTY 50', line=dict(color='#38BDF8', width=2),
        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.08)'
    ), row=1, col=1)

    fig_trends.add_trace(go.Scatter(
        x=df_view.index, y=df_view['Nifty_Close'].rolling(50).mean(),
        name='50-Day SMA', line=dict(color='#F59E0B', width=1.5, dash='dash')
    ), row=1, col=1)

    fig_trends.add_trace(go.Scatter(
        x=df_view.index, y=df_view['Nifty_Close'].rolling(200).mean(),
        name='200-Day SMA', line=dict(color='#10B981', width=1.5, dash='dot')
    ), row=1, col=1)

    fig_trends.add_trace(go.Scatter(
        x=df_view.index, y=df_view['VIX'],
        name='India VIX', line=dict(color='#EF4444', width=1.5),
        fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.15)'
    ), row=2, col=1)

    mean_vix = df_view['VIX'].mean()
    fig_trends.add_hline(y=mean_vix, line=dict(color='#94A3B8', width=1, dash='dash'),
                         annotation_text=f"Mean VIX ({mean_vix:.1f})", row=2, col=1)

    apply_plot_theme(fig_trends, height=520, hovermode='x unified')
    fig_trends.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig_trends, use_container_width=True)

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown('<div class="card-title">📊 Daily Returns Distribution (Fat Tails)</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            df_view, x='Daily_Return', nbins=70, marginal='box',
            color_discrete_sequence=['#38BDF8'], opacity=0.8
        )
        two_std = df_view['Daily_Return'].mean() - 2*df_view['Daily_Return'].std()
        fig_dist.add_vline(x=two_std, line=dict(color='#EF4444', dash='dash', width=2),
                           annotation_text="Extreme -2σ Shock")
        apply_plot_theme(fig_dist, height=360, bargap=0.05,
                         xaxis_title="Daily Percentage Return", yaxis_title="Session Count")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_eda2:
        st.markdown('<div class="card-title">🔥 Cross-Asset Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ['Daily_Return', 'Return_5d', 'Return_21d', 'Volatility_20d', 'SMA_50_Ratio', 'VIX', 'Crash']
        corr_matrix = df_view[corr_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1
        )
        apply_plot_theme(fig_corr, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_corr, use_container_width=True)

def render_benchmarks():
    st.markdown('<div class="card-title">🤖 3. Supervised Model Benchmarking & Crisis Sensitivity Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Comparison of 7 classification models evaluated strictly out-of-sample (525 sessions). Prioritize Crisis Recall & F1-Score over raw Accuracy.</div>', unsafe_allow_html=True)

    c_b1, c_b2 = st.columns([1.5, 1])
    with c_b1:
        st.caption("Classification Performance Matrix")
        st.dataframe(cls_results, use_container_width=True, hide_index=True)
    with c_b2:
        st.caption("Continuous Drawdown Regression (Forward Return)")
        st.dataframe(reg_results, use_container_width=True, hide_index=True)

    # Plotly Grouped Bar Chart
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(
        x=cls_results['Algorithm'], y=cls_results['Accuracy'],
        name='Accuracy', marker_color='#38BDF8'
    ))
    fig_models.add_trace(go.Bar(
        x=cls_results['Algorithm'], y=cls_results['Recall'],
        name='Recall (Crisis Sensitivity)', marker_color='#10B981'
    ))
    fig_models.add_trace(go.Bar(
        x=cls_results['Algorithm'], y=cls_results['F1_Score'],
        name='F1-Score', marker_color='#F59E0B'
    ))
    apply_plot_theme(
        fig_models, height=380, barmode='group',
        title='Algorithmic Performance Comparison (Accuracy vs. Recall vs. F1-Score)'
    )
    fig_models.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig_models.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig_models, use_container_width=True)

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown(f'<div class="card-title">🧩 Confusion Matrix ({cm_model_name})</div>', unsafe_allow_html=True)
        clf_diag = models[cm_model_name]
        if cm_model_name == 'Naive Bayes':
            preds_cm = (clf_diag.predict_proba(X_test_scaled)[:, 1] > 0.15).astype(int)
        else:
            preds_cm = clf_diag.predict(X_test_scaled)
        cm = confusion_matrix(y_test_class, preds_cm)
        
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: Normal', 'Actual: Crash'],
            columns=['Pred: Normal', 'Pred: Crash']
        )
        fig_cm = px.imshow(
            cm_df, text_auto=True, color_continuous_scale='Blues', aspect="auto"
        )
        apply_plot_theme(fig_cm, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_diag2:
        st.markdown('<div class="card-title">🌳 Random Forest Gini Feature Attribution</div>', unsafe_allow_html=True)
        rf_model = models['Random Forest']
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Teal'
        )
        apply_plot_theme(fig_imp, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

def render_clusters():
    st.markdown('<div class="card-title">🛡️ 4. Unsupervised Constituent Downside Risk Profiling & PCA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Bridging macro crash warnings with micro portfolio allocation using K-Means and Hierarchical Clustering.</div>', unsafe_allow_html=True)
    
    st.dataframe(stock_df.reset_index(), use_container_width=True, hide_index=True)

    col_scat, col_dend = st.columns(2)
    with col_scat:
        st.markdown('<div class="card-title">📍 Stock Risk vs. Return Tiers</div>', unsafe_allow_html=True)
        tier_names = {0: 'Tier 1: Defensive', 1: 'Tier 2: Moderate', 2: 'Tier 3: High Beta'}
        stock_plot_df = stock_df.reset_index().copy()
        stock_plot_df['Risk_Tier'] = stock_plot_df['Cluster'].map(tier_names)
        stock_plot_df['Ticker_Clean'] = stock_plot_df['Ticker'].apply(lambda x: x.replace('.NS', ''))

        fig_cluster = px.scatter(
            stock_plot_df, x='Annual_Volatility', y='Annual_Return',
            color='Risk_Tier', text='Ticker_Clean', size=[24]*len(stock_plot_df),
            color_discrete_map={
                'Tier 1: Defensive': '#10B981',
                'Tier 2: Moderate': '#38BDF8',
                'Tier 3: High Beta': '#EF4444'
            }
        )
        fig_cluster.update_traces(textposition='top center', marker=dict(opacity=0.9, line=dict(width=1.5, color='#F8FAFC')))
        apply_plot_theme(fig_cluster, height=380)
        fig_cluster.update_layout(legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_dend:
        st.markdown('<div class="card-title">🌲 Hierarchical Clustering Dendrogram</div>', unsafe_allow_html=True)
        fig_dend, ax = plt.subplots(figsize=(6.5, 3.8), facecolor='none')
        ax.set_facecolor('none')
        Z = linkage(stock_df[['Annual_Return', 'Annual_Volatility']], method='ward')
        dendrogram(Z, labels=[t.replace('.NS', '') for t in stock_df.index], ax=ax,
                   color_threshold=0.1, above_threshold_color='#94A3B8')
        ax.tick_params(colors='#CBD5E1', which='both')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig_dend)

    col_el, col_pca = st.columns(2)
    with col_el:
        st.markdown('<div class="card-title">📐 K-Means Elbow Curve</div>', unsafe_allow_html=True)
        inertias = []
        K_range = range(1, min(7, len(stock_df) + 1))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(stock_df[['Annual_Return', 'Annual_Volatility']])
            inertias.append(km.inertia_)
        
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(K_range), y=inertias,
            mode='lines+markers', line=dict(color='#A855F7', width=2.5),
            marker=dict(size=8, color='#C084FC')
        ))
        fig_elbow.add_vline(x=3, line=dict(color='#10B981', dash='dash', width=2),
                            annotation_text="Optimal k = 3")
        apply_plot_theme(fig_elbow, height=330, xaxis_title="Clusters (k)", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_pca:
        st.markdown('<div class="card-title">🌌 PCA 2D Market Regime Map</div>', unsafe_allow_html=True)
        X_all_scaled = scaler.transform(df_full[feature_cols])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Regime'] = df_full['Crash'].map({0: 'Normal Day', 1: 'Crash Warning'})

        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2', color='Regime',
            color_discrete_map={'Normal Day': '#1E3A8A', 'Crash Warning': '#EF4444'},
            opacity=0.75
        )
        apply_plot_theme(fig_pca, height=330)
        fig_pca.update_layout(legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig_pca, use_container_width=True)

def render_rules():
    st.markdown('<div class="card-title">🔍 5. Pre-Crash Pattern Discovery (Apriori Association Rules)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Transparent, explainable IF-THEN rules mined from market stress confluence. Lift > 1.0 proves significant risk multiplication.</div>', unsafe_allow_html=True)
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    if len(rules_df) > 0:
        plot_rules = rules_df.head(6).copy()
        plot_rules['Rule'] = plot_rules['antecedents'] + '  ==>  ' + plot_rules['consequents']
        fig_rules = px.bar(
            plot_rules, x='lift', y='Rule', orientation='h',
            color='lift', color_continuous_scale='Reds',
            labels={'lift': 'Lift (Crisis Multiplier Over Baseline)'}
        )
        fig_rules.add_vline(x=1.0, line=dict(color='#94A3B8', dash='dash', width=2),
                            annotation_text="Random Baseline (Lift = 1.0)")
        apply_plot_theme(fig_rules, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_rules, use_container_width=True)

# =========================================================
# MAIN PRESENTATION ROUTER (UNIFIED VS TABBED)
# =========================================================
if view_mode == "Unified All-in-One Dashboard":
    render_live_predictor()
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    render_market_trends()
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    render_benchmarks()
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    render_clusters()
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    render_rules()
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Risk Predictor",
        "📈 Market Trends & EDA",
        "🤖 Model Benchmarks",
        "🛡️ Stock Clusters & PCA",
        "🔍 Pre-Crash Rules"
    ])
    with tab1:
        render_live_predictor()
    with tab2:
        render_market_trends()
    with tab3:
        render_benchmarks()
    with tab4:
        render_clusters()
    with tab5:
        render_rules()
