import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import os
from sklearn.metrics import mean_squared_error

# --- DEPLOYMENT CONFIG ---
st.set_page_config(page_title="ESG AUDITOR PRO", page_icon="🏦", layout="wide")

# (The theme is handled by your .streamlit/config.toml file for maximum stability)

@st.cache_data
def load_and_audit_data():
    # Use relative path for deployment
    data_file = "ESG_Wide_ML_Dataset.xlsx"
    if not os.path.exists(data_file):
        st.error("DATAFILE MISSING IN REPOSITORY. Please ensure 'ESG_Wide_ML_Dataset.xlsx' is in your root folder.")
        st.stop()
    
    df = pd.read_excel(data_file, sheet_name="ML_Dataset")
    
    # 1. Clean Leakage
    df_clean = df.copy()
    leakage = ['ESG Scr Percntle', 'E_Env Pillr Percntle', 'S_Soc Pillr Percntle', 'G_Gov Pillr Percntle']
    df_clean = df_clean.drop(columns=leakage, errors='ignore')
    
    # 2. Impute NaNs
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    
    # 3. Log Transforms
    if 'Mkt Cap' in df_clean.columns:
        df_clean['log_Mkt_Cap'] = np.log1p(df_clean['Mkt Cap'])
    
    # 4. Feature Selection
    target = 'ESG Scr'
    cors = df_clean.select_dtypes(include=[np.number]).corr()[target].abs().sort_values(ascending=False)
    features = [c for c in cors.index[1:11] if c in df_clean.columns]
    
    # 5. Model
    X = sm.add_constant(df_clean[features])
    y = df_clean[target]
    model = sm.OLS(y, X).fit()
    y_pred = pd.Series(model.predict(X), index=df_clean.index)
    
    return df, df_clean, y, y_pred, features, model

try:
    df_orig, df_work, y_act, y_pre, feature_set, ols_results = load_and_audit_data()
    
    # HEADER SECTION
    st.markdown("# ESG PERFORMANCE AUDIT REPORT")
    st.markdown("### Institutional Predictive Suite | Gaming Sector Analytics")
    st.markdown("---")

    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("AUDIT CONTROLS")
        asset_select = st.selectbox("SEARCH COMPANY:", df_orig['Short Name'].sort_values())
        st.markdown("---")
        st.caption("Deployment Version 4.1")

    # KEY METRICS
    idx = df_orig[df_orig['Short Name'] == asset_select].index[0]
    act_v, pre_v = y_act.loc[idx], y_pre.loc[idx]
    
    st.subheader(f"ASSET SCORECARD: {asset_select.upper()}")
    k1, k2, k3 = st.columns(3)
    k1.metric("ESTIMATED FAIR VALUE", f"{pre_v:.3f}")
    k2.metric("ACTUAL REPORTED SCORE", f"{act_v:.3f}")
    k3.metric("VARIANCE (ALPHA)", f"{act_v-pre_v:+.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ANALYSIS TABS
    nav1, nav2, nav3 = st.tabs(["📊 MODEL ANALYSIS", "📈 DRIVER INSIGHTS", "🏁 BENCHMARKING"])

    with nav1:
        st.write("#### AI Model Accuracy vs Market Reality")
        fig1 = px.scatter(x=y_act, y=y_pre, template="plotly_white", labels={'x':'Actual','y':'Predicted'}, hover_name=df_orig['Short Name'])
        fig1.add_shape(type="line", x0=y_act.min(), y0=y_act.min(), x1=y_act.max(), y1=y_act.max(), line=dict(color="black", dash="dash"))
        fig1.update_traces(marker=dict(color='black', size=10))
        st.plotly_chart(fig1, use_container_width=True)

    with nav2:
        st.write("#### Primary ESG Value Drivers")
        coeffs = ols_results.params.drop('const').sort_values()
        fig2 = px.bar(x=coeffs.values, y=coeffs.index, orientation='h', template="plotly_white", color_discrete_sequence=['black'])
        st.plotly_chart(fig2, use_container_width=True)

    with nav3:
        st.write(f"#### Benchmarking: {asset_select} vs Industry Mean")
        cp_v, sc_a = df_work.loc[idx, feature_set], df_work[feature_set].mean()
        b_df = pd.DataFrame({'Metric': feature_set, 'Asset': cp_v.values, 'Industry': sc_a.values}).melt(id_vars='Metric')
        fig3 = px.bar(b_df, x='value', y='Metric', color='variable', barmode='group', 
                     color_discrete_map={'Asset': 'black', 'Industry': '#D1D5DB'}, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # DIAGNOSTICS
    st.markdown("---")
    st.subheader("MODEL DIAGNOSTICS")
    d1, d2, d3 = st.columns(3)
    d1.metric("PRECISION (R²)", f"{ols_results.rsquared:.4f}")
    d2.metric("ADJUSTED R²", f"{ols_results.rsquared_adj:.4f}")
    d3.metric("RMSE ERROR", f"{np.sqrt(mean_squared_error(y_act, y_pre)):.4f}")

except Exception as ex:
    st.error(f"AUDITOR SYSTEM FAILURE: {ex}")

st.markdown("<br><hr><center>AUDIT OFFICE INTERNAL DOCUMENT</center>", unsafe_allow_html=True)
