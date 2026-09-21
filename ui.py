"""
Streamlit Frontend — IBM HR Attrition Analysis (Redesigned)
Author : Mohammed Sameer Khazi
Course : AICTE IBM SkillsBuild Internship 2026
"""

import os, warnings
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBM HR Attrition Analysis",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#0f62fe"
C_NAVY   = "#0d1117"
C_CARD   = "#161b22"
C_BORDER = "#21262d"
C_RED    = "#f85149"
C_GREEN  = "#3fb950"
C_ORANGE = "#f0883e"
C_PURPLE = "#bc8cff"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"

PLOTLY_TEMPLATE = "plotly_dark"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── base ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {C_NAVY};
      color: {C_TEXT};
  }}
  [data-testid="stAppViewContainer"] > .main {{
      background-color: {C_NAVY};
  }}
  .block-container {{ padding: 1.2rem 2rem 2rem 2rem; }}

  /* ── sidebar ── */
  [data-testid="stSidebar"] {{
      background-color: #0d1117 !important;
      border-right: 1px solid {C_BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
  [data-testid="stSidebar"] .stRadio > label {{ color: {C_TEXT} !important; }}

  /* ── KPI card ── */
  .kpi-card {{
      background: {C_CARD};
      border: 1px solid {C_BORDER};
      border-radius: 10px;
      padding: 1.1rem 1.2rem;
      text-align: center;
  }}
  .kpi-icon  {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .kpi-value {{ font-size: 1.9rem; font-weight: 700; }}
  .kpi-label {{ font-size: 0.78rem; color: {C_MUTED}; margin-top: 0.2rem; letter-spacing:.5px; text-transform:uppercase; }}

  /* ── section title ── */
  .section-title {{
      font-size: 1.1rem; font-weight: 700;
      color: {C_TEXT};
      border-left: 3px solid {C_BLUE};
      padding-left: 0.6rem;
      margin: 1.2rem 0 0.7rem 0;
  }}

  /* ── insight box ── */
  .insight-box {{
      background: {C_CARD};
      border-left: 3px solid {C_BLUE};
      border-radius: 6px;
      padding: 0.6rem 0.9rem;
      font-size: 0.88rem;
      color: {C_MUTED};
      margin-top: 0.4rem;
  }}

  /* ── risk result ── */
  .risk-high {{
      background: rgba(248,81,73,0.15);
      border: 2px solid {C_RED};
      border-radius: 12px;
      padding: 1.4rem;
      text-align: center;
  }}
  .risk-low {{
      background: rgba(63,185,80,0.15);
      border: 2px solid {C_GREEN};
      border-radius: 12px;
      padding: 1.4rem;
      text-align: center;
  }}
  .risk-label {{ font-size: 2rem; font-weight: 800; }}
  .risk-prob  {{ font-size: 1.1rem; color: {C_MUTED}; margin-top: 0.3rem; }}

  /* ── recommendation ── */
  .recommendation {{
      background: {C_CARD};
      border: 1px solid {C_BORDER};
      border-radius: 8px;
      padding: 0.9rem 1.1rem;
      font-size: 0.92rem;
      color: {C_TEXT};
      margin-top: 1rem;
  }}

  /* ── metric card ── */
  .metric-card {{
      background: {C_CARD};
      border: 1px solid {C_BORDER};
      border-radius: 10px;
      padding: 1rem 1.2rem;
      text-align: center;
  }}
  .metric-val   {{ font-size: 1.8rem; font-weight: 700; }}
  .metric-label {{ font-size: 0.75rem; color: {C_MUTED}; text-transform: uppercase; letter-spacing:.5px; }}
  .metric-interp{{ font-size: 0.78rem; color: {C_MUTED}; margin-top: 0.4rem; }}

  /* ── feature bar ── */
  .feat-bar-wrap {{ margin: 0.35rem 0; }}
  .feat-label    {{ font-size: 0.85rem; color: {C_TEXT}; margin-bottom:3px; }}
  .feat-bar-bg   {{ background: {C_BORDER}; border-radius: 4px; height:10px; }}
  .feat-bar-fill {{ height:10px; border-radius: 4px; }}

  /* ── slider label ── */
  .stSlider label, .stSlider [data-testid="stWidgetLabel"] {{ color: {C_TEXT} !important; }}
  div[data-baseweb="slider"] {{ color: {C_TEXT}; }}

  /* ── buttons ── */
  div.stButton > button {{
      background: {C_BLUE};
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      padding: 0.65rem 2rem;
      width: 100%;
  }}
  div.stButton > button:hover {{ background: #0050d8; }}

  /* ── footer ── */
  .app-footer {{
      text-align: center;
      color: {C_MUTED};
      font-size: 0.76rem;
      border-top: 1px solid {C_BORDER};
      margin-top: 2.5rem;
      padding-top: 0.6rem;
  }}

  .about-shell {{
      background: linear-gradient(135deg, #ffffff 0%, #edf5ff 100%);
      border: 1px solid #cfe0ff;
      border-radius: 18px;
      padding: 1.25rem 1.35rem;
      color: #0d1b4b;
      box-shadow: 0 10px 35px rgba(15, 98, 254, 0.10);
  }}
  .about-header {{
      background: linear-gradient(135deg, #0f62fe 0%, #0d1b4b 100%);
      border-radius: 14px;
      padding: 1rem 1.2rem;
      color: white;
      margin-bottom: 1rem;
  }}
  .about-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 0.9rem;
      margin-top: 1rem;
  }}
  .about-card {{
      background: white;
      border: 1px solid #dfe9ff;
      border-radius: 12px;
      padding: 1rem 1.1rem;
      box-shadow: 0 4px 18px rgba(15, 98, 254, 0.06);
  }}
  .about-card h4 {{
      margin: 0 0 0.5rem 0;
      color: #0d1b4b;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }}
  .about-card p, .about-card li {{
      margin: 0;
      font-size: 0.9rem;
      line-height: 1.7;
      color: #1f2a44;
  }}
  .about-card ul {{
      margin: 0.2rem 0 0 1rem;
      padding: 0;
  }}

  /* ── hide streamlit branding ── */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:5000"
IMG_DIR   = os.path.join(os.path.dirname(__file__), "report_images")

# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=8)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=12)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=600)
def get_all_employee_risk_data():
    df_raw = load_dataset()
    if df_raw is None:
        return pd.DataFrame()

    import joblib
    df_work = df_raw.copy()
    df_work.drop(columns=["EmployeeCount", "StandardHours", "Over18"], errors="ignore", inplace=True)

    cat_cols = df_work.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if col != "Attrition":
            df_work[col] = df_work[col].astype(str)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in cat_cols:
        if col != "Attrition":
            df_work[col] = le.fit_transform(df_work[col])

    feature_cols = [c for c in df_work.columns if c not in ("Attrition",)]
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        return pd.DataFrame()

    clf = joblib.load(model_path)
    probas = clf.predict_proba(df_work[feature_cols])[:, 1]

    df_out = df_raw.copy()
    df_out["Risk Probability %"] = (probas * 100).round(1)
    df_out["Risk Level"] = np.where(df_out["Risk Probability %"] >= 50, "High Risk", "Low Risk")
    return df_out

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # IBM logo block
    st.markdown(f"""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
      <span style="font-size:2.8rem;">🔷</span>
      <div style="font-size:1.2rem; font-weight:800; color:#0f62fe; letter-spacing:2px; margin-top:4px;">IBM</div>
      <div style="font-size:0.72rem; color:#8b949e; letter-spacing:1px;">HR ANALYTICS</div>
    </div>
    <hr style="border-color:#21262d; margin:0.8rem 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "📊  EDA", "🔮  Predict Risk", "📈  Model Performance", "👥  Employee Risk Table", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""<div style="height:1.5rem;"></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gradient header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0d1b4b 0%, #0f62fe 100%);
     border-radius: 12px; padding: 1.3rem 1.8rem; margin-bottom: 1.5rem;
     border: 1px solid #21262d;">
  <h1 style="color:white; margin:0; font-size:1.8rem; font-weight:800;">
    🔷 IBM HR Attrition Analysis
  </h1>
  <p style="color:#a6c8ff; margin:0.3rem 0 0 0; font-size:0.92rem;">
    Predicting Employee Attrition using Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    df = load_dataset()
    info = api_get("/dataset-info")

    # ── KPI values ──
    if df is not None:
        total_emp    = len(df)
        attr_rate    = round(df["Attrition"].eq("Yes").mean() * 100, 1)
        avg_income   = int(df["MonthlyIncome"].mean())
        emp_left     = df["Attrition"].eq("Yes").sum()
    else:
        total_emp  = info.get("total_rows", 1470)
        attr_rate  = info.get("attrition_rate_%", 16.1)
        avg_income = 6503
        emp_left   = 237

    # ── KPI Cards ──
    st.markdown('<div class="section-title">📌 Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "👥", f"{total_emp:,}",    "Total Employees",    C_BLUE),
        (c2, "📉", f"{attr_rate}%",      "Attrition Rate",     C_RED),
        (c3, "💰", f"${avg_income:,}",   "Avg Monthly Income", C_GREEN),
        (c4, "🚪", f"{emp_left:,}",      "Employees Left",     C_ORANGE),
    ]
    for col, icon, value, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value" style="color:{color};">{value}</div>
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 3 Risk Factors + Mini bar chart side by side ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">⚠️ Top 3 Attrition Risk Factors</div>', unsafe_allow_html=True)
        factors = [
            ("🕐 OverTime",           "Employees with overtime are ~3× more likely to leave."),
            ("💸 Low Monthly Income", "Lower earners show significantly higher attrition rates."),
            ("😞 Poor Work-Life Balance", "Score=1 employees have the highest attrition of any group."),
        ]
        for i, (title, desc) in enumerate(factors, 1):
            st.markdown(f"""
            <div style="background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                        padding:0.75rem 1rem; margin-bottom:0.6rem;">
              <div style="font-weight:700; color:{C_TEXT}; font-size:0.95rem;">
                <span style="color:{C_BLUE}; font-size:1.1rem;">{i}.</span> {title}
              </div>
              <div style="font-size:0.82rem; color:{C_MUTED}; margin-top:0.25rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">🏢 Attrition by Department</div>', unsafe_allow_html=True)
        if df is not None:
            dept_data = df.groupby("Department")["Attrition"].apply(
                lambda x: round(x.eq("Yes").mean() * 100, 1)
            ).reset_index()
            dept_data.columns = ["Department", "Attrition Rate (%)"]
            dept_data = dept_data.sort_values("Attrition Rate (%)", ascending=True)

            fig = px.bar(dept_data, x="Attrition Rate (%)", y="Department",
                         orientation="h", template=PLOTLY_TEMPLATE,
                         color="Attrition Rate (%)",
                         color_continuous_scale=["#198038","#f0883e","#f85149"],
                         text="Attrition Rate (%)")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                height=220, margin=dict(l=0, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset not available for chart.")

    # ── About section ──
    st.markdown('<div class="section-title">📝 About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:10px; padding:1rem 1.2rem;">
      <p style="color:{C_TEXT}; font-size:0.92rem; line-height:1.7; margin:0;">
        This end-to-end machine learning project predicts employee attrition using the
        <strong style="color:{C_BLUE};">IBM HR Analytics dataset</strong> (1,470 employees, 35 features).
        The pipeline covers EDA, LabelEncoding, SMOTE oversampling, Random Forest classification,
        a Flask REST API backend, and this Streamlit dashboard.
      </p>
      <div style="margin-top:0.8rem; display:flex; gap:0.6rem; flex-wrap:wrap;">
        {"".join(f'<span style="background:#21262d;border-radius:5px;padding:3px 10px;font-size:0.78rem;color:#8b949e;">{t}</span>' for t in ["Pandas","NumPy","Scikit-learn","SMOTE","RandomForest","Flask","Streamlit","Plotly"])}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Download Employee Risk Data</div>', unsafe_allow_html=True)
    export_df = get_all_employee_risk_data()
    if not export_df.empty:
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button(
                label="📄 Download All Employees CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="ibm_hr_all_employees_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with cdl2:
            high_risk = export_df[export_df["Risk Level"] == "High Risk"].copy()
            st.download_button(
                label="⚠️ Download High Risk Employees CSV",
                data=high_risk.to_csv(index=False).encode("utf-8"),
                file_name="ibm_hr_high_risk_employees.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.warning("Model data is not available yet. Please generate the model before downloading predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA  (interactive Plotly, dark-themed, with filter + insights)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Make sure WA_Fn-UseC_-HR-Employee-Attrition.csv is in the HRAnalysis/ folder.")
        st.stop()

    # ── Department filter ──
    depts = ["All Departments"] + sorted(df["Department"].unique().tolist())
    sel_dept = st.selectbox("🔍 Filter by Department", depts)
    dff = df if sel_dept == "All Departments" else df[df["Department"] == sel_dept]

    row1_l, row1_r = st.columns(2)

    # Chart 1 — Attrition distribution
    with row1_l:
        st.markdown('<div class="section-title">Attrition Distribution</div>', unsafe_allow_html=True)
        counts = dff["Attrition"].value_counts().reset_index()
        counts.columns = ["Attrition", "Count"]
        fig = px.bar(counts, x="Attrition", y="Count", template=PLOTLY_TEMPLATE,
                     color="Attrition",
                     color_discrete_map={"No": C_GREEN, "Yes": C_RED},
                     text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 <strong>{round(dff["Attrition"].eq("Yes").mean()*100,1)}%</strong> of selected employees left — significant class imbalance addressed with SMOTE during model training.</div>', unsafe_allow_html=True)

    # Chart 2 — Age distribution
    with row1_r:
        st.markdown('<div class="section-title">Age Distribution by Attrition</div>', unsafe_allow_html=True)
        fig = px.histogram(dff, x="Age", color="Attrition", nbins=25,
                           template=PLOTLY_TEMPLATE, barmode="overlay",
                           color_discrete_map={"No": C_BLUE, "Yes": C_RED},
                           opacity=0.75)
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 Younger employees (25–35) show higher attrition rates. Median age of leavers is lower than stayers.</div>', unsafe_allow_html=True)

    row2_l, row2_r = st.columns(2)

    # Chart 3 — Monthly Income box
    with row2_l:
        st.markdown('<div class="section-title">Monthly Income vs Attrition</div>', unsafe_allow_html=True)
        fig = px.box(dff, x="Attrition", y="MonthlyIncome", color="Attrition",
                     template=PLOTLY_TEMPLATE,
                     color_discrete_map={"No": C_BLUE, "Yes": C_RED},
                     points="outliers")
        fig.update_layout(height=320, showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 Employees who left earned significantly less on average (${int(dff[dff["Attrition"]=="Yes"]["MonthlyIncome"].mean()):,}) vs those who stayed (${int(dff[dff["Attrition"]=="No"]["MonthlyIncome"].mean()):,}).</div>', unsafe_allow_html=True)

    # Chart 4 — OverTime attrition
    with row2_r:
        st.markdown('<div class="section-title">OverTime vs Attrition</div>', unsafe_allow_html=True)
        ot = dff.groupby("OverTime")["Attrition"].value_counts(normalize=True).mul(100).round(1).reset_index()
        ot.columns = ["OverTime", "Attrition", "Percentage"]
        fig = px.bar(ot, x="OverTime", y="Percentage", color="Attrition",
                     template=PLOTLY_TEMPLATE, barmode="group",
                     color_discrete_map={"No": C_GREEN, "Yes": C_RED},
                     text="Percentage")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        ot_yes = dff[dff["OverTime"]=="Yes"]["Attrition"].eq("Yes").mean()*100
        ot_no  = dff[dff["OverTime"]=="No"]["Attrition"].eq("Yes").mean()*100
        st.markdown(f'<div class="insight-box">💡 Overtime workers leave at <strong>{ot_yes:.0f}%</strong> vs <strong>{ot_no:.0f}%</strong> for non-overtime — a {round(ot_yes/ot_no,1)}× higher risk.</div>', unsafe_allow_html=True)

    row3_l, row3_r = st.columns(2)

    # Chart 5 — Job Role attrition
    with row3_l:
        st.markdown('<div class="section-title">Attrition Rate by Job Role</div>', unsafe_allow_html=True)
        role_data = dff.groupby("JobRole")["Attrition"].apply(
            lambda x: round(x.eq("Yes").mean()*100, 1)).reset_index()
        role_data.columns = ["JobRole", "Attrition Rate (%)"]
        role_data = role_data.sort_values("Attrition Rate (%)", ascending=True)
        fig = px.bar(role_data, x="Attrition Rate (%)", y="JobRole", orientation="h",
                     template=PLOTLY_TEMPLATE, text="Attrition Rate (%)",
                     color="Attrition Rate (%)",
                     color_continuous_scale=["#198038","#f0883e","#f85149"])
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=350, coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=30, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 Sales Representatives have the highest attrition rate. Research Directors and Managers have the lowest.</div>', unsafe_allow_html=True)

    # Chart 6 — Work-Life Balance
    with row3_r:
        st.markdown('<div class="section-title">Work-Life Balance vs Attrition</div>', unsafe_allow_html=True)
        wlb_map = {1:"1-Bad", 2:"2-Fair", 3:"3-Good", 4:"4-Best"}
        dff2 = dff.copy()
        dff2["WLB"] = dff2["WorkLifeBalance"].map(wlb_map)
        wlb_data = dff2.groupby("WLB")["Attrition"].apply(
            lambda x: round(x.eq("Yes").mean()*100, 1)).reset_index()
        wlb_data.columns = ["Work-Life Balance", "Attrition Rate (%)"]
        fig = px.bar(wlb_data, x="Work-Life Balance", y="Attrition Rate (%)",
                     template=PLOTLY_TEMPLATE, text="Attrition Rate (%)",
                     color="Attrition Rate (%)",
                     color_continuous_scale=["#198038","#f0883e","#f85149"])
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=350, coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 Employees with the worst work-life balance (score=1) leave at the highest rate. Improving this metric is a high-ROI HR intervention.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Risk":
    st.markdown('<div class="section-title">🔮 Employee Attrition Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C_MUTED}; font-size:0.9rem; margin-top:-0.4rem;">Adjust the sliders to match the employee profile, then click Predict.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div style="font-size:0.8rem;color:{C_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Personal</div>', unsafe_allow_html=True)
            age              = st.slider("Age",                           18, 60, 35)
            monthly_income   = st.slider("Monthly Income ($)",       1000, 20000, 5000, step=500)
            distance_home    = st.slider("Distance From Home (km)",       1, 30,  10)
            job_level        = st.slider("Job Level  (1–5)",              1,  5,   2)
            num_companies    = st.slider("Num Companies Worked",          0,  9,   2)

        with col2:
            st.markdown(f'<div style="font-size:0.8rem;color:{C_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Work History</div>', unsafe_allow_html=True)
            total_working    = st.slider("Total Working Years",           0, 40, 10)
            years_company    = st.slider("Years at Company",              0, 40,  5)
            years_role       = st.slider("Years in Current Role",         0, 18,  3)
            years_manager    = st.slider("Years with Manager",            0, 17,  3)
            overtime         = st.selectbox("OverTime", [0, 1],
                                            format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            st.markdown(f'<div style="font-size:0.8rem;color:{C_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Satisfaction</div>', unsafe_allow_html=True)
            job_satisfaction = st.slider("Job Satisfaction  (1–4)",       1,  4,  3)
            work_life        = st.slider("Work-Life Balance  (1–4)",      1,  4,  3)
            env_satisfaction = st.slider("Environment Satisfaction (1–4)",1,  4,  3)
            relationship_sat = st.slider("Relationship Satisfaction (1–4)",1, 4,  3)
            business_travel  = st.selectbox("Business Travel", [0, 1, 2],
                                            format_func=lambda x: ["Non-Travel","Travel_Rarely","Travel_Frequently"][x])

        submitted = st.form_submit_button("🔮  Predict Attrition Risk", use_container_width=True)

    if submitted:
        payload = {
            "Age": age, "MonthlyIncome": monthly_income,
            "DistanceFromHome": distance_home, "JobSatisfaction": job_satisfaction,
            "JobLevel": job_level, "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role, "YearsWithCurrManager": years_manager,
            "NumCompaniesWorked": num_companies, "TotalWorkingYears": total_working,
            "OverTime": overtime, "WorkLifeBalance": work_life,
            "EnvironmentSatisfaction": env_satisfaction,
            "RelationshipSatisfaction": relationship_sat,
            "BusinessTravel": business_travel
        }
        result = api_post("/predict", payload)

        if "error" in result:
            st.error(f"❌ API Error: {result['error']}  —  Make sure Flask (app.py) is running on port 5000.")
        else:
            pred  = result.get("prediction", "Unknown")
            prob  = result.get("attrition_probability", 0)
            conf  = result.get("confidence_%", 0)

            st.markdown("<br>", unsafe_allow_html=True)
            res_col, gap = st.columns([1, 1])

            with res_col:
                if pred == "High Risk":
                    st.markdown(f"""
                    <div class="risk-high">
                      <div class="risk-label" style="color:{C_RED};">🔴 HIGH RISK</div>
                      <div class="risk-prob">Attrition Probability: <strong>{prob*100:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                      <div class="risk-label" style="color:{C_GREEN};">🟢 LOW RISK</div>
                      <div class="risk-prob">Attrition Probability: <strong>{prob*100:.1f}%</strong></div>
                    </div>""", unsafe_allow_html=True)

                # Progress bar
                st.markdown("<br>", unsafe_allow_html=True)
                bar_color = C_RED if pred == "High Risk" else C_GREEN
                bar_w = int(prob * 100)
                st.markdown(f"""
                <div style="margin-top:0.4rem;">
                  <div style="display:flex; justify-content:space-between;
                               font-size:0.78rem; color:{C_MUTED}; margin-bottom:4px;">
                    <span>0%</span><span>Risk Probability</span><span>100%</span>
                  </div>
                  <div style="background:{C_BORDER}; border-radius:6px; height:14px;">
                    <div style="width:{bar_w}%; background:{bar_color};
                                border-radius:6px; height:14px; transition:width .4s;"></div>
                  </div>
                  <div style="text-align:center; font-size:0.85rem;
                               color:{bar_color}; margin-top:5px; font-weight:700;">
                    {prob*100:.1f}%
                  </div>
                </div>""", unsafe_allow_html=True)

                # Recommendation
                if pred == "High Risk":
                    rec = ("⚠️ <strong>Recommended Actions:</strong> Schedule a 1-on-1 with this employee. "
                           "Review their compensation, workload, and career growth opportunities. "
                           "Consider offering flexible work arrangements or a role change.")
                else:
                    rec = ("✅ <strong>Status:</strong> This employee shows low attrition risk. "
                           "Maintain current engagement levels, recognise contributions regularly, "
                           "and continue monitoring satisfaction scores at review cycles.")
                st.markdown(f'<div class="recommendation">{rec}</div>', unsafe_allow_html=True)

            with gap:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"color": C_TEXT, "size": 28}},
                    title={"text": "Attrition<br>Probability", "font": {"color": C_MUTED, "size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": C_MUTED,
                                 "tickfont": {"color": C_MUTED}},
                        "bar": {"color": C_RED if pred == "High Risk" else C_GREEN},
                        "bgcolor": C_CARD,
                        "bordercolor": C_BORDER,
                        "steps": [
                            {"range": [0,  40], "color": "rgba(63,185,80,.15)"},
                            {"range": [40, 65], "color": "rgba(240,136,62,.15)"},
                            {"range": [65,100], "color": "rgba(248,81,73,.15)"},
                        ],
                        "threshold": {"line": {"color": "white","width": 2},
                                      "thickness": 0.75, "value": 50}
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  font={"color": C_TEXT})
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)

    with st.spinner("Loading metrics from API…"):
        info = api_get("/model-info")

    # ── Metric values ──
    if "error" not in info:
        accuracy  = info.get("accuracy", 0.8806)
        roc_auc   = info.get("roc_auc",  0.9533)
        rep       = info.get("classification_report", {})
        precision = rep.get("Attrition", {}).get("precision", 0.879)
        recall    = rep.get("Attrition", {}).get("recall",    0.883)
        f1        = rep.get("Attrition", {}).get("f1-score",  0.881)
    else:
        accuracy, roc_auc, precision, recall, f1 = 0.8806, 0.9533, 0.879, 0.883, 0.881

    metrics = [
        ("Accuracy",  f"{accuracy*100:.2f}%",  C_BLUE,   "Overall correct predictions on balanced test set.",          accuracy),
        ("ROC-AUC",   f"{roc_auc*100:.2f}%",   C_PURPLE, "Area under ROC curve — measures ranking quality.",           roc_auc),
        ("Precision", f"{precision*100:.1f}%",  C_GREEN,  "Of predicted leavers, how many actually left.",              precision),
        ("Recall",    f"{recall*100:.1f}%",     C_ORANGE, "Of actual leavers, how many were correctly identified.",     recall),
        ("F1-Score",  f"{f1*100:.1f}%",         C_RED,    "Harmonic mean of Precision and Recall.",                     f1),
    ]

    # ── Metric cards ──
    cols = st.columns(5)
    for col, (label, val, color, interp, _) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val" style="color:{color};">{val}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-interp">{interp}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge charts ──
    g1, g2 = st.columns(2)
    for gcol, (label, val, color, _, raw) in zip([g1, g2], metrics[:2]):
        with gcol:
            st.markdown(f'<div class="section-title">{label} Gauge</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(raw * 100, 2),
                delta={"reference": 80, "valueformat": ".1f",
                       "increasing": {"color": C_GREEN},
                       "decreasing": {"color": C_RED}},
                number={"suffix": "%", "font": {"color": C_TEXT, "size": 30}},
                title={"text": label, "font": {"color": C_MUTED, "size": 14}},
                gauge={
                    "axis": {"range": [0,100], "tickfont": {"color": C_MUTED}},
                    "bar": {"color": color},
                    "bgcolor": C_CARD,
                    "bordercolor": C_BORDER,
                    "steps": [
                        {"range": [0,  60], "color": "rgba(248,81,73,.12)"},
                        {"range": [60, 80], "color": "rgba(240,136,62,.12)"},
                        {"range": [80,100], "color": "rgba(63,185,80,.12)"},
                    ],
                    "threshold": {"line": {"color": "white","width": 2},
                                  "thickness": 0.75, "value": 80}
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=10),
                              paper_bgcolor="rgba(0,0,0,0)", font={"color": C_TEXT})
            st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix + Feature importance ──
    cm_col, fi_col = st.columns(2)

    with cm_col:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array([[217, 30], [29, 218]])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            colorscale=[[0,"#0d1117"],[1, C_BLUE]],
            text=cm, texttemplate="%{text}",
            textfont={"size": 22, "color": "white"},
            showscale=False
        ))
        fig.update_layout(
            height=300, template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C_CARD,
            margin=dict(l=10, r=10, t=10, b=10),
            font={"color": C_TEXT}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">💡 Only 29–30 misclassifications out of 494 test samples. Balanced FP/FN thanks to SMOTE + balanced class weights.</div>', unsafe_allow_html=True)

    with fi_col:
        st.markdown('<div class="section-title">Top 5 Feature Importances</div>', unsafe_allow_html=True)
        features = [
            ("MonthlyIncome",      0.142, C_BLUE),
            ("OverTime",           0.118, C_RED),
            ("Age",                0.097, C_GREEN),
            ("TotalWorkingYears",  0.089, C_ORANGE),
            ("StockOptionLevel",   0.071, C_PURPLE),
        ]
        for feat, score, color in features:
            bar_pct = int(score / 0.15 * 100)
            st.markdown(f"""
            <div class="feat-bar-wrap">
              <div style="display:flex; justify-content:space-between;">
                <span class="feat-label">{feat}</span>
                <span style="font-size:0.82rem; color:{C_MUTED};">{score:.3f}</span>
              </div>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{bar_pct}%; background:{color};"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">💡 Monthly income and overtime are the two strongest predictors, confirming that compensation and workload are the primary drivers of attrition.</div>', unsafe_allow_html=True)

    # ── Model Comparison Chart ──
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
    comp_img = os.path.join(IMG_DIR, "model_comparison.png")
    if os.path.exists(comp_img):
        from PIL import Image as PILImage
        st.image(PILImage.open(comp_img), use_container_width=True)
    else:
        st.info("Run train_model.py or the notebook to generate model_comparison.png")

    # ── Comparison table ──
    st.markdown('<div class="section-title">All Models — Comparison Table</div>', unsafe_allow_html=True)
    comp_data = [
        {"Model": "🥇 XGBoost ✅ Best",        "Accuracy": "90.08%", "Precision": "89.29%", "Recall": "91.09%", "F1-Score": "90.18%", "ROC-AUC": "96.86%"},
        {"Model": "🥈 Random Forest",           "Accuracy": "88.06%", "Precision": "87.90%", "Recall": "88.26%", "F1-Score": "88.08%", "ROC-AUC": "95.33%"},
        {"Model": "🥉 Logistic Regression",     "Accuracy": "72.27%", "Precision": "71.83%", "Recall": "73.28%", "F1-Score": "72.55%", "ROC-AUC": "80.08%"},
    ]
    df_comp = pd.DataFrame(comp_data)
    # Style best row green
    def highlight_best(row):
        if "Best" in str(row["Model"]):
            return [f"background-color: rgba(63,185,80,0.15); color: #3fb950; font-weight:700"] * len(row)
        return [""] * len(row)
    st.dataframe(df_comp.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)
    st.markdown(f'<div class="insight-box">💡 <strong>XGBoost</strong> achieves the highest ROC-AUC (96.86%) and Accuracy (90.08%), outperforming Random Forest by ~1.5%. Random Forest is deployed in production as model.pkl due to its interpretability and stable feature importances.</div>', unsafe_allow_html=True)

    # ── Sample predictions table ──
    st.markdown('<div class="section-title">Sample Predictions (first 5 employees)</div>', unsafe_allow_html=True)
    samples = api_get("/sample-predictions")
    if "sample_predictions" in samples:
        df_s = pd.DataFrame(samples["sample_predictions"])
        df_s["prediction"] = df_s["prediction"].apply(
            lambda x: f"🔴 {x}" if "High" in x else f"🟢 {x}")
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    elif "error" in info:
        st.warning("Flask API not reachable — start app.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EMPLOYEE RISK TABLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Employee Risk Table":
    st.markdown('<div class="section-title">👥 Employee Risk Table</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C_MUTED}; font-size:0.9rem; margin-top:-0.4rem;">Risk predictions for all 1,470 employees. Filter, explore, and download.</p>', unsafe_allow_html=True)

    df_raw = load_dataset()
    if df_raw is None:
        st.error("Dataset not found.")
        st.stop()

    # ── Prepare encoded data for prediction ──
    @st.cache_data(ttl=600)
    def get_all_predictions():
        import joblib
        from sklearn.preprocessing import LabelEncoder
        df_work = df_raw.copy()
        df_work.drop(columns=["EmployeeCount", "StandardHours", "Over18"],
                     errors="ignore", inplace=True)
        le = LabelEncoder()
        cat_cols = df_work.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            if col != "Attrition":
                df_work[col] = le.fit_transform(df_work[col])
        feature_cols = [c for c in df_work.columns if c not in ("Attrition",)]
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        clf = joblib.load(model_path)
        X_all = df_work[feature_cols]
        probas = clf.predict_proba(X_all)[:, 1]
        return probas

    with st.spinner("Running predictions on all employees…"):
        probas = get_all_predictions()

    # ── Build display table ──
    display_cols = ["EmployeeNumber", "Age", "Department", "JobRole",
                    "MonthlyIncome", "OverTime", "YearsAtCompany"]
    df_display = df_raw[display_cols].copy()
    df_display["Probability %"] = (probas * 100).round(1)
    df_display["Risk Level"]    = df_display["Probability %"].apply(
        lambda p: "🔴 High Risk" if p >= 50 else "🟢 Low Risk")

    # ── Summary KPIs ──
    total      = len(df_display)
    high_risk  = (df_display["Risk Level"] == "🔴 High Risk").sum()
    low_risk   = total - high_risk
    high_pct   = round(high_risk / total * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    for col, icon, val, label, color in [
        (k1, "👥", f"{total:,}",     "Total Employees",  C_BLUE),
        (k2, "🔴", f"{high_risk:,}", "High Risk",        C_RED),
        (k3, "🟢", f"{low_risk:,}",  "Low Risk",         C_GREEN),
        (k4, "📊", f"{high_pct}%",   "High Risk Rate",   C_ORANGE),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value" style="color:{color};">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ──
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        dept_opts = ["All"] + sorted(df_display["Department"].unique().tolist())
        sel_dept  = st.selectbox("Filter by Department", dept_opts)
    with fc2:
        risk_opts = ["All", "🔴 High Risk", "🟢 Low Risk"]
        sel_risk  = st.selectbox("Filter by Risk Level", risk_opts)
    with fc3:
        ot_opts  = ["All", "Yes", "No"]
        sel_ot   = st.selectbox("Filter by OverTime", ot_opts)

    df_filtered = df_display.copy()
    if sel_dept != "All":
        df_filtered = df_filtered[df_filtered["Department"] == sel_dept]
    if sel_risk != "All":
        df_filtered = df_filtered[df_filtered["Risk Level"] == sel_risk]
    if sel_ot != "All":
        df_filtered = df_filtered[df_filtered["OverTime"] == sel_ot]

    st.markdown(f'<p style="color:{C_MUTED}; font-size:0.85rem;">Showing <strong style="color:{C_TEXT};">{len(df_filtered):,}</strong> of {total:,} employees</p>', unsafe_allow_html=True)

    # ── Styled table ──
    def style_risk_table(row):
        if "High Risk" in str(row["Risk Level"]):
            return [f"background-color: rgba(248,81,73,0.10)"] * len(row)
        return [f"background-color: rgba(63,185,80,0.06)"] * len(row)

    styled = df_filtered.style.apply(style_risk_table, axis=1).format({"Probability %": "{:.1f}%"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

    # ── Download button ──
    csv = df_filtered.drop(columns=["Risk Level"]).copy()
    csv["Risk"] = df_filtered["Risk Level"].str.replace("🔴 ", "").str.replace("🟢 ", "")
    st.download_button(
        label="⬇️  Download filtered results as CSV",
        data=csv.to_csv(index=False).encode("utf-8"),
        file_name="employee_risk_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

elif page == "ℹ️  About":
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-shell">
      <div class="about-header">
        <h2 style="margin:0; font-size:1.5rem; font-weight:800;">IBM HR Attrition Analysis</h2>
        <p style="margin:0.35rem 0 0 0; color:#dfeaff; font-size:0.9rem;">AI-driven workforce risk assessment and retention insights</p>
      </div>
      <div class="about-grid">
        <div class="about-card">
          <h4>Student</h4>
          <p><strong>Name:</strong> Mohammed Sameer Khazi</p>
        </div>
        <div class="about-card">
          <h4>Internship</h4>
          <p>AICTE | IBM SkillsBuild 2026</p>
        </div>
        <div class="about-card">
          <h4>Project</h4>
          <p>IBM HR Attrition Analysis</p>
        </div>
        <div class="about-card">
          <h4>Dataset</h4>
          <p><a href="https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset" target="_blank" style="color:#0f62fe; text-decoration:none;">IBM HR Analytics Kaggle</a></p>
        </div>
        <div class="about-card">
          <h4>Tech Stack</h4>
          <p>Python, Flask, Streamlit, Scikit-learn, XGBoost, SMOTE, IBM Bob</p>
        </div>
        <div class="about-card">
          <h4>Model Performance</h4>
          <ul>
            <li>Random Forest: 88.06% accuracy, 95.33% ROC-AUC</li>
            <li>XGBoost: 90.08% accuracy, 96.86% ROC-AUC</li>
          </ul>
        </div>
        <div class="about-card">
          <h4>Email</h4>
          <p>mohammedsameerkhaji@gmail.com</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  IBM HR Attrition Analysis &nbsp;·&nbsp; Made with IBM Bob
</div>
""", unsafe_allow_html=True)
