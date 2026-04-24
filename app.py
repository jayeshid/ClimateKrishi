import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIMATEKRISHI AI : An AI-Powered Platform for Climate Smart Decision Making in Rice Farming",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
    body {
        background: linear-gradient(180deg, #f7fdf8 0%, #eafaf1 40%, #ffffff 100%);
        color: #1f2937;
    }

    .css-1d391kg {background-color: rgba(255,255,255,0.75);} /* main container fallback */

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.9);
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }

    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 999px;
        font-weight: 600;
        padding: 0.75rem 1.2rem;
        border: none;
        color: #374151;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #16a34a 0%, #0f766e 100%);
        color: white !important;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.15);
    }

    .stButton button {
        background: linear-gradient(135deg, #16a34a 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 0.85rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(16, 185, 129, 0.22);
    }

    .css-1l02zno {
        background: rgba(255,255,255,0.9);
        border-radius: 18px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }

    .stMetric {
        border-radius: 18px;
        background: rgba(255,255,255,0.95);
        padding: 1rem 1.2rem;
        border-left: 4px solid #10b981;
    }

    .css-1avcm0n {
        border-radius: 18px;
        background: rgba(255,255,255,0.95);
        padding: 1rem;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.09);
    }

    .stSidebar .css-1d391kg {
        background: linear-gradient(180deg, #ecfdf5 0%, #dcfce7 100%);
    }

    .stSidebar [data-testid="stMarkdownContainer"] p,
    .stSidebar [data-testid="stMarkdownContainer"] li {
        color: #334155;
    }

    .sidebar .block-container {
        padding-top: 2rem;
    }

    .css-1kyxreq {
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models + Scalers ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all models and scalers with error handling"""
    try:
        model_conventional  = joblib.load("model_conventional.pkl")
        scaler_conventional = joblib.load("scaler_conventional.pkl")
    except:
        model_conventional  = None
        scaler_conventional = None

    model_organic = None
    scaler_organic = None
    has_organic = False
    
    try:
        model_organic = joblib.load("model_organic.pkl")
        scaler_organic = joblib.load("scaler_organic.pkl")
        
        # Test if organic model works with standard feature names
        test_input = pd.DataFrame([[135, 50, 35, 20]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
        try:
            _ = scaler_organic.transform(test_input)
            has_organic = True
        except (ValueError, KeyError):
            # Try with organic feature names
            try:
                test_input = pd.DataFrame([[10000, 1500]], columns=['Manure_rate', 'Compost_rate'])
                _ = scaler_organic.transform(test_input)
                has_organic = True
            except:
                has_organic = False
                model_organic = None
                scaler_organic = None
    except:
        pass
    
    return model_conventional, scaler_conventional, model_organic, scaler_organic, has_organic

model_conventional, scaler_conventional, model_organic, scaler_organic, has_organic = load_models()

# ── Sidebar Info ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 CLIMATEKRISHI AI")
    st.markdown("---")
    st.markdown("#### Quick Start")
    st.markdown("""
    - Choose your system and mode.
    - Enter recommended fertiliser or amendment rates.
    - Click Predict or Compare for instant insights.
    """)
    st.markdown("---")
    st.markdown("#### Model Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if model_conventional:
            st.success("Conventional")
        else:
            st.error("Conventional")
    with status_col2:
        if has_organic:
            st.success("Organic")
        else:
            st.warning("Organic")
    st.markdown("---")
    st.markdown("**Data Source:** ICAR-IIRR & KVK Medak, Telangana")
    st.markdown("**Method:** Ridge Regression, LCA-based")
    st.markdown("---")
    st.markdown("#### What this app does")
    st.markdown("""
    - Compare conventional and organic impact profiles.
    - Explore blend transitions and costs.
    - Estimate field emissions by irrigation and amendments.
    - View model validation and input intensity.
    """)

# ── Helper Constants ───────────────────────────────────────────────────────────
CONV_RANGES = {'N': (120, 150), 'P': (40, 60), 'K': (30, 40), 'Zn': (10, 30)}
ORG_RANGES  = {'Manure': (5000, 15000), 'Compost': (1000, 2000)}

COST_RATES = {
    'N_synthetic':  6.5,      # ₹/kg N
    'P_synthetic':  28.0,     # ₹/kg P
    'K_synthetic':  15.0,     # ₹/kg K
    'Zn_synthetic': 120.0,    # ₹/kg Zn
    'Manure':       0.8,      # ₹/kg FYM
    'Compost':      3.5,      # ₹/kg Compost
}

IMPACT_DATA = {
    "Input": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Zinc (Zn)"],
    "Global Warming (kg CO₂-eq)": [4.964134, 2.906595, 3.016340, 0.777299],
    "Terrestrial Acidification (kg SO₂-eq)": [0.021555, 0.014359, 0.012404, 0.006452],
    "Freshwater Eutrophication (kg P-eq)": [0.001469, 0.001000, 0.000679, 0.000460],
    "Terrestrial Ecotoxicity (CTUe)": [5.186745, 4.168297, 2.671163, 612.915864],
}

EMISSION_COLORS = {
    "CH₄": "#1f77b4",
    "N₂O": "#ff7f0e",
    "NO₃": "#2ca02c",
    "NH₃": "#d62728",
    "PO₄": "#9467bd",
}

IMPACT_LABELS = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
IMPACT_UNITS  = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
IMPACT_FORMATS = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]

# ── Helper Functions ───────────────────────────────────────────────────────────
def predict_conventional(N, P, K, Zn):
    """Predict conventional farming impacts"""
    try:
        inputs = pd.DataFrame([[N, P, K, Zn]], columns=['N_rate', 'P_rate', 'K_rate', 'Zn_rate'])
        if scaler_conventional:
            scaled = scaler_conventional.transform(inputs)
            pred = model_conventional.predict(scaled)[0]
        else:
            pred = model_conventional.predict(inputs)[0]
        return pred[0], pred[1], pred[2], pred[3] if len(pred) >= 4 else (pred[0], pred[1], pred[2], 0)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return 0, 0, 0, 0

def predict_organic(manure, compost):
    """Predict organic farming impacts"""
    try:
        inputs = pd.DataFrame([[manure, compost]], columns=['Manure_rate', 'Compost_rate'])
        if scaler_organic:
            scaled = scaler_organic.transform(inputs)
            pred = model_organic.predict(scaled)[0]
        else:
            pred = model_organic.predict(inputs)[0]
        return pred[0], pred[1], pred[2], pred[3] if len(pred) >= 4 else (pred[0], pred[1], pred[2], 0)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return 0, 0, 0, 0

def blend(conv_output, org_output, alpha):
    """Blend conventional and organic predictions"""
    return (1 - alpha) * np.array(conv_output) + alpha * np.array(org_output)

def calc_cost(N, P, K, Zn, manure, compost, alpha):
    """Calculate fertiliser cost"""
    conv_cost = (N * COST_RATES['N_synthetic'] +
                 P * COST_RATES['P_synthetic'] +
                 K * COST_RATES['K_synthetic'] +
                 Zn * COST_RATES['Zn_synthetic'])
    org_cost = (manure * COST_RATES['Manure'] +
                compost * COST_RATES['Compost'])
    return (1 - alpha) * conv_cost + alpha * org_cost

def validate_conventional(N, P, K, Zn):
    """Validate conventional inputs"""
    vals = {'N': N, 'P': P, 'K': K, 'Zn': Zn}
    return [
        f"**{k}** = {v} kg/ha (valid: {CONV_RANGES[k][0]}–{CONV_RANGES[k][1]} kg/ha)"
        for k, v in vals.items()
        if not (CONV_RANGES[k][0] <= v <= CONV_RANGES[k][1])
    ]

def validate_organic(manure, compost):
    """Validate organic inputs"""
    vals = {'Manure': manure, 'Compost': compost}
    return [
        f"**{k}** = {v:.0f} kg/ha (valid: {ORG_RANGES[k][0]:,}–{ORG_RANGES[k][1]:,} kg/ha)"
        for k, v in vals.items()
        if not (ORG_RANGES[k][0] <= v <= ORG_RANGES[k][1])
    ]


def compute_field_emissions(synthetic_n, synthetic_p, irrigation, amendment_1, amendment_2):
    """Estimate field emissions using simplified IPCC and SALCA factors"""
    irrigation_factors = {
        'Fully Flooded': {'ch4': 0.793, 'n2o': 0.00471, 'nh3_mult': 1.0},
        'Alternate Wetting and Drying': {'ch4': 0.12, 'n2o': 0.0055, 'nh3_mult': 1.05},
        'Rainfed': {'ch4': 0.05, 'n2o': 0.0062, 'nh3_mult': 1.08},
    }
    factors = irrigation_factors.get(irrigation, irrigation_factors['Fully Flooded'])

    ch4 = synthetic_n * factors['ch4']
    n2o = synthetic_n * factors['n2o']
    no3 = synthetic_n * 0.03986
    po4 = synthetic_p * 0.03065
    nh3 = synthetic_n * 0.364 * factors['nh3_mult'] + synthetic_p * 0.005

    amendment_count = sum(1 for a in [amendment_1, amendment_2] if a != 'None')
    if amendment_count > 0:
        n2o *= 0.95
        nh3 *= 1.08
        ch4 *= 1.02

    return {
        'CH4': round(ch4, 3),
        'N2O': round(n2o, 4),
        'NO3': round(no3, 4),
        'NH3': round(nh3, 4),
        'PO4': round(po4, 4),
    }


def build_impact_dataframe(values, label):
    return pd.DataFrame({
        "Impact Category": ["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
        "Value": values,
        "Source": [label] * 4
    })


def build_comparison_dataframe(outA, outB):
    return pd.DataFrame({
        "Impact Category": ["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
        "Combination A": outA,
        "Combination B": outB,
    })


def style_chart(chart):
    return chart.configure_view(
        strokeOpacity=0
    ).configure_axis(
        gridColor="#d1d5db",
        gridDash=[2, 2],
        labelFontSize=11,
        titleFontSize=13,
        labelColor="#334155",
        titleColor="#111827",
        labelPadding=10,
        tickColor="#9ca3af"
    ).configure_legend(
        labelFontSize=11,
        titleFontSize=12,
        titleColor="#111827",
        symbolSize=120
    ).configure_title(
        fontSize=15,
        anchor="start",
        color="#0f172a"
    )


def build_impact_chart(df):
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Value:Q", title="Impact value"),
        y=alt.Y("Impact Category:N", sort=["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"]),
        color=alt.Color("Source:N", title="Scenario", scale=alt.Scale(range=["#1f77b4", "#ff7f0e", "#2ca02c"])),
        xOffset=alt.XOffset("Source:N"),
        tooltip=["Impact Category", "Value", "Source"]
    ).properties(height=320, width=700)
    return style_chart(chart)


def build_gradient_impact_chart(df):
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Value:Q", title="Impact value"),
        y=alt.Y("Source:N", sort=["Conventional", "Blend", "Organic"], title="Scenario",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelPadding=10)),
        color=alt.Color("Source:N", title="Scenario", scale=alt.Scale(range=["#ff7f0e", "#1f77b4", "#2ca02c"])),
        tooltip=["Impact Category", "Source", alt.Tooltip("Value:Q", format=",.3f")]
    ).properties(width=700, height=140)

    return style_chart(
        chart.facet(
            row=alt.Row("Impact Category:N",
                        sort=["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
                        title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12, labelPadding=12, labelColor="#334155"))
        ).properties(title="Impact Breakdown", spacing=24)
    )


def build_blend_chart(conv_out, org_out):
    rows = []
    for alpha in np.linspace(0, 1, 11):
        blend_values = (1 - alpha) * np.array(conv_out) + alpha * np.array(org_out)
        for idx, label in enumerate(["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"]):
            rows.append({"Organic %": alpha * 100, "Impact Category": label, "Impact Value": blend_values[idx]})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X("Organic %:Q", title="Organic blend (%)"),
        y=alt.Y("Impact Value:Q", title="Predicted impact"),
        color=alt.Color("Impact Category:N", title="Impact"),
        tooltip=["Impact Category", alt.Tooltip("Impact Value:Q", format=",.3f"), alt.Tooltip("Organic %:Q", format=".0f")]
    ).properties(height=340, width=700)
    return style_chart(chart)


def build_comparison_chart(df):
    stacked = df.melt(id_vars=["Impact Category"], value_vars=["Combination A", "Combination B"], var_name="Combination", value_name="Value")
    chart = alt.Chart(stacked).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Value:Q", title="Impact value"),
        y=alt.Y("Combination:N", sort=["Combination A", "Combination B"], title="Combination",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelPadding=8, titlePadding=10)),
        color=alt.Color("Combination:N", scale=alt.Scale(range=["#3b82f6", "#f97316"])),
        tooltip=["Impact Category", "Combination", alt.Tooltip("Value:Q", format=",.3f")]
    ).properties(width=720, height=140)

    return style_chart(
        chart.facet(
            row=alt.Row("Impact Category:N",
                        sort=["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
                        title=None,
                        header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12, labelPadding=12, labelColor="#334155"))
        ).properties(title="Impact Breakdown", spacing=24)
    )


def build_cost_vs_impact_scatter(outA, outB, costA, costB, labelA="Combination A", labelB="Combination B"):
    """Build scatter plot of cost vs GWP (global warming potential)"""
    scatter_data = pd.DataFrame({
        "Cost (₹/ha)": [costA, costB],
        "GWP (kg CO₂-eq)": [outA[0], outB[0]],
        "Combination": [labelA, labelB]
    })
    chart = alt.Chart(scatter_data).mark_circle(size=220).encode(
        x=alt.X("Cost (₹/ha):Q", title="Input Cost (₹/ha)"),
        y=alt.Y("GWP (kg CO₂-eq):Q", title="Global Warming Potential"),
        color=alt.Color("Combination:N", scale=alt.Scale(range=["#3b82f6", "#f97316"])),
        tooltip=["Combination", alt.Tooltip("Cost (₹/ha):Q", format=",.0f"), alt.Tooltip("GWP (kg CO₂-eq):Q", format=",.2f")]
    ).properties(height=340, width=700)
    return style_chart(chart)


def build_emission_pie_chart(emissions):
    """Build pie chart for emission share breakdown"""
    pie_data = pd.DataFrame({
        "Emission": ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"],
        "Value": [emissions['CH4'], emissions['N2O'], emissions['NO3'], emissions['NH3'], emissions['PO4']]
    })
    pie_data["Percent"] = pie_data["Value"] / pie_data["Value"].sum()
    pie_data["LegendLabel"] = pie_data.apply(
        lambda row: f"{row['Emission']} ({row['Percent']:.1%})", axis=1
    )

    pie = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("Value:Q"),
        color=alt.Color("LegendLabel:N", scale=alt.Scale(domain=pie_data["LegendLabel"].tolist(), range=[EMISSION_COLORS[e] for e in pie_data["Emission"]]), legend=alt.Legend(title="Emission", labelLimit=300)),
        tooltip=["Emission", alt.Tooltip("Value:Q", format=",.3f"), alt.Tooltip("Percent:Q", format=".1%")]
    ).properties(height=340, width=420)

    return style_chart(pie)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    hero_left, hero_right = st.columns([2.4, 1.6])
    with hero_left:
        st.markdown("""
        ### Welcome to CLIMATEKRISHI AI
        ## Smart farming for a sustainable Bharat 
        """)
        st.markdown("""
        Predict and compare the **full environmental impact** of fertiliser application in rice cultivation — including upstream production emissions and field-level emissions, powered by ISO 14040/44-compliant LCA.
        """)
        st.markdown("""
        **Fast insights, actionable comparisons, and cost-aware recommendations for rice farmers and agronomists.**
        """)
        c1, c2 = st.columns([1,1])
        with c1:
            st.button("Start Predicting")
        with c2:
            st.button("View Documentation")
    with hero_right:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ecfdf5 0%, #bbf7d0 100%); border-radius: 28px; padding: 2rem; text-align: center; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);'>
            <h3 style='color:#064e3b; margin-bottom: 0.75rem;'>Climate Smart LCA Tool</h3>
            <p style='color:#065f46; margin-bottom: 1.5rem;'>Enter your inputs and instantly compare the environmental cost of rice fertiliser choices.</p>
            <div style='font-size: 3rem; line-height: 1; color: #059669;'>🌾</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
with st.container():
    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.markdown("""
        <div style='background:white; padding:1.2rem; border-radius:20px; box-shadow:0 16px 30px rgba(15,23,42,0.08);'>
            <h4 style='color:#047857;'>Real-time soil insights</h4>
            <p style='color:#334155;'>Understand the environmental footprint of each nutrient input.</p>
        </div>
        """, unsafe_allow_html=True)
    with card2:
        st.markdown("""
        <div style='background:white; padding:1.2rem; border-radius:20px; box-shadow:0 16px 30px rgba(15,23,42,0.08);'>
            <h4 style='color:#047857;'>LCA Predictor</h4>
            <p style='color:#334155;'>Compare conventional and organic strategies at a glance.</p>
        </div>
        """, unsafe_allow_html=True)
    with card3:
        st.markdown("""
        <div style='background:white; padding:1.2rem; border-radius:20px; box-shadow:0 16px 30px rgba(15,23,42,0.08);'>
            <h4 style='color:#047857;'>Field Emission Calculator</h4>
            <p style='color:#334155;'>Estimate CH₄, N₂O and nutrient emissions by irrigation type.</p>
        </div>
        """, unsafe_allow_html=True)
    with card4:
        st.markdown("""
        <div style='background:white; padding:1.2rem; border-radius:20px; box-shadow:0 16px 30px rgba(15,23,42,0.08);'>
            <h4 style='color:#047857;'>Model Validation</h4>
            <p style='color:#334155;'>Built with real-world rice cultivation datasets from ICAR-IIRR & KVK.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔬 LCA Impact Predictor", "🎚️ Organic–Conventional Gradient", "🌱 Field Emission Calculator", "📊 Model Information"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LCA IMPACT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("LCA Impact Predictor")
    st.markdown(
        "Predict environmental impact using either conventional or organic fertiliser systems. "
        "Compare single inputs or two fertiliser combinations for better decision making."
    )
    st.markdown("---")

    system = st.radio("Select System", ["🧪 Conventional", "🌿 Organic"], horizontal=True)
    mode = st.radio("Select Mode", ["Single Prediction", "Compare Two Combinations"], horizontal=True)
    st.markdown("---")

    if mode == "Single Prediction":
        if system == "🧪 Conventional":
            st.subheader("Enter Conventional Fertiliser Rates")
            st.info("**Recommended Ranges:** N: 120–150 | P: 40–60 | K: 30–40 | Zn: 10–30")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("**🌾 Nitrogen (N)** — Primary nutrient for leaf and grain formation")
                sN  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5, key="lN")
                st.caption("**🌾 Potassium (K)** — Improves grain quality and disease resistance")
                sK  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5, key="lK")
            with col2:
                st.caption("**🌿 Phosphorus (P)** — Strengthens root development and tillering")
                sP  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5, key="lP")
                st.caption("**⚡ Zinc (Zn)** — Micronutrient critical for enzyme function")
                sZn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5, key="lZn")

            warn = validate_conventional(sN, sP, sK, sZn)
            if warn:
                st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in warn))
            else:
                st.success("✅ All inputs within recommended ranges.")

            if st.button("🔍 Predict", use_container_width=True, key="lbtn_conv"):
                if model_conventional:
                    out = predict_conventional(sN, sP, sK, sZn)
                    cost = calc_cost(sN, sP, sK, sZn, 0, 0, 0.0)
                    
                    st.markdown("---")
                    st.subheader("✅ Prediction Complete")
                    
                    # Summary callout
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    with col_summary1:
                        st.metric("💰 Input Cost", f"₹{cost:,.0f}/ha")
                    with col_summary2:
                        st.metric("🌍 Climate Impact", f"{out[0]:,.2f} kg CO₂-eq")
                    with col_summary3:
                        st.metric("⚠️ Highest Risk", "Ecotoxicity" if out[3] > out[1] and out[3] > out[2] else ("Acidification" if out[2] > out[1] else "Eutrophication"))
                    
                    st.markdown("---")
                    st.subheader("📊 Predicted LCA Impact Scores")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🌍 Global Warming",            f"{out[0]:,.2f} kg CO₂-eq")
                        st.metric("💧 Freshwater Eutrophication", f"{out[1]:.6f} kg P-eq")
                    with col2:
                        st.metric("🌫️ Terrestrial Acidification", f"{out[2]:.4f} kg SO₂-eq")
                        st.metric("☠️ Terrestrial Ecotoxicity",   f"{out[3]:,.2f} CTUe")
                    
                    chart_df = build_impact_dataframe(out, "Conventional")
                    st.altair_chart(build_impact_chart(chart_df), use_container_width=True)
                else:
                    st.error("❌ Conventional model not available.")

        else:
            st.subheader("Enter Organic Amendment Rates")
            st.info("**Recommended Ranges:** Manure: 5,000–15,000 | Compost: 1,000–2,000 kg/ha")

            col1, col2 = st.columns(2)
            with col1:
                sManure  = st.number_input("Farm Yard Manure (kg/ha)", min_value=0.0, value=10000.0, step=100.0, key="lManure")
            with col2:
                sCompost = st.number_input("Compost (kg/ha)", min_value=0.0, value=1500.0, step=50.0, key="lCompost")

            warn_org = validate_organic(sManure, sCompost)
            if warn_org:
                st.error("🚨 Out of range:\n\n" + "\n".join(f"- {w}" for w in warn_org))
            else:
                st.success("✅ All inputs within recommended ranges.")

            if st.button("🔍 Predict", use_container_width=True, key="lbtn_org"):
                if has_organic and model_organic:
                    out = predict_organic(sManure, sCompost)
                    cost = calc_cost(0, 0, 0, 0, sManure, sCompost, 1.0)
                    
                    st.markdown("---")
                    st.subheader("✅ Prediction Complete")
                    
                    # Summary callout
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    with col_summary1:
                        st.metric("💰 Amendment Cost", f"₹{cost:,.0f}/ha")
                    with col_summary2:
                        st.metric("🌍 Climate Impact", f"{out[0]:,.2f} kg CO₂-eq")
                    with col_summary3:
                        st.metric("⚠️ Highest Risk", "Ecotoxicity" if out[3] > out[1] and out[3] > out[2] else ("Acidification" if out[2] > out[1] else "Eutrophication"))
                    
                    st.markdown("---")
                    st.subheader("📊 Predicted LCA Impact Scores")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🌍 Global Warming",            f"{out[0]:,.2f} kg CO₂-eq")
                        st.metric("💧 Freshwater Eutrophication", f"{out[1]:.6f} kg P-eq")
                    with col2:
                        st.metric("🌫️ Terrestrial Acidification", f"{out[2]:.4f} kg SO₂-eq")
                        st.metric("☠️ Terrestrial Ecotoxicity",   f"{out[3]:,.2f} CTUe")
                    
                    chart_df = build_impact_dataframe(out, "Organic")
                    st.altair_chart(build_impact_chart(chart_df), use_container_width=True)
                else:
                    st.error("❌ Organic model not available. Please ensure model files are present.")

    else:
        if system == "🧪 Conventional":
            st.subheader("Compare Two Conventional Fertiliser Combinations")
            st.markdown("Compare two conventional NPKZn input sets to see which has lower impact.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🅰️ Combination A")
                NA = st.number_input("Nitrogen (N) — A",   min_value=0.0, value=120.0, step=0.5, key="cNA")
                PA = st.number_input("Phosphorus (P) — A", min_value=0.0, value=40.0,  step=0.5, key="cPA")
                KA = st.number_input("Potassium (K) — A",  min_value=0.0, value=30.0,  step=0.5, key="cKA")
                ZnA = st.number_input("Zinc (Zn) — A",      min_value=0.0, value=10.0,  step=0.5, key="cZnA")
            with col_b:
                st.markdown("### 🅱️ Combination B")
                NB = st.number_input("Nitrogen (N) — B",   min_value=0.0, value=150.0, step=0.5, key="cNB")
                PB = st.number_input("Phosphorus (P) — B", min_value=0.0, value=60.0,  step=0.5, key="cPB")
                KB = st.number_input("Potassium (K) — B",  min_value=0.0, value=40.0,  step=0.5, key="cKB")
                ZnB = st.number_input("Zinc (Zn) — B",      min_value=0.0, value=30.0,  step=0.5, key="cZnB")

            if st.button("🔍 Compare Combinations", use_container_width=True, key="compare_conv"):
                outA = predict_conventional(NA, PA, KA, ZnA)
                outB = predict_conventional(NB, PB, KB, ZnB)
                costA = calc_cost(NA, PA, KA, ZnA, 0, 0, 0.0)
                costB = calc_cost(NB, PB, KB, ZnB, 0, 0, 0.0)
                
                # Summary callout
                gwp_winner = "A" if outA[0] < outB[0] else "B" if outB[0] < outA[0] else "Tie"
                cost_winner = "A" if costA < costB else "B" if costB < costA else "Tie"
                st.markdown("---")
                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.success(f"🌍 Lower GWP: **Combination {gwp_winner}** ({min(outA[0], outB[0]):,.2f} kg CO₂-eq)")
                with col_summary2:
                    st.info(f"💰 Cheaper: **Combination {cost_winner}** (₹{min(costA, costB):,.0f}/ha)")
                st.markdown("---")
                
                st.subheader("📊 Comparison Results")
                categories = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
                units      = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
                formats    = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]
                for i, cat in enumerate(categories):
                    col1, col2, col3 = st.columns([2,2,1])
                    with col1:
                        st.metric(f"{cat} — A", f"{formats[i].format(outA[i])} {units[i]}")
                    with col2:
                        st.metric(f"{cat} — B", f"{formats[i].format(outB[i])} {units[i]}")
                    with col3:
                        winner = "🔽 A" if outA[i] < outB[i] else "🔽 B" if outB[i] < outA[i] else "="
                        st.markdown(f"<br><b>{winner}</b>", unsafe_allow_html=True)
                
                st.markdown("**Impact Breakdown**")
                compare_df = build_comparison_dataframe(outA, outB)
                st.altair_chart(build_comparison_chart(compare_df), use_container_width=True)

                st.markdown("---")
                st.markdown("**Cost vs GWP Trade-off**")
                st.altair_chart(build_cost_vs_impact_scatter(outA, outB, costA, costB, 
                                "Comb. A", "Comb. B"), use_container_width=True)
        else:
            st.subheader("Compare Two Organic Amendment Combinations")
            st.markdown("Compare two organic amendment strategies to see how they affect impact scores.")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 🅰️ Combination A")
                M_A = st.number_input("Farm Yard Manure — A", min_value=0.0, value=10000.0, step=100.0, key="cMA")
                C_A = st.number_input("Compost — A", min_value=0.0, value=1500.0, step=50.0, key="cCA")
            with col_b:
                st.markdown("### 🅱️ Combination B")
                M_B = st.number_input("Farm Yard Manure — B", min_value=0.0, value=12000.0, step=100.0, key="cMB")
                C_B = st.number_input("Compost — B", min_value=0.0, value=1800.0, step=50.0, key="cCB")

            if st.button("🔍 Compare Combinations", use_container_width=True, key="compare_org"):
                outA = predict_organic(M_A, C_A)
                outB = predict_organic(M_B, C_B)
                costA = calc_cost(0, 0, 0, 0, M_A, C_A, 1.0)
                costB = calc_cost(0, 0, 0, 0, M_B, C_B, 1.0)
                
                # Summary callout
                gwp_winner = "A" if outA[0] < outB[0] else "B" if outB[0] < outA[0] else "Tie"
                cost_winner = "A" if costA < costB else "B" if costB < costA else "Tie"
                st.markdown("---")
                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.success(f"🌍 Lower GWP: **Combination {gwp_winner}** ({min(outA[0], outB[0]):,.2f} kg CO₂-eq)")
                with col_summary2:
                    st.info(f"💰 Cheaper: **Combination {cost_winner}** (₹{min(costA, costB):,.0f}/ha)")
                st.markdown("---")
                
                st.subheader("📊 Comparison Results")
                categories = ["🌍 Global Warming", "💧 Freshwater Eutrophication", "🌫️ Terrestrial Acidification", "☠️ Terrestrial Ecotoxicity"]
                units      = ["kg CO₂-eq", "kg P-eq", "kg SO₂-eq", "CTUe"]
                formats    = ["{:,.2f}", "{:.6f}", "{:.4f}", "{:,.2f}"]
                for i, cat in enumerate(categories):
                    col1, col2, col3 = st.columns([2,2,1])
                    with col1:
                        st.metric(f"{cat} — A", f"{formats[i].format(outA[i])} {units[i]}")
                    with col2:
                        st.metric(f"{cat} — B", f"{formats[i].format(outB[i])} {units[i]}")
                    with col3:
                        winner = "🔽 A" if outA[i] < outB[i] else "🔽 B" if outB[i] < outA[i] else "="
                        st.markdown(f"<br><b>{winner}</b>", unsafe_allow_html=True)
                
                st.markdown("**Impact Breakdown**")
                compare_df = build_comparison_dataframe(outA, outB)
                st.altair_chart(build_comparison_chart(compare_df), use_container_width=True)

                st.markdown("---")
                st.markdown("**Cost vs GWP Trade-off**")
                st.altair_chart(build_cost_vs_impact_scatter(outA, outB, costA, costB, 
                                "Comb. A", "Comb. B"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRADIENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Organic–Conventional Transition Analyser")
    st.markdown(
        "Enter your conventional and organic input rates, then use the slider to explore "
        "the environmental and economic impact at any point along the transition."
    )
    st.markdown("---")

    col_conv, col_org = st.columns(2)

    with col_conv:
        st.markdown("### 🧪 Conventional Inputs (kg/ha)")
        N  = st.number_input("Nitrogen (N)",   min_value=0.0, value=135.0, step=0.5, key="gN")
        P  = st.number_input("Phosphorus (P)", min_value=0.0, value=50.0,  step=0.5, key="gP")
        K  = st.number_input("Potassium (K)",  min_value=0.0, value=35.0,  step=0.5, key="gK")
        Zn = st.number_input("Zinc (Zn)",      min_value=0.0, value=20.0,  step=0.5, key="gZn")
        st.caption("Valid ranges — N: 120–150 | P: 40–60 | K: 30–40 | Zn: 10–30")

    with col_org:
        st.markdown("### 🌿 Organic Inputs (kg/ha)")
        manure  = st.number_input("Farm Yard Manure",  min_value=0.0, value=10000.0, step=100.0, key="gManure")
        compost = st.number_input("Compost",           min_value=0.0, value=1500.0,  step=50.0,  key="gCompost")
        st.caption("Valid ranges — Manure: 5,000–15,000 | Compost: 1,000–2,000 kg/ha")

    warnings = validate_conventional(N, P, K, Zn)
    if warnings:
        st.warning("⚠️ Conventional inputs out of range:\n\n" + "\n".join(f"- {w}" for w in warnings))

    st.markdown("---")

    st.subheader("🎚️ Organic Input Fraction")
    alpha = st.slider(
        "0% = Fully Conventional    →    100% = Fully Organic",
        min_value=0, max_value=100, value=0, step=5
    ) / 100

    if alpha == 0.0:
        st.info("🧪 Currently showing: **Fully Conventional**")
    elif alpha == 1.0:
        st.success("🌿 Currently showing: **Fully Organic**")
    else:
        st.info(f"🔄 Currently showing: **{int(alpha*100)}% Organic / {int((1-alpha)*100)}% Conventional blend**")

    st.markdown("---")

    if model_conventional and has_organic and model_organic:
        conv_out  = predict_conventional(N, P, K, Zn)
        org_out   = predict_organic(manure, compost)
        blend_out = blend(conv_out, org_out, alpha)

        conv_cost  = calc_cost(N, P, K, Zn, manure, compost, 0.0)
        org_cost   = calc_cost(N, P, K, Zn, manure, compost, 1.0)
        blend_cost = calc_cost(N, P, K, Zn, manure, compost, alpha)

        col_sidebar, col_main = st.columns([1, 3])
        with col_sidebar:
            st.markdown("""
                <div style='background:white; border-radius:24px; padding:1.2rem; box-shadow:0 18px 35px rgba(15,23,42,0.08); margin-bottom:1rem;'>
                    <div style='display:flex; align-items:center; gap:0.75rem;'>
                        <div style='width:12px; height:48px; background:#1d4ed8; border-radius:999px;'></div>
                        <div>
                            <h4 style='margin:0; color:#1d4ed8;'>Conventional</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Current baseline synthetic system.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {conv_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{conv_cost:,.0f}/ha</p>
                </div>
            """.format(conv_gwp=conv_out[0], conv_cost=conv_cost), unsafe_allow_html=True)
            st.markdown("""
                <div style='background:white; border-radius:24px; padding:1.2rem; box-shadow:0 18px 35px rgba(15,23,42,0.08); margin-bottom:1rem;'>
                    <div style='display:flex; align-items:center; gap:0.75rem;'>
                        <div style='width:12px; height:48px; background:#f97316; border-radius:999px;'></div>
                        <div>
                            <h4 style='margin:0; color:#b45309;'>Blend</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Current organic/conventional transition.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {blend_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{blend_cost:,.0f}/ha</p>
                </div>
            """.format(blend_gwp=blend_out[0], blend_cost=blend_cost), unsafe_allow_html=True)
            st.markdown("""
                <div style='background:white; border-radius:24px; padding:1.2rem; box-shadow:0 18px 35px rgba(15,23,42,0.08);'>
                    <div style='display:flex; align-items:center; gap:0.75rem;'>
                        <div style='width:12px; height:48px; background:#16a34a; border-radius:999px;'></div>
                        <div>
                            <h4 style='margin:0; color:#15803d;'>Organic</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Full organic amendment values.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {org_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{org_cost:,.0f}/ha</p>
                </div>
            """.format(org_gwp=org_out[0], org_cost=org_cost), unsafe_allow_html=True)

        with col_main:
            st.subheader("📊 Environmental Impact at Selected Blend")
            st.caption("Includes upstream fertiliser production + field-level emissions (N₂O, NO₃, NH₃, PO₄)")
            
            # Summary metrics
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            gwp_reduction = ((conv_out[0] - blend_out[0]) / conv_out[0] * 100) if conv_out[0] != 0 else 0
            with col_kpi1:
                st.metric("📉 GWP Change", f"{gwp_reduction:+.1f}%", delta_color="inverse")
            with col_kpi2:
                cost_delta_pct = ((blend_cost - conv_cost) / conv_cost * 100) if conv_cost != 0 else 0
                st.metric("💰 Cost Change", f"{cost_delta_pct:+.1f}%", delta_color="normal")
            with col_kpi3:
                acidif_reduction = ((conv_out[2] - blend_out[2]) / conv_out[2] * 100) if conv_out[2] != 0 else 0
                st.metric("Acidification", f"{acidif_reduction:+.1f}%", delta_color="inverse")
            with col_kpi4:
                eutroph_reduction = ((conv_out[1] - blend_out[1]) / conv_out[1] * 100) if conv_out[1] != 0 else 0
                st.metric("Eutrophication", f"{eutroph_reduction:+.1f}%", delta_color="inverse")
        
        st.markdown("---")

        for i, (label, unit, fmt) in enumerate(zip(IMPACT_LABELS, IMPACT_UNITS, IMPACT_FORMATS)):
            col1, col2, col3 = st.columns(3)
            delta_pct = ((blend_out[i] - conv_out[i]) / conv_out[i]) * 100 if conv_out[i] != 0 else 0
            with col1:
                st.metric(f"{label} — Conv.", f"{fmt.format(conv_out[i])} {unit}")
            with col2:
                st.metric(f"{label} — Blend", f"{fmt.format(blend_out[i])} {unit}",
                          delta=f"{delta_pct:+.1f}% vs Conv.")
            with col3:
                st.metric(f"{label} — Organic", f"{fmt.format(org_out[i])} {unit}")

        st.markdown("---")

        blend_df = pd.concat([
            build_impact_dataframe(conv_out, "Conventional"),
            build_impact_dataframe(blend_out, "Blend") ,
            build_impact_dataframe(org_out, "Organic")
        ])
        st.altair_chart(build_gradient_impact_chart(blend_df), use_container_width=True)

        st.markdown("---")
        st.subheader("💰 Input Cost Analysis (₹/ha)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧪 Conventional Cost", f"₹{conv_cost:,.0f}/ha")
        with col2:
            cost_delta = blend_cost - conv_cost
            st.metric("🎚️ Blend Cost", f"₹{blend_cost:,.0f}/ha",
                      delta=f"₹{cost_delta:+,.0f} vs Conv.")
        with col3:
            st.metric("🌿 Organic Cost", f"₹{org_cost:,.0f}/ha")

        st.markdown("---")
        st.subheader("📈 Impact Trend Across Organic Blend")
        st.altair_chart(build_blend_chart(conv_out, org_out), use_container_width=True)

        gwp_reduction = conv_out[0] - blend_out[0]
        if gwp_reduction > 0:
            cost_per_co2 = cost_delta / gwp_reduction if gwp_reduction != 0 else 0
            if cost_delta > 0:
                st.info(f"💡 Reducing **{gwp_reduction:.1f} kg CO₂-eq/ha** costs an additional **₹{cost_per_co2:,.1f} per kg CO₂ avoided**")
            else:
                st.success(f"✅ Reducing **{gwp_reduction:.1f} kg CO₂-eq/ha** while **saving ₹{abs(cost_delta):,.0f}/ha**")
        elif gwp_reduction < 0:
            st.warning("⚠️ This blend increases GWP compared to conventional — consider adjusting your organic inputs.")
        else:
            st.info("No GWP change at this blend point.")

        with st.expander("📋 Full Comparison Table"):
            categories_clean = ["Global Warming (kg CO₂-eq)", "Freshwater Eutrophication (kg P-eq)",
                                "Terrestrial Acidification (kg SO₂-eq)", "Terrestrial Ecotoxicity (CTUe)"]
            comparison_df = pd.DataFrame({
                "Impact Category" : categories_clean,
                "Conventional"    : [IMPACT_FORMATS[i].format(conv_out[i])  for i in range(4)],
                f"Blend ({int(alpha*100)}% Org)": [IMPACT_FORMATS[i].format(blend_out[i]) for i in range(4)],
                "Full Organic"    : [IMPACT_FORMATS[i].format(org_out[i])   for i in range(4)],
                "Δ Conv→Blend"    : [f"{((blend_out[i]-conv_out[i])/conv_out[i])*100:+.1f}%" if conv_out[i] != 0 else "N/A" for i in range(4)],
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.error("❌ Required models not fully available. Need both conventional and organic models for gradient analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FIELD EMISSION CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Field Emission Calculator")
    st.markdown("Calculate direct field emissions from fertiliser application based on IPCC and SALCA methodologies.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        irrigation = st.selectbox("Irrigation Mode", ["Fully Flooded", "Alternate Wetting and Drying", "Rainfed"], key="irrigation")
        synthetic_n = st.number_input("Synthetic Nitrogen — FSN (kg/ha)", min_value=0.0, value=135.0, step=0.5, key="emN")
        synthetic_p = st.number_input("Synthetic Phosphorus — P (kg/ha)", min_value=0.0, value=50.0, step=0.5, key="emP")
    with col2:
        amendment_1 = st.selectbox("Organic Amendment 1", ["None", "Farm Yard Manure", "Compost"], key="am1")
        amendment_2 = st.selectbox("Organic Amendment 2", ["None", "Farm Yard Manure", "Compost"], key="am2")
        st.caption("Organic amendments adjust emissions based on typical nutrient release and decomposition.")

    if st.button("📈 Calculate Field Emissions", use_container_width=True, key="calc_emissions"):
        emissions = compute_field_emissions(synthetic_n, synthetic_p, irrigation, amendment_1, amendment_2)
        st.subheader("Field Emission Results (kg/ha/season)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Methane (CH₄)", f"{emissions['CH4']:,} kg/ha/season")
            st.metric("⚡ Nitrous Oxide (N₂O)", f"{emissions['N2O']:,} kg/ha/season")
        with col2:
            st.metric("💧 Phosphate (PO₄)", f"{emissions['PO4']:,} kg/ha/season")
            st.metric("💦 Nitrate (NO₃)", f"{emissions['NO3']:,} kg/ha/season")
        
        st.markdown("---")
        st.markdown("**💡 Why it matters:** Reducing N₂O and CH₄ emissions improves air quality, reduces climate impact, and can enhance soil health and long-term yield stability.")
        st.markdown("---")
        
        st.subheader("Summary")
        summary_df = pd.DataFrame({
            "Emission": ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"],
            "Value (kg/ha/season)": [emissions['CH4'], emissions['N2O'], emissions['NO3'], emissions['NH3'], emissions['PO4']],
            "Methodology": ["IPCC", "IPCC", "IPCC", "IPCC", "SALCA"],
        })
        st.table(summary_df)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Emission Magnitude**")
            emission_chart_df = pd.DataFrame({
                "Emission": ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"],
                "Value": [emissions['CH4'], emissions['N2O'], emissions['NO3'], emissions['NH3'], emissions['PO4']]
            })
            bar_chart = alt.Chart(emission_chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Value:Q", title="kg/ha/season"),
                y=alt.Y("Emission:N", sort=["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"]),
                color=alt.Color("Emission:N", legend=None, scale=alt.Scale(domain=list(EMISSION_COLORS.keys()), range=list(EMISSION_COLORS.values()))),
                tooltip=["Emission", alt.Tooltip("Value:Q", format=",.3f")]
            ).properties(height=340, width=520)
            st.altair_chart(style_chart(bar_chart), use_container_width=True)
        
        with col_chart2:
            st.markdown("**Emission Share (%)**")
            st.altair_chart(build_emission_pie_chart(emissions), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL VALIDATION & INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📈 Model Validation Plots"):
            st.markdown("#### Validate the trained ML models on test data")
            try:
                if os.path.exists("Plots/plot1_predicted_vs_actual.png"):
                    st.image("Plots/plot1_predicted_vs_actual.png", use_container_width=True,
                            caption="Predicted vs Actual: Points close to red line indicate accuracy.")
                if os.path.exists("Plots/plot2_feature_importance.png"):
                    st.image("Plots/plot2_feature_importance.png", use_container_width=True,
                            caption="Feature Importance: How each input influences each impact.")
                if os.path.exists("Plots/plot3_input_vs_impact.png"):
                    st.image("Plots/plot3_input_vs_impact.png", use_container_width=True,
                            caption="Input vs Impact Trends: How scores change with inputs.")
            except:
                st.info("📊 Validation plots not yet available.")

    with col2:
        with st.expander("⚗️ Fertiliser Impact Intensity"):
            st.markdown("#### Environmental impact per kg of each input")
            st.markdown("*(Calculated independently in OpenLCA)*")
            df_intensity = pd.DataFrame(IMPACT_DATA)
            st.dataframe(df_intensity, use_container_width=True, hide_index=True)
            st.info("💡 **Zinc (Zn)** has dramatically higher ecotoxicity (612.9 CTUe/kg) vs N,P,K (2.7–5.2).")

    st.markdown("---")

    with st.expander("ℹ️ About CLIMATEKRISHI AI & Model Methodology"):
        st.markdown(f"""
        **Algorithm:** Ridge Regression (Multivariate, L2-regularised) — trained separately for
        conventional and organic rice systems.

        **System Boundary:** Cradle-to-field — upstream fertiliser/amendment production, transport,
        and field-level emissions (N₂O, NO₃⁻, NH₃, PO₄³⁻) via IPCC and SALCA methodologies.

        **Training Data:**
        - Conventional: 7,116 LCA records from ecoinvent database
        - Organic: 5,926 LCA records (FYM + Compost based systems)
        - Run in OpenLCA with parametric sampling

        **Primary Data:** ICAR-Indian Institute of Rice Research (IIRR), Hyderabad
        and Krishi Vigyan Kendra (KVK), Medak, Telangana.

        **Impact Assessment:** ReCiPe 2016 Midpoint (H)

        **Cost References:** Indian market rates for fertilisers (2024) and organic amendments.

        **Features:**
        - ✅ Single prediction for conventional or organic systems
        - ✅ Blend analysis: explore transition from conventional to organic
        - ✅ Cost-benefit analysis with Indian market rates
        - ✅ Environmental impact quantification (GWP, Eutrophication, Acidification, Ecotoxicity)
        - ✅ Validation plots and model statistics
        - ✅ Recommendations based on agronomic best practices
        """)

    with st.expander("📋 Datasets Available"):
        st.markdown("""
        **Datasets in `/Data/` folder:**
        - `ML_Dataset_v2_7116.csv` — Conventional cultivation (7,116 records)
        - `ML_Dataset_Organic_v2_5926.csv` — Organic cultivation (5,926 records)
        - `Trial Run 2 ML.csv` — Pilot validation data

        **Scripts Available:**
        - `Model-Training.py` — Full training pipeline
        - `evaluate.py` — Model performance evaluation
        - `predict.py` — Batch prediction utilities
        - `visualise.py` — Visualization and analysis tools
        """)

st.markdown("---")


st.markdown("---")
st.caption("© 2026 CLIMATEKRISHI AI | Data from ICAR-IIRR & KVK Medak")

with st.container():
    logo_cols = st.columns([1, 1, 1, 1], gap="small")
    logo_files = [
         "media/footer_logos/iith_logo.jpeg",
        "media/footer_logos/278969668_371139835029083_611259122675019803_n.jpg",
        "media/footer_logos/about-removebg-preview.png",
        "media/footer_logos/olcalogo.png",
    ]
    for col, logo in zip(logo_cols, logo_files):
        with col:
            st.image(logo, width=2000)
