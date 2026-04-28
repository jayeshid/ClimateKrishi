import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import plotly.graph_objects as go

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLIMATEKRISHI AI : An AI-Powered Platform for Climate Smart Decision Making in Rice Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { width: 0 !important; min-width: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --ck-surface: rgba(255, 255, 255, 0.78);
        --ck-surface-strong: rgba(255, 255, 255, 0.95);
        --ck-border: rgba(15, 23, 42, 0.08);
        --ck-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        --ck-shadow-soft: 0 8px 22px rgba(15, 23, 42, 0.06);
        --ck-accent-1: #16a34a;
        --ck-accent-2: #0f766e;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        letter-spacing: -0.01em;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(16,185,129,0.10), transparent 60%),
            radial-gradient(900px 500px at 95% 0%, rgba(14,165,233,0.08), transparent 55%),
            linear-gradient(180deg, #f7fdf8 0%, #eafaf1 40%, #ffffff 100%);
        color: #1f2937;
    }

    body {
        background: transparent;
        color: #1f2937;
    }

    .css-1d391kg {background-color: rgba(255,255,255,0.75);} /* main container fallback */

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.9);
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid var(--ck-border);
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

    .stTabs [role="tabpanel"] {
        background: linear-gradient(180deg, var(--ck-surface-strong) 0%, var(--ck-surface) 100%);
        border: 1px solid var(--ck-border);
        border-radius: 18px;
        padding: 1rem 1.1rem 1.25rem;
        box-shadow: var(--ck-shadow);
    }

    .stTabs [role="tabpanel"] h3 {
        color: #0f172a;
        letter-spacing: 0.1px;
    }

    .stMainBlockContainer.block-container {
        padding-left: clamp(1rem, 3vw, 5rem) !important;
        padding-right: clamp(1rem, 3vw, 5rem) !important;
        padding-top: 0.75rem !important;
    }
    /* hide the empty Streamlit toolbar gap above content */
    [data-testid="stHeader"] {
        height: 0 !important;
        background: transparent !important;
    }
    [data-testid="stToolbar"] {
        right: 0.5rem;
        top: 0.25rem;
    }
    [data-testid="stDecoration"] { display: none; }

    [data-testid="stHorizontalBlock"] {
        row-gap: clamp(0.65rem, 1.15vw, 1rem) !important;
        column-gap: clamp(0.65rem, 1.15vw, 1rem) !important;
    }

    .ck-feature-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fffb 100%);
        padding: 1.2rem;
        border-radius: 20px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
        min-height: clamp(170px, 20vw, 210px);
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.55rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }

    .ck-feature-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(90deg, #16a34a, #0f766e, #0ea5e9);
        opacity: 0.85;
    }

    .ck-feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(16, 185, 129, 0.35);
        box-shadow: 0 22px 44px rgba(15, 23, 42, 0.12);
    }

    .ck-feature-card h4,
    .ck-feature-card p {
        margin: 0;
    }

    .ck-scenario-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
        min-height: 132px;
    }

    .ck-scenario-card + .ck-scenario-card {
        margin-top: 0.65rem;
    }

    .ck-scenario-head {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.5rem;
    }

    .ck-scenario-accent {
        width: 10px;
        height: 44px;
        border-radius: 999px;
        flex: 0 0 10px;
    }

    .ck-scenario-card h4,
    .ck-scenario-card p {
        margin: 0;
    }

    @media (max-width: 900px) {
        .stTabs [data-baseweb="tab-list"] {
            padding: 0.4rem;
        }

        .stTabs [data-baseweb="tab-list"] button {
            padding: 0.55rem 0.9rem;
            font-size: 0.92rem;
        }

        .stTabs [role="tabpanel"] {
            padding: 0.9rem;
        }
    }

    @media (max-width: 640px) {
        .stMainBlockContainer.block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.9rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: none;
        }

        .stTabs [data-baseweb="tab-list"] button {
            flex: 0 0 auto;
        }

        .ck-feature-card {
            min-height: auto;
        }
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

    .stRadio [role="radiogroup"] {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 14px;
        padding: 0.35rem 0.45rem;
        box-shadow: var(--ck-shadow-soft);
    }

    .stRadio [role="radiogroup"] > label {
        border-radius: 12px;
        padding: 0.3rem 0.55rem;
        transition: background 0.18s ease, box-shadow 0.18s ease;
    }

    .stRadio [role="radiogroup"] > label:has(input:checked) {
        background: rgba(16, 185, 129, 0.14);
        box-shadow: inset 0 0 0 1px rgba(5, 150, 105, 0.24);
    }

    [data-testid="stNumberInputContainer"] [data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
    }

    .stAlert [data-testid="stAlertContainer"] {
        border-radius: 14px;
        box-shadow: var(--ck-shadow-soft);
        border: 1px solid rgba(15, 23, 42, 0.08);
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(148,163,184,0.12), rgba(148,163,184,0.45), rgba(148,163,184,0.12));
    }

    .css-1l02zno {
        background: rgba(255,255,255,0.9);
        border-radius: 18px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }

    .stMetric {
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(240,253,244,0.96) 100%);
        padding: 1rem 1.2rem;
        border: 1px solid rgba(16, 185, 129, 0.28);
        outline: 1px solid rgba(5, 150, 105, 0.14);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10), inset 0 1px 0 rgba(255,255,255,0.75);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .stMetric:hover {
        transform: translateY(-1px);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.13), 0 0 0 2px rgba(16, 185, 129, 0.10);
    }

    .ck-winner-wrap {
        height: 100%;
        min-height: 94px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .ck-winner-badge {
        width: 100%;
        text-align: center;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.55rem 0.7rem;
        border: 1px solid rgba(15, 23, 42, 0.12);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        background: rgba(255, 255, 255, 0.9);
        color: #0f172a;
    }

    .ck-winner-badge.win-a {
        background: linear-gradient(180deg, rgba(219, 252, 231, 0.96) 0%, rgba(187, 247, 208, 0.9) 100%);
        border-color: rgba(22, 163, 74, 0.35);
        color: #166534;
    }

    .ck-winner-badge.win-b {
        background: linear-gradient(180deg, rgba(254, 240, 138, 0.94) 0%, rgba(253, 224, 71, 0.9) 100%);
        border-color: rgba(202, 138, 4, 0.35);
        color: #854d0e;
    }

    .ck-winner-badge.win-tie {
        background: linear-gradient(180deg, rgba(226, 232, 240, 0.95) 0%, rgba(203, 213, 225, 0.9) 100%);
        border-color: rgba(100, 116, 139, 0.35);
        color: #334155;
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
        return (pred[0], pred[1], pred[2], pred[3] if len(pred) >= 4 else 0)
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
        return (pred[0], pred[1], pred[2], pred[3] if len(pred) >= 4 else 0)
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


def build_winner_badge(value_a, value_b):
    """Return styled winner badge HTML for comparison columns."""
    if value_a < value_b:
        label = "Lower: A"
        klass = "win-a"
    elif value_b < value_a:
        label = "Lower: B"
        klass = "win-b"
    else:
        label = "Tie"
        klass = "win-tie"

    return f"<div class='ck-winner-wrap'><div class='ck-winner-badge {klass}'>{label}</div></div>"


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


def build_live_blend_index_chart(conv_out, blend_out, org_out):
    """Build slider-reactive chart with values indexed to Conventional=100."""
    categories = [
        "Global Warming",
        "Freshwater Eutrophication",
        "Terrestrial Acidification",
        "Terrestrial Ecotoxicity",
    ]

    rows = []
    scenarios = [
        ("Conventional", conv_out),
        ("Blend", blend_out),
        ("Organic", org_out),
    ]
    for idx, category in enumerate(categories):
        base = conv_out[idx]
        for scenario, values in scenarios:
            index_value = (values[idx] / base * 100) if base != 0 else 0
            rows.append({
                "Impact Category": category,
                "Scenario": scenario,
                "Index (Conv=100)": index_value,
            })

    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        y=alt.Y(
            "Impact Category:N",
            sort=["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
            title=None,
        ),
        x=alt.X("Index (Conv=100):Q", title="Indexed impact (Conventional = 100)"),
        color=alt.Color(
            "Scenario:N",
            legend=None,
            scale=alt.Scale(
                domain=["Conventional", "Blend", "Organic"],
                range=["#1f77b4", "#f97316", "#16a34a"],
            ),
        ),
        tooltip=[
            "Impact Category",
            "Scenario",
            alt.Tooltip("Index (Conv=100):Q", format=".1f"),
        ],
    ).properties(height=220, width=250)

    faceted = chart.facet(
        column=alt.Column(
            "Scenario:N",
            sort=["Conventional", "Blend", "Organic"],
            header=alt.Header(labelFontSize=12, labelColor="#0f172a", title=None),
        )
    ).resolve_scale(x="independent")

    return style_chart(faceted).properties(title="Live Impact Index (updates with slider)")


def build_blend_frontier_chart(conv_out, org_out, conv_cost, org_cost, selected_alpha):
    """Show cost-climate frontier across blend percentages."""
    rows = []
    for alpha in np.linspace(0, 1, 11):
        rows.append({
            "Organic %": alpha * 100,
            "Cost (INR/ha)": (1 - alpha) * conv_cost + alpha * org_cost,
            "GWP (kg CO2-eq)": (1 - alpha) * conv_out[0] + alpha * org_out[0],
            "Selected": "Selected" if np.isclose(alpha, selected_alpha) else "Other",
        })

    frontier_df = pd.DataFrame(rows)
    line = alt.Chart(frontier_df).mark_line(strokeWidth=2.5, color="#0f766e").encode(
        x=alt.X("Cost (INR/ha):Q", title="Input Cost (INR/ha)"),
        y=alt.Y("GWP (kg CO2-eq):Q", title="Global Warming Potential (kg CO2-eq)"),
    )

    points = alt.Chart(frontier_df).mark_circle(size=110).encode(
        x="Cost (INR/ha):Q",
        y="GWP (kg CO2-eq):Q",
        color=alt.Color("Organic %:Q", title="Organic blend (%)", scale=alt.Scale(scheme="teals")),
        tooltip=[
            alt.Tooltip("Organic %:Q", format=".0f"),
            alt.Tooltip("Cost (INR/ha):Q", format=",.0f"),
            alt.Tooltip("GWP (kg CO2-eq):Q", format=",.2f"),
        ],
    )

    selected = alt.Chart(frontier_df[frontier_df["Selected"] == "Selected"]).mark_point(
        shape="diamond", size=260, filled=True, color="#f97316"
    ).encode(
        x="Cost (INR/ha):Q",
        y="GWP (kg CO2-eq):Q",
        tooltip=[
            alt.Tooltip("Organic %:Q", format=".0f"),
            alt.Tooltip("Cost (INR/ha):Q", format=",.0f"),
            alt.Tooltip("GWP (kg CO2-eq):Q", format=",.2f"),
        ],
    )

    return style_chart((line + points + selected).properties(height=350, width=720, title="Cost-Climate Trade-off Frontier"))


def build_impact_delta_chart(conv_out, blend_out):
    """Show percentage change of each impact category vs conventional baseline."""
    categories = [
        "Global Warming",
        "Freshwater Eutrophication",
        "Terrestrial Acidification",
        "Terrestrial Ecotoxicity",
    ]
    rows = []
    for idx, cat in enumerate(categories):
        change_pct = ((blend_out[idx] - conv_out[idx]) / conv_out[idx] * 100) if conv_out[idx] != 0 else 0.0
        rows.append({
            "Impact Category": cat,
            "Change vs Conventional (%)": change_pct,
            "Direction": "Lower than Conv (better)" if change_pct < 0 else "Higher than Conv (worse)",
        })

    delta_df = pd.DataFrame(rows)
    chart = alt.Chart(delta_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        y=alt.Y(
            "Impact Category:N",
            sort=["Global Warming", "Freshwater Eutrophication", "Terrestrial Acidification", "Terrestrial Ecotoxicity"],
            title=None,
        ),
        x=alt.X("Change vs Conventional (%):Q", title="Change vs Conventional (%)"),
        color=alt.Color(
            "Direction:N",
            title="Direction",
            scale=alt.Scale(
                domain=["Lower than Conv (better)", "Higher than Conv (worse)"],
                range=["#16a34a", "#ef4444"],
            ),
        ),
        tooltip=[
            "Impact Category",
            alt.Tooltip("Change vs Conventional (%):Q", format="+.1f"),
            "Direction",
        ],
    ).properties(height=280, width=720, title="Impact Change vs Conventional Baseline")

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


def build_ccts_source_chart(result):
    """Show how much each amendment contributes to SOC credits after buffer."""
    source_df = pd.DataFrame({
        "Source": ["Farm Yard Manure", "Compost"],
        "Credits (t CO2-eq/ha)": [result["fym_credits_tco2"], result["compost_credits_tco2"]],
    })

    chart = alt.Chart(source_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Source:N", sort=["Farm Yard Manure", "Compost"], title=None),
        y=alt.Y("Credits (t CO2-eq/ha):Q", title="Credits (t CO2-eq/ha)"),
        color=alt.Color("Source:N", legend=None, scale=alt.Scale(range=["#16a34a", "#0ea5e9"])),
        tooltip=["Source", alt.Tooltip("Credits (t CO2-eq/ha):Q", format=".3f")],
    ).properties(title="Credit Contribution by Input", height=300)
    return style_chart(chart)


def build_ccts_buffer_chart(result):
    """Compare SOC before and after permanence buffer and show withheld amount."""
    buffer_df = pd.DataFrame({
        "Stage": ["Before Buffer", "After Buffer", "Buffer Withheld"],
        "t CO2-eq/ha": [
            result["soc_before_buffer_tco2"],
            result["credits_tco2"],
            result["buffer_withheld_tco2"],
        ],
    })

    chart = alt.Chart(buffer_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Stage:N", sort=["Before Buffer", "After Buffer", "Buffer Withheld"], title=None),
        y=alt.Y("t CO2-eq/ha:Q", title="t CO2-eq/ha"),
        color=alt.Color(
            "Stage:N",
            legend=None,
            scale=alt.Scale(
                domain=["Before Buffer", "After Buffer", "Buffer Withheld"],
                range=["#0ea5e9", "#16a34a", "#f59e0b"],
            ),
        ),
        tooltip=["Stage", alt.Tooltip("t CO2-eq/ha:Q", format=".3f")],
    ).properties(title="Permanence Buffer Effect", height=300)
    return style_chart(chart)


def build_ccts_value_chart(result):
    """Display estimated market value range per hectare."""
    value_df = pd.DataFrame({
        "Scenario": ["Low Price", "High Price"],
        "Value (INR/ha)": [result["value_low_inr"], result["value_high_inr"]],
    })

    chart = alt.Chart(value_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Scenario:N", sort=["Low Price", "High Price"], title=None),
        y=alt.Y("Value (INR/ha):Q", title="Estimated Value (INR/ha)"),
        color=alt.Color("Scenario:N", legend=None, scale=alt.Scale(range=["#94a3b8", "#10b981"])),
        tooltip=["Scenario", alt.Tooltip("Value (INR/ha):Q", format=",.0f")],
    ).properties(title="Estimated Credit Value Range", height=300)
    return style_chart(chart)


def render_gauge_html(label, value_pct, icon="", inverse=True, max_abs=60):
    """Conic-gradient circular gauge for % change.

    inverse=True => negative values are 'good' (green), positive 'bad' (red).
    inverse=False => opposite (e.g. cost where lower is also good but shown neutrally).
    """
    if value_pct == 0:
        color = "#94a3b8"
    elif (value_pct < 0) == inverse:
        color = "#16a34a"
    else:
        color = "#ef4444"
    pct_capped = min(abs(value_pct) / max_abs * 100, 100)
    arrow = "▼" if value_pct < 0 else ("▲" if value_pct > 0 else "■")
    return f"""
    <div style='background: linear-gradient(180deg,#ffffff 0%, #f8fffb 100%);
                border-radius:18px; padding:1rem 0.9rem; border:1px solid rgba(15,23,42,0.08);
                box-shadow:0 10px 24px rgba(15,23,42,0.07); text-align:center; height:100%;'>
        <div style='font-size:0.82rem; color:#475569; font-weight:600; margin-bottom:0.6rem;
                    letter-spacing:0.2px;'>{icon} {label}</div>
        <div style='width:118px; height:118px; margin:0 auto; border-radius:50%;
                    background: conic-gradient({color} 0% {pct_capped:.1f}%, #e5e7eb {pct_capped:.1f}% 100%);
                    display:flex; align-items:center; justify-content:center;
                    box-shadow: 0 6px 18px rgba(15,23,42,0.08);'>
            <div style='width:88px; height:88px; border-radius:50%; background:white;
                        display:flex; align-items:center; justify-content:center; flex-direction:column;
                        box-shadow: inset 0 2px 8px rgba(15,23,42,0.06);'>
                <div style='font-size:1.05rem; font-weight:700; color:{color};'>{arrow} {abs(value_pct):.1f}%</div>
                <div style='font-size:0.62rem; color:#94a3b8; letter-spacing:0.4px;'>vs CONV</div>
            </div>
        </div>
    </div>
    """


def render_value_gauge_html(label, value, max_value, unit="", icon="", value_fmt="{:,.2f}",
                            color_low="#16a34a", color_mid="#f59e0b", color_high="#ef4444"):
    """Circular conic gauge showing a value as % of a max benchmark.

    Color graduates from low (green) → mid (amber) → high (red) at 33% / 66% thresholds.
    """
    pct = (value / max_value * 100) if max_value and max_value > 0 else 0
    pct_capped = max(0, min(pct, 100))
    if pct_capped < 50:
        color = color_low
    elif pct_capped < 80:
        color = color_mid
    else:
        color = color_high
    return f"""
    <div style='background: linear-gradient(180deg,#ffffff 0%, #f8fffb 100%);
                border-radius:18px; padding:1rem 0.9rem; border:1px solid rgba(15,23,42,0.08);
                box-shadow:0 10px 24px rgba(15,23,42,0.07); text-align:center; height:100%;'>
        <div style='font-size:0.82rem; color:#475569; font-weight:600; margin-bottom:0.6rem;
                    letter-spacing:0.2px;'>{icon} {label}</div>
        <div style='width:118px; height:118px; margin:0 auto; border-radius:50%;
                    background: conic-gradient({color} 0% {pct_capped:.1f}%, #e5e7eb {pct_capped:.1f}% 100%);
                    display:flex; align-items:center; justify-content:center;
                    box-shadow: 0 6px 18px rgba(15,23,42,0.08);'>
            <div style='width:88px; height:88px; border-radius:50%; background:white;
                        display:flex; align-items:center; justify-content:center; flex-direction:column;
                        box-shadow: inset 0 2px 8px rgba(15,23,42,0.06);'>
                <div style='font-size:0.95rem; font-weight:700; color:#0f172a;'>{value_fmt.format(value)}</div>
                <div style='font-size:0.6rem; color:#94a3b8; letter-spacing:0.3px;'>{unit}</div>
            </div>
        </div>
        <div style='margin-top:0.55rem; font-size:0.7rem; color:{color}; font-weight:700;'>
            {pct:.0f}% of benchmark
        </div>
    </div>
    """


def build_cost_comparison_chart(conv_cost, blend_cost, org_cost, alpha):
    """Horizontal bar chart comparing the three cost scenarios."""
    df = pd.DataFrame({
        "Scenario": ["Conventional", f"Blend ({int(alpha*100)}% Org)", "Organic"],
        "Cost (₹/ha)": [conv_cost, blend_cost, org_cost],
    })
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=8, height=28).encode(
        y=alt.Y("Scenario:N", sort=["Conventional", f"Blend ({int(alpha*100)}% Org)", "Organic"], title=None),
        x=alt.X("Cost (₹/ha):Q", title="Input Cost (₹/ha)"),
        color=alt.Color("Scenario:N", legend=None, scale=alt.Scale(
            domain=["Conventional", f"Blend ({int(alpha*100)}% Org)", "Organic"],
            range=["#1d4ed8", "#f97316", "#16a34a"])),
        tooltip=["Scenario", alt.Tooltip("Cost (₹/ha):Q", format=",.0f")],
    )
    labels = alt.Chart(df).mark_text(align="left", dx=6, color="#0f172a", fontWeight="bold").encode(
        y=alt.Y("Scenario:N", sort=["Conventional", f"Blend ({int(alpha*100)}% Org)", "Organic"]),
        x="Cost (₹/ha):Q",
        text=alt.Text("Cost (₹/ha):Q", format=",.0f"),
    )
    return style_chart((bars + labels).properties(height=180, title="Input Cost by Scenario"))


# ── Advanced visualization helpers ────────────────────────────────────────────
IMPACT_NAMES = ["Global Warming", "Eutrophication", "Acidification", "Ecotoxicity"]


def build_radar_chart(scenarios, title="Impact Profile (normalized)"):
    """Plotly radar chart. scenarios: list of (label, [v1,v2,v3,v4], color)."""
    # Normalize each axis to 0–100 by dividing by max across scenarios for that axis
    arrs = np.array([s[1] for s in scenarios], dtype=float)
    maxes = arrs.max(axis=0)
    maxes = np.where(maxes == 0, 1, maxes)
    fig = go.Figure()
    for label, vals, color in scenarios:
        norm = (np.array(vals) / maxes * 100).tolist()
        fig.add_trace(go.Scatterpolar(
            r=norm + [norm[0]],
            theta=IMPACT_NAMES + [IMPACT_NAMES[0]],
            fill="toself",
            name=label,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.45,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e8f0")),
        showlegend=True,
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15, color="#0f172a")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def build_sankey_chart(emissions, synthetic_n, synthetic_p, amendment_1, amendment_2):
    """Sankey: Inputs → Emission types → Final impact category."""
    inputs = ["Synthetic N", "Synthetic P"]
    if amendment_1 != "None":
        inputs.append(f"Amend: {amendment_1}")
    if amendment_2 != "None":
        inputs.append(f"Amend: {amendment_2}")
    em_labels = ["CH₄", "N₂O", "NO₃", "NH₃", "PO₄"]
    impact_labels = ["Global Warming", "Eutrophication", "Acidification"]
    nodes = inputs + em_labels + impact_labels
    idx = {n: i for i, n in enumerate(nodes)}

    sources, targets, values, link_colors = [], [], [], []

    # N drives CH₄, N₂O, NO₃, NH₃; P drives PO₄, partial NH₃
    n_total = max(synthetic_n, 0.001)
    p_total = max(synthetic_p, 0.001)
    contrib = {
        ("Synthetic N", "CH₄"): emissions["CH4"],
        ("Synthetic N", "N₂O"): emissions["N2O"],
        ("Synthetic N", "NO₃"): emissions["NO3"],
        ("Synthetic N", "NH₃"): emissions["NH3"] * 0.95,
        ("Synthetic P", "NH₃"): emissions["NH3"] * 0.05,
        ("Synthetic P", "PO₄"): emissions["PO4"],
    }
    for (src, dst), val in contrib.items():
        sources.append(idx[src]); targets.append(idx[dst]); values.append(max(val, 1e-6))
        link_colors.append("rgba(59,130,246,0.35)" if "N" in src else "rgba(168,85,247,0.35)")

    # amendments distribute across N₂O / CH₄ / NH₃ proportionally
    for am in [amendment_1, amendment_2]:
        if am != "None":
            src = f"Amend: {am}"
            for em, frac in [("CH₄", 0.10), ("N₂O", 0.05), ("NH₃", 0.08)]:
                sources.append(idx[src]); targets.append(idx[em])
                values.append(max(emissions[{"CH₄":"CH4","N₂O":"N2O","NH₃":"NH3"}[em]] * frac, 1e-6))
                link_colors.append("rgba(34,197,94,0.35)")

    # Emissions → final impact categories (climate impact mapping)
    em_to_impact = {
        "CH₄": ("Global Warming", 28.0),    # GWP factor
        "N₂O": ("Global Warming", 273.0),
        "NO₃": ("Eutrophication", 0.42),
        "PO₄": ("Eutrophication", 1.0),
        "NH₃": ("Acidification", 1.88),
    }
    for em, (impact, factor) in em_to_impact.items():
        sources.append(idx[em]); targets.append(idx[impact])
        values.append(max(emissions[{"CH₄":"CH4","N₂O":"N2O","NO₃":"NO3","NH₃":"NH3","PO₄":"PO4"}[em]] * factor, 1e-6))
        link_colors.append("rgba(239,68,68,0.4)")

    node_colors = (["#3b82f6"] * 2 +
                   ["#a855f7"] * (len(inputs) - 2) +
                   ["#f97316"] * len(em_labels) +
                   ["#16a34a"] * len(impact_labels))

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=18,
            line=dict(color="rgba(15,23,42,0.2)", width=0.5),
            label=nodes, color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        title=dict(text="Inputs → Emissions → Impact Categories", x=0.02,
                   font=dict(size=15, color="#0f172a")),
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        height=460, margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_speedometer(value_pct, label, max_abs=50, inverse=True):
    """Plotly speedometer indicator. Negative = good when inverse=True."""
    if value_pct == 0:
        bar_color = "#94a3b8"
    elif (value_pct < 0) == inverse:
        bar_color = "#16a34a"
    else:
        bar_color = "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value_pct,
        number=dict(suffix="%", font=dict(size=22, color="#0f172a")),
        title=dict(text=label, font=dict(size=13, color="#475569")),
        gauge=dict(
            axis=dict(range=[-max_abs, max_abs], tickwidth=1, tickcolor="#94a3b8",
                      tickfont=dict(size=10)),
            bar=dict(color=bar_color, thickness=0.30),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=1,
            bordercolor="#e2e8f0",
            steps=[
                dict(range=[-max_abs, -max_abs/3], color="rgba(34,197,94,0.18)"),
                dict(range=[-max_abs/3, max_abs/3], color="rgba(148,163,184,0.18)"),
                dict(range=[max_abs/3, max_abs], color="rgba(239,68,68,0.18)"),
            ],
            threshold=dict(line=dict(color="#0f172a", width=3), thickness=0.85, value=value_pct),
        ),
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif"))
    return fig


def build_impact_heatmap():
    """Heatmap of input × impact category from IMPACT_DATA."""
    rows = []
    for i, inp in enumerate(IMPACT_DATA["Input"]):
        rows.append({"Input": inp, "Impact": "Global Warming",
                     "Value": IMPACT_DATA["Global Warming (kg CO₂-eq)"][i]})
        rows.append({"Input": inp, "Impact": "Acidification",
                     "Value": IMPACT_DATA["Terrestrial Acidification (kg SO₂-eq)"][i]})
        rows.append({"Input": inp, "Impact": "Eutrophication",
                     "Value": IMPACT_DATA["Freshwater Eutrophication (kg P-eq)"][i]})
        rows.append({"Input": inp, "Impact": "Ecotoxicity",
                     "Value": IMPACT_DATA["Terrestrial Ecotoxicity (CTUe)"][i]})
    df = pd.DataFrame(rows)
    # log-scale value for color (huge spread Zn ecotox vs others)
    df["LogValue"] = np.log10(df["Value"].clip(lower=1e-6))
    chart = alt.Chart(df).mark_rect(stroke="white", strokeWidth=2).encode(
        x=alt.X("Impact:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=11)),
        y=alt.Y("Input:N", title=None, axis=alt.Axis(labelFontSize=11)),
        color=alt.Color("LogValue:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                        legend=alt.Legend(title="log₁₀(value)", orient="right")),
        tooltip=["Input", "Impact", alt.Tooltip("Value:Q", format=",.4f")],
    ).properties(height=240, title="Per-kg Environmental Footprint (log scale)")
    text = alt.Chart(df).mark_text(fontSize=10, fontWeight="bold").encode(
        x="Impact:N", y="Input:N",
        text=alt.Text("Value:Q", format=",.3g"),
        color=alt.condition("datum.LogValue > 1", alt.value("white"), alt.value("#0f172a")),
    )
    return style_chart(chart + text)


def build_bullet_chart(value, low, high, label, unit=""):
    """Linear bullet chart: zones (low/ok/high), value as marker, target band."""
    span = max(high - low, 1)
    band_min = max(low - span * 0.5, 0)
    band_max = high + span * 0.5
    zones = pd.DataFrame({
        "zone": ["Below", "Recommended", "Above"],
        "start": [band_min, low, high],
        "end":   [low, high, band_max],
    })
    zone_color = alt.Scale(domain=["Below", "Recommended", "Above"],
                           range=["#fde68a", "#86efac", "#fca5a5"])
    bg = alt.Chart(zones).mark_bar(height=18).encode(
        x=alt.X("start:Q", title=None, scale=alt.Scale(domain=[band_min, band_max])),
        x2="end:Q",
        color=alt.Color("zone:N", scale=zone_color, legend=None),
        tooltip=["zone", alt.Tooltip("start:Q", format=",.1f"),
                 alt.Tooltip("end:Q", format=",.1f")],
    )
    marker_df = pd.DataFrame({"v": [value]})
    marker = alt.Chart(marker_df).mark_tick(thickness=4, size=26, color="#0f172a").encode(x="v:Q")
    text = alt.Chart(marker_df).mark_text(dy=-16, fontWeight="bold", color="#0f172a", fontSize=11).encode(
        x="v:Q", text=alt.Text("v:Q", format=",.1f")
    )
    return style_chart((bg + marker + text).properties(height=70, width=300, title=f"{label} {unit}"))


def build_emission_treemap(emissions):
    """Plotly treemap of emissions broken down by category."""
    leaves = [
        ("CH₄", emissions["CH4"], "Climate", "#1f77b4"),
        ("N₂O", emissions["N2O"], "Climate", "#ff7f0e"),
        ("NO₃", emissions["NO3"], "Eutrophication", "#2ca02c"),
        ("PO₄", emissions["PO4"], "Eutrophication", "#9467bd"),
        ("NH₃", emissions["NH3"], "Acidification", "#d62728"),
    ]
    # Build hierarchy with explicit parent rows so Plotly treemap works
    labels = ["Total"]
    parents = [""]
    values = [0.0]
    colors = ["#0f172a"]
    groups = {}
    for em, val, grp, color in leaves:
        groups.setdefault(grp, 0.0)
        groups[grp] += val
    for grp, gval in groups.items():
        labels.append(grp); parents.append("Total")
        values.append(0.0); colors.append("#94a3b8")
    for em, val, grp, color in leaves:
        labels.append(f"{em} ({val:.3f})")
        parents.append(grp)
        values.append(max(val, 1e-6))
        colors.append(color)

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        branchvalues="remainder",
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textfont=dict(family="Inter, sans-serif", size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Parent: %{parent}<br>Value: %{value:.4f}<extra></extra>",
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text="Emission Treemap", x=0.02,
                                 font=dict(size=15, color="#0f172a")),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif"))
    return fig


def build_blend_trend_with_marker(conv_out, org_out, alpha):
    """Blend trend line with a moving marker at the selected slider position."""
    rows = []
    for a in np.linspace(0, 1, 21):
        vals = (1 - a) * np.array(conv_out) + a * np.array(org_out)
        for idx, label in enumerate(IMPACT_NAMES):
            rows.append({"Organic %": a * 100, "Impact": label, "Value": vals[idx]})
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(point=False, strokeWidth=2.5, interpolate="monotone").encode(
        x=alt.X("Organic %:Q", title="Organic blend (%)"),
        y=alt.Y("Value:Q", title="Predicted impact"),
        color=alt.Color("Impact:N", title="Impact"),
        tooltip=["Impact", alt.Tooltip("Value:Q", format=",.3f"),
                 alt.Tooltip("Organic %:Q", format=".0f")],
    )
    marker_pct = alpha * 100
    marker_df = pd.DataFrame([
        {"Organic %": marker_pct, "Impact": label,
         "Value": (1 - alpha) * conv_out[i] + alpha * org_out[i]}
        for i, label in enumerate(IMPACT_NAMES)
    ])
    marker = alt.Chart(marker_df).mark_point(
        size=180, filled=True, shape="circle", stroke="#0f172a", strokeWidth=2
    ).encode(
        x="Organic %:Q", y="Value:Q",
        color=alt.Color("Impact:N", legend=None),
        tooltip=["Impact", alt.Tooltip("Value:Q", format=",.3f")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [marker_pct]})).mark_rule(
        color="#f97316", strokeDash=[4, 4], strokeWidth=1.5
    ).encode(x="x:Q")
    return style_chart((line + rule + marker).properties(height=340, width=720,
                       title=f"Impact Trend Across Blend (marker @ {int(marker_pct)}% organic)"))


def build_pareto_with_isolines(conv_out, org_out, conv_cost, org_cost, selected_alpha):
    """Pareto frontier with iso-cost-per-CO2 dashed reference lines."""
    rows = []
    for alpha in np.linspace(0, 1, 21):
        rows.append({
            "Organic %": alpha * 100,
            "Cost (INR/ha)": (1 - alpha) * conv_cost + alpha * org_cost,
            "GWP (kg CO2-eq)": (1 - alpha) * conv_out[0] + alpha * org_out[0],
            "Selected": np.isclose(alpha, selected_alpha),
        })
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(strokeWidth=3, color="#0f766e",
                                    interpolate="monotone").encode(
        x=alt.X("Cost (INR/ha):Q"),
        y=alt.Y("GWP (kg CO2-eq):Q"),
    )
    pts = alt.Chart(df).mark_circle(size=90).encode(
        x="Cost (INR/ha):Q", y="GWP (kg CO2-eq):Q",
        color=alt.Color("Organic %:Q", title="Organic blend (%)",
                        scale=alt.Scale(scheme="teals")),
        tooltip=[alt.Tooltip("Organic %:Q", format=".0f"),
                 alt.Tooltip("Cost (INR/ha):Q", format=",.0f"),
                 alt.Tooltip("GWP (kg CO2-eq):Q", format=",.2f")],
    )
    selected = alt.Chart(df[df["Selected"]]).mark_point(
        shape="diamond", size=320, filled=True, color="#f97316",
        stroke="#0f172a", strokeWidth=2
    ).encode(x="Cost (INR/ha):Q", y="GWP (kg CO2-eq):Q")

    # Iso-cost-per-CO2 reference lines passing through conv & organic endpoints
    cost_min = min(conv_cost, org_cost) * 0.9
    cost_max = max(conv_cost, org_cost) * 1.1
    iso_lines = []
    for ratio_label, slope in [("Cheap-to-cut", 0.2), ("Moderate", 0.5), ("Expensive", 1.0)]:
        intercept = conv_out[0] - slope * conv_cost
        iso_lines.append(pd.DataFrame({
            "Cost (INR/ha)": [cost_min, cost_max],
            "GWP (kg CO2-eq)": [intercept + slope * cost_min, intercept + slope * cost_max],
            "Iso": [ratio_label, ratio_label],
        }))
    iso_df = pd.concat(iso_lines)
    iso = alt.Chart(iso_df).mark_line(strokeDash=[5, 4], strokeWidth=1, opacity=0.5).encode(
        x="Cost (INR/ha):Q", y="GWP (kg CO2-eq):Q",
        color=alt.Color("Iso:N", title="Iso-trade-off",
                        scale=alt.Scale(range=["#94a3b8", "#64748b", "#475569"])),
    )
    return style_chart((iso + line + pts + selected).properties(
        height=360, width=720, title="Cost-Climate Frontier with Iso-trade-off Lines"))


def build_blend_streamgraph(conv_out, org_out):
    """Stacked area showing each impact's share of total across blend %."""
    rows = []
    for a in np.linspace(0, 1, 21):
        vals = (1 - a) * np.array(conv_out) + a * np.array(org_out)
        # Normalize each row so total=100% (composition view)
        total = vals.sum() or 1
        for idx, label in enumerate(IMPACT_NAMES):
            rows.append({"Organic %": a * 100, "Impact": label,
                         "Share": vals[idx] / total * 100})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_area(interpolate="monotone", opacity=0.85).encode(
        x=alt.X("Organic %:Q", title="Organic blend (%)"),
        y=alt.Y("Share:Q", stack="normalize", title="Share of total impact"),
        color=alt.Color("Impact:N",
                        scale=alt.Scale(range=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])),
        tooltip=["Impact", alt.Tooltip("Share:Q", format=".1f"),
                 alt.Tooltip("Organic %:Q", format=".0f")],
    ).properties(height=300, width=720, title="Impact Composition Across Blend Transition")
    return style_chart(chart)


def render_waffle_html(parts, total_label="GWP", squares=100):
    """HTML waffle: 100 colored squares showing % share of each part.

    parts: list of (label, value, color)
    """
    total = sum(p[1] for p in parts) or 1
    pct = []
    cumulative = 0
    for label, val, color in parts:
        share = round(val / total * squares)
        pct.append((label, share, color))
        cumulative += share
    # adjust last to make sum = squares
    if cumulative != squares and pct:
        last = list(pct[-1])
        last[1] += squares - cumulative
        pct[-1] = tuple(last)

    cells = []
    for label, count, color in pct:
        for _ in range(count):
            cells.append(f"<div style='width:18px;height:18px;background:{color};"
                         f"border-radius:4px;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.4);'"
                         f" title='{label}'></div>")

    legend = "".join(
        f"<div style='display:flex;align-items:center;gap:0.4rem;font-size:0.78rem;color:#334155;'>"
        f"<span style='width:12px;height:12px;border-radius:3px;background:{color};'></span>"
        f"<span>{label} ({val/total*100:.0f}%)</span></div>"
        for label, val, color in parts
    )
    return f"""
    <div style='background:white;border-radius:18px;padding:1rem 1.1rem;
                border:1px solid rgba(15,23,42,0.08);box-shadow:0 10px 24px rgba(15,23,42,0.07);'>
        <div style='font-weight:700;color:#0f172a;margin-bottom:0.6rem;'>{total_label} Composition (1 square = 1%)</div>
        <div style='display:grid;grid-template-columns:repeat(20,1fr);gap:3px;margin-bottom:0.85rem;'>
            {''.join(cells)}
        </div>
        <div style='display:flex;flex-wrap:wrap;gap:0.9rem;'>{legend}</div>
    </div>
    """


def build_confidence_band_chart(conv_out, org_out, band_pct=0.10):
    """Blend GWP across alpha with ±band_pct confidence band (illustrative)."""
    rows = []
    for a in np.linspace(0, 1, 21):
        gwp = (1 - a) * conv_out[0] + a * org_out[0]
        rows.append({"Organic %": a * 100, "GWP": gwp,
                     "low": gwp * (1 - band_pct), "high": gwp * (1 + band_pct)})
    df = pd.DataFrame(rows)
    band = alt.Chart(df).mark_area(opacity=0.25, color="#0f766e").encode(
        x=alt.X("Organic %:Q", title="Organic blend (%)"),
        y=alt.Y("low:Q", title="GWP (kg CO₂-eq)"),
        y2="high:Q",
    )
    line = alt.Chart(df).mark_line(strokeWidth=2.5, color="#0f766e").encode(
        x="Organic %:Q", y="GWP:Q",
        tooltip=[alt.Tooltip("Organic %:Q", format=".0f"),
                 alt.Tooltip("GWP:Q", format=",.2f"),
                 alt.Tooltip("low:Q", format=",.2f"),
                 alt.Tooltip("high:Q", format=",.2f")],
    )
    return style_chart((band + line).properties(height=300, width=720,
                       title=f"GWP vs Blend with ±{int(band_pct*100)}% Confidence Band"))


def render_diff_arrow_html(value_a, value_b, unit="", label="", fmt="{:.2f}"):
    """Show A vs B with animated arrow + % diff."""
    if value_a == 0:
        pct = 0
    else:
        pct = (value_b - value_a) / value_a * 100
    if pct < 0:
        color = "#16a34a"; arrow = "▼"; verdict = "B is greener"
    elif pct > 0:
        color = "#ef4444"; arrow = "▲"; verdict = "B is worse"
    else:
        color = "#64748b"; arrow = "■"; verdict = "Tie"
    return f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
                background:white;border-radius:14px;padding:0.8rem 1rem;
                border:1px solid rgba(15,23,42,0.08);box-shadow:0 8px 18px rgba(15,23,42,0.06);
                gap:0.8rem;'>
        <div style='text-align:center;flex:1;'>
            <div style='font-size:0.7rem;color:#64748b;letter-spacing:0.4px;'>A</div>
            <div style='font-weight:700;color:#0f172a;'>{fmt.format(value_a)} {unit}</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.4rem;color:{color};font-weight:800;line-height:1;'>{arrow}</div>
            <div style='font-size:0.85rem;color:{color};font-weight:700;'>{abs(pct):+.1f}%</div>
            <div style='font-size:0.65rem;color:#94a3b8;'>{verdict}</div>
        </div>
        <div style='text-align:center;flex:1;'>
            <div style='font-size:0.7rem;color:#64748b;letter-spacing:0.4px;'>B</div>
            <div style='font-weight:700;color:#0f172a;'>{fmt.format(value_b)} {unit}</div>
        </div>
        <div style='font-size:0.78rem;color:#475569;font-weight:600;min-width:90px;'>{label}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# FARMER-FRIENDLY INFERENCE ENGINE
# Multilingual (English / Hindi / Telugu) · Band-aware · Context-aware
# ══════════════════════════════════════════════════════════════════════════════
REFERENCES = {
    "co2_per_km_car":     0.244,
    "co2_per_tree_year":  60.0,
    "co2_per_l_diesel":   2.69,
    "co2_per_lpg_cyl":    43.0,
    "rice_gwp_low":       3000,
    "rice_gwp_high":      7000,
    "ch4_flooded_low":    200,
    "ch4_flooded_high":   500,
    "no3_who_limit":      50.0,
    "no3_bis_limit":      45.0,
    "po4_algae_thresh":   0.1,
    "urea_price_inr_kg":  6.5,
    "farm_wage_inr_day":  350,
    "ccts_price_low":     600,
    "ccts_price_high":    900,
}

SOURCES = {
    "epa":           ("EPA GHG Equivalencies Calculator (2024)",
                      "https://www.epa.gov/energy/greenhouse-gas-equivalencies-calculator-calculations-and-references"),
    "ipcc_ar6":      ("IPCC AR6 Working Group I (2021)",
                      "https://www.ipcc.ch/report/ar6/wg1/"),
    "ipcc_2019":     ("IPCC 2019 Refinement (N₂O EF)",
                      "https://www.ipcc-nggip.iges.or.jp/public/2019rf/"),
    "ipcc_rice":     ("IPCC – Methane Emissions from Rice Cultivation",
                      "https://www.ipcc-nggip.iges.or.jp/public/gl/guidelin/ch4ref5.pdf"),
    "frontiers22":   ("Frontiers in Sustainable Food Systems (2022)",
                      "https://www.frontiersin.org/journals/sustainable-food-systems/articles/10.3389/fsufs.2022.868479/full"),
    "who_no3":       ("WHO Nitrate/Nitrite Drinking-water Fact Sheet (2022)",
                      "https://cdn.who.int/media/docs/default-source/wash-documents/water-safety-and-quality/chemical-fact-sheets-2022/nitrate-and-nitrite-fact-sheet-2022.pdf"),
    "bis":           ("BIS IS 10500:2012 – Indian Drinking Water Standard",
                      "https://www.bis.gov.in/"),
    "gold_awm":      ("Gold Standard – AWD Methane Reduction Methodology",
                      "https://globalgoals.goldstandard.org/standards/437_V1.0_LUF_AGR_Methane-emission-reduction-by-AWM-practice-in-rice-cultivation.pdf"),
    "who_air":       ("WHO Global Air Quality Guidelines (2021)",
                      "https://www.who.int/publications/i/item/9789240034228"),
    "usda":          ("USDA Forest Service – Urban Tree Database",
                      "https://www.fs.usda.gov/treesearch/pubs/52933"),
    "moefcc_ccts":   ("MoEFCC / BEE – Carbon Credit Trading Scheme (2024)",
                      "https://moef.gov.in/"),
    "verra":         ("Verra VM0042 – Improved Agricultural Land Management",
                      "https://verra.org/methodologies/vm0042-methodology-for-improved-agricultural-land-management-v2-0/"),
    "icar_inm":      ("ICAR – Integrated Nutrient Management",
                      "https://icar.org.in/"),
    "salca":         ("SALCA – Swiss Agricultural LCA Methodology",
                      "https://www.agroscope.admin.ch/agroscope/en/home/topics/environment-resources/life-cycle-assessment/salca.html"),
}


def cite(*keys):
    """Render one or more clickable reference pills."""
    pills = []
    for k in keys:
        name, url = SOURCES.get(k, (k, "#"))
        short = k.replace("_", " ").upper().split()[0]
        pills.append(
            f'<a href="{url}" target="_blank" rel="noopener" title="{name}" '
            f'style="display:inline-flex;align-items:center;gap:0.15rem;'
            f'background:#eef2ff;color:#4338ca;border:1px solid #c7d2fe;'
            f'padding:0.05rem 0.5rem;border-radius:999px;font-size:0.7rem;'
            f'font-weight:600;text-decoration:none;margin:0 0.15rem;'
            f'vertical-align:middle;white-space:nowrap;">↗ {short}</a>'
        )
    return "".join(pills)


def _band(value, low, high):
    if value < low:
        return ("low", "🟢", "#16a34a")
    if value > high:
        return ("high", "🔴", "#ef4444")
    return ("mid", "🟡", "#f59e0b")


def _t(en, hi, te, lang, mr=None):
    # Marathi falls back to Hindi when explicit Marathi text is not provided
    return {"en": en, "hi": hi, "te": te, "mr": mr if mr is not None else hi}[lang]


def build_inference_card(domain, value, ctx=None, lang="en"):
    """Return HTML for a band-aware, context-aware, multilingual inference card."""
    R = REFERENCES
    ctx = ctx or {}
    title, paragraph, refs, color, emoji = "", "", "", "#0ea5e9", "💡"

    if domain == "gwp_total":
        km = value / R["co2_per_km_car"]
        trees = value / R["co2_per_tree_year"]
        band, emoji, color = _band(value, R["rice_gwp_low"], R["rice_gwp_high"])
        title = _t("🌍 Global Warming", "🌍 जलवायु प्रभाव (Global Warming)", "🌍 వాతావరణ ప్రభావం (Global Warming)", lang,
                   mr="🌍 हवामान परिणाम (Global Warming)")
        irrig = ctx.get("irrigation", "")
        amendments_used = ctx.get("amendments_used", False)
        if band == "low":
            paragraph = _t(
                f"Excellent — your <b>{value:,.0f} kg CO₂-eq/ha</b> is <b>below</b> the typical 3,000–7,000 kg/ha range for irrigated rice, equivalent to driving only <b>~{km:,.0f} km</b> or what <b>~{trees:.0f} trees</b> absorb in a year. Keep this low-input strategy and consider documenting it for CCTS soil-carbon credits.",
                f"बहुत अच्छा — आपका <b>{value:,.0f} kg CO₂-eq/ha</b> सिंचित धान के सामान्य 3,000–7,000 kg/ha स्तर से <b>कम</b> है, यानी कार से सिर्फ़ <b>~{km:,.0f} किमी</b> चलाने जितना या <b>~{trees:.0f} पेड़</b> एक साल में जितना सोखते हैं। इस कम-इनपुट तरीक़े को बनाए रखें और CCTS soil-carbon credits के लिए दस्तावेज़ बनाएँ।",
                f"అద్భుతం — మీ <b>{value:,.0f} kg CO₂-eq/ha</b> సాధారణ 3,000–7,000 kg/ha కంటే <b>తక్కువ</b>, అంటే కేవలం <b>~{km:,.0f} కి.మీ.</b> కారు నడిపినంత లేదా <b>~{trees:.0f} చెట్లు</b> ఒక ఏడాదిలో పీల్చుకునేంత. ఈ low-input విధానాన్ని కొనసాగించి CCTS soil-carbon credits కోసం రికార్డ్ చేయండి.",
                lang,
                mr=f"उत्तम — तुमचा <b>{value:,.0f} kg CO₂-eq/ha</b> सिंचित भातशेतीच्या नेहमीच्या 3,000–7,000 kg/ha पातळीपेक्षा <b>कमी</b> आहे, म्हणजे कारने फक्त <b>~{km:,.0f} किमी</b> चालवण्याइतके किंवा <b>~{trees:.0f} झाडे</b> वर्षभरात शोषून घेतात तितके. ही कमी-input पद्धत कायम ठेवा आणि CCTS soil-carbon credits साठी नोंद ठेवा.")
        elif band == "mid":
            tip_en = "switching to <b>AWD irrigation</b> cuts CH₄ by 30–70%, and splitting urea into 3 doses cuts N₂O ~20%"
            tip_hi = "<b>AWD सिंचाई</b> से CH₄ 30–70% कम होता है, और यूरिया को 3 बार में देने से N₂O ~20% कम होता है"
            tip_te = "<b>AWD పద్ధతి</b> తో CH₄ 30–70% తగ్గుతుంది, యూరియాను 3 సార్లు వేస్తే N₂O ~20% తగ్గుతుంది"
            tip_mr = "<b>AWD सिंचन</b> केल्याने CH₄ 30–70% कमी होते, आणि युरिया 3 हप्त्यांत दिल्याने N₂O ~20% कमी होते"
            if irrig == "Alternate Wetting and Drying":
                tip_en = "you're already using AWD — next gains come from neem-coated urea and 5 t/ha FYM substitution"
                tip_hi = "आप पहले से AWD कर रहे हैं — अगला लाभ neem-coated urea और 5 t/ha FYM से मिलेगा"
                tip_te = "మీరు ఇప్పటికే AWD వాడుతున్నారు — తదుపరి మెరుగు neem-coated urea మరియు 5 t/ha FYM వల్ల వస్తుంది"
                tip_mr = "तुम्ही आधीच AWD वापरत आहात — पुढचा फायदा neem-coated urea आणि 5 t/ha शेणखत बदलामुळे मिळेल"
            paragraph = _t(
                f"Your <b>{value:,.0f} kg CO₂-eq/ha</b> sits in the <b>middle</b> of the typical 3,000–7,000 kg/ha range — equivalent to driving <b>~{km:,.0f} km</b>. There's clear room to improve: {tip_en}.",
                f"आपका <b>{value:,.0f} kg CO₂-eq/ha</b> सामान्य 3,000–7,000 kg/ha स्तर के <b>बीच</b> में है — कार से <b>~{km:,.0f} किमी</b> चलाने जितना। सुधार की गुंजाइश है: {tip_hi}।",
                f"మీ <b>{value:,.0f} kg CO₂-eq/ha</b> సాధారణ 3,000–7,000 kg/ha రేంజ్ <b>మధ్యలో</b> ఉంది — <b>~{km:,.0f} కి.మీ.</b> కారు నడిపినంత. మెరుగుపరచటానికి అవకాశం: {tip_te}.",
                lang,
                mr=f"तुमचा <b>{value:,.0f} kg CO₂-eq/ha</b> नेहमीच्या 3,000–7,000 kg/ha पातळीच्या <b>मध्यभागी</b> आहे — कारने <b>~{km:,.0f} किमी</b> चालवण्याइतके. सुधारणेला वाव आहे: {tip_mr}.")
        else:
            urgent_en = "adopt <b>AWD irrigation</b> (cuts CH₄ 30–70%), split urea into 3 doses, and substitute 20% synthetic N with FYM/compost"
            urgent_hi = "<b>AWD सिंचाई</b> अपनाएँ (CH₄ 30–70% कम), यूरिया को 3 बार में दें, और 20% सिंथेटिक N को FYM/compost से बदलें"
            urgent_te = "<b>AWD పద్ధతి</b> అవలంబించండి (CH₄ 30–70% తగ్గింపు), యూరియాను 3 సార్లు వేయండి, 20% synthetic N ను FYM/compost తో భర్తీ చేయండి"
            urgent_mr = "<b>AWD सिंचन</b> स्वीकारा (CH₄ 30–70% कमी), युरिया 3 हप्त्यांत द्या, आणि 20% synthetic N शेणखत/कंपोस्टने बदला"
            if not amendments_used:
                urgent_en += "; <b>start with 5 t/ha FYM</b> — biggest win for both emissions and soil carbon credits"
                urgent_hi += "; <b>5 t/ha FYM से शुरू करें</b> — उत्सर्जन और soil carbon credits दोनों के लिए सबसे बड़ा फ़ायदा"
                urgent_te += "; <b>5 t/ha FYM తో మొదలుపెట్టండి</b> — ఉద్గారాలు మరియు soil carbon credits రెండింటికీ అతిపెద్ద లాభం"
                urgent_mr += "; <b>5 t/ha शेणखताने सुरुवात करा</b> — उत्सर्जन आणि soil carbon credits दोन्हीसाठी सर्वात मोठा फायदा"
            paragraph = _t(
                f"⚠️ Your <b>{value:,.0f} kg CO₂-eq/ha</b> is <b>above</b> the typical 3,000–7,000 kg/ha range — equivalent to driving <b>~{km:,.0f} km</b>. Priority actions: {urgent_en}.",
                f"⚠️ आपका <b>{value:,.0f} kg CO₂-eq/ha</b> सामान्य 3,000–7,000 kg/ha स्तर से <b>ऊपर</b> है — कार से <b>~{km:,.0f} किमी</b> चलाने जितना। प्राथमिकता: {urgent_hi}।",
                f"⚠️ మీ <b>{value:,.0f} kg CO₂-eq/ha</b> సాధారణ 3,000–7,000 kg/ha రేంజ్‌ను <b>మించింది</b> — <b>~{km:,.0f} కి.మీ.</b> కారు నడిపినంత. ప్రాధాన్యత: {urgent_te}.",
                lang,
                mr=f"⚠️ तुमचा <b>{value:,.0f} kg CO₂-eq/ha</b> नेहमीच्या 3,000–7,000 kg/ha पातळीपेक्षा <b>जास्त</b> आहे — कारने <b>~{km:,.0f} किमी</b> चालवण्याइतके. प्राथमिकता: {urgent_mr}.")
        refs = cite("epa", "frontiers22", "ipcc_ar6", "gold_awm")

    elif domain == "ch4":
        co2eq = value * 27.9
        band, emoji, color = _band(value, R["ch4_flooded_low"] * 0.4, R["ch4_flooded_high"] * 0.7)
        title = _t("🔥 Methane (CH₄)", "🔥 मीथेन (CH₄)", "🔥 మీథేన్ (CH₄)", lang, mr="🔥 मिथेन (CH₄)")
        irrig = ctx.get("irrigation", "")
        if irrig == "Alternate Wetting and Drying":
            extra_en = "You're already using AWD (cutting CH₄ by ~30–70% vs continuous flooding) — fine-tune drainage timing and avoid incorporating fresh straw under flood for an extra <b>~10–15%</b> reduction."
            extra_hi = "आप पहले से AWD कर रहे हैं (continuous flooding की तुलना में CH₄ ~30–70% कम) — drainage समय सुधारें और ताज़ा पुआल पानी में न मिलाएँ — <b>~10–15%</b> और कमी संभव।"
            extra_te = "మీరు ఇప్పటికే AWD వాడుతున్నారు (continuous flooding తో పోలిస్తే CH₄ ~30–70% తక్కువ) — drainage timing సరిచేసి, ఫ్రెష్ గడ్డిని నీటిలో కలపొద్దు — మరో <b>~10–15%</b> తగ్గింపు."
            extra_mr = "तुम्ही आधीच AWD वापरत आहात (continuous flooding च्या तुलनेत CH₄ ~30–70% कमी) — drainage वेळ सुधारा आणि ताजे पेंढा पाण्यात मिसळवू नका — आणखी <b>~10–15%</b> कपात शक्य."
        elif irrig == "Rainfed":
            extra_en = "Rainfed systems already have low CH₄ (~<b>60–80% lower</b> than continuously flooded paddies); focus on N management instead — split urea + neem-coating cuts N₂O by another ~20%."
            extra_hi = "Rainfed में CH₄ पहले से कम है (~<b>60–80%</b> लगातार सिंचित से कम); ध्यान N प्रबंधन पर दें — split urea + neem-coating से N₂O ~20% कम।"
            extra_te = "Rainfed లో CH₄ ఇప్పటికే తక్కువ (~<b>60–80%</b> continuous flooding కంటే తక్కువ); దృష్టి N management పై పెట్టండి — split urea + neem-coating తో N₂O ~20% తగ్గింపు."
            extra_mr = "Rainfed मध्ये CH₄ आधीच कमी आहे (continuous flooding पेक्षा ~<b>60–80% कमी</b>); N व्यवस्थापनावर लक्ष द्या — split urea + neem-coating ने N₂O ~20% कमी."
        else:
            extra_en = "Practising <b>AWD</b> — draining the field 2–3 times mid-season — is the single biggest CH₄ reducer (<b>cuts 30–70%</b>); adding 5 t/ha FYM instead of fresh straw saves another ~10%."
            extra_hi = "<b>AWD</b> अपनाना — मध्य-सीज़न में 2–3 बार पानी निकालना — CH₄ कम करने का सबसे बड़ा तरीक़ा है (<b>30–70% कमी</b>); ताज़ा पुआल की जगह 5 t/ha FYM डालने से ~10% और बचत।"
            extra_te = "<b>AWD</b> పద్ధతి — mid-season లో 2–3 సార్లు నీరు తీసేయడం — CH₄ తగ్గించడంలో అతిపెద్ద చర్య (<b>30–70% తగ్గింపు</b>); ఫ్రెష్ గడ్డికి బదులు 5 t/ha FYM వేస్తే మరో ~10% ఆదా."
            extra_mr = "<b>AWD</b> अवलंबणे — मध्य-हंगामात 2–3 वेळा पाणी काढणे — CH₄ कमी करण्याचा सर्वात मोठा मार्ग आहे (<b>30–70% कपात</b>); ताज्या पेंढ्याऐवजी 5 t/ha शेणखत दिल्यास आणखी ~10% बचत."
        if band == "low":
            paragraph = _t(
                f"🟢 Your paddy releases <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — <b>well below</b> the IPCC default of 200–500 kg/ha for fully-flooded paddies (you're roughly <b>{(1-value/350)*100:.0f}% lower</b> than the 350 kg/ha midpoint). {extra_en}",
                f"🟢 आपके खेत से <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC default 200–500 kg/ha से <b>बहुत कम</b> (350 kg/ha मिडपॉइंट से लगभग <b>{(1-value/350)*100:.0f}% कम</b>)। {extra_hi}",
                f"🟢 మీ పొలం నుంచి <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC default 200–500 kg/ha కంటే <b>చాలా తక్కువ</b> (350 kg/ha మిడ్‌పాయింట్ కంటే ~<b>{(1-value/350)*100:.0f}% తక్కువ</b>). {extra_te}",
                lang,
                mr=f"🟢 तुमच्या भातशेतीतून <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC default 200–500 kg/ha पेक्षा <b>खूप कमी</b> (350 kg/ha मध्यबिंदूपेक्षा सुमारे <b>{(1-value/350)*100:.0f}% कमी</b>). {extra_mr}")
        elif band == "mid":
            paragraph = _t(
                f"Your paddy releases <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq), within the IPCC default 200–500 kg/ha range for flooded paddies. Adopting AWD here can <b>cut CH₄ by 30–70%</b> (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg/ha saved, worth ₹{value*0.5*27.9*0.0009*1000:,.0f}–₹{value*0.7*27.9*0.0009*1000:,.0f}/ha at CCTS rates). {extra_en}",
                f"आपके खेत से <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) निकल रहा है, जो IPCC default 200–500 kg/ha के बीच है। AWD अपनाने से <b>CH₄ 30–70% कम</b> हो सकता है (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg/ha बचत, CCTS दर पर ₹{value*0.5*27.9*0.0009*1000:,.0f}–₹{value*0.7*27.9*0.0009*1000:,.0f}/ha)। {extra_hi}",
                f"మీ పొలం నుంచి <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) విడుదలవుతోంది, IPCC default 200–500 kg/ha రేంజ్‌లో. AWD అవలంబిస్తే <b>CH₄ 30–70% తగ్గుతుంది</b> (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg/ha ఆదా, CCTS ధర వద్ద ₹{value*0.5*27.9*0.0009*1000:,.0f}–₹{value*0.7*27.9*0.0009*1000:,.0f}/ha). {extra_te}",
                lang,
                mr=f"तुमच्या भातशेतीतून <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) बाहेर पडत आहे, जे IPCC default 200–500 kg/ha च्या मध्यवर्ती आहे. AWD स्वीकारल्यास <b>CH₄ 30–70% कमी</b> होऊ शकते (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg/ha बचत, CCTS दराने ₹{value*0.5*27.9*0.0009*1000:,.0f}–₹{value*0.7*27.9*0.0009*1000:,.0f}/ha). {extra_mr}")
        else:
            paragraph = _t(
                f"⚠️ Your paddy releases <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — <b>{(value/350-1)*100:.0f}% above</b> the IPCC 350 kg/ha midpoint. {extra_en} Expect a <b>30–70% cut on adoption</b> (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg CH₄/ha saved).",
                f"⚠️ आपके खेत से <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC 350 kg/ha मिडपॉइंट से <b>{(value/350-1)*100:.0f}% ज़्यादा</b>। {extra_hi} <b>30–70% तक कमी</b> संभव (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg CH₄/ha बचत)।",
                f"⚠️ మీ పొలం నుంచి <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC 350 kg/ha మిడ్‌పాయింట్ కంటే <b>{(value/350-1)*100:.0f}% ఎక్కువ</b>. {extra_te} అమలుతో <b>30–70% తగ్గింపు</b> (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg CH₄/ha ఆదా).",
                lang,
                mr=f"⚠️ तुमच्या भातशेतीतून <b>{value:,.1f} kg CH₄/ha/season</b> (= {co2eq:,.0f} kg CO₂-eq) — IPCC 350 kg/ha मध्यबिंदूपेक्षा <b>{(value/350-1)*100:.0f}% जास्त</b>. {extra_mr} अमलानंतर <b>30–70% कपात</b> शक्य (≈ {value*0.3:,.0f}–{value*0.7:,.0f} kg CH₄/ha बचत).")
        refs = cite("ipcc_rice", "gold_awm", "ipcc_ar6")

    elif domain == "n2o":
        co2eq = value * 273
        band, emoji, color = _band(value, 1.0, 3.0)
        title = _t("⚡ Nitrous Oxide (N₂O)", "⚡ नाइट्रस ऑक्साइड (N₂O)", "⚡ నైట్రస్ ఆక్సైడ్ (N₂O)", lang,
                   mr="⚡ नायट्रस ऑक्साइड (N₂O)")
        n_high = ctx.get("synthetic_n", 0) > 145
        action_en = "Apply N in <b>3 splits</b> (basal + tillering + panicle initiation) and use neem-coated urea to cut this by ~20%"
        action_hi = "N को <b>3 बार में</b> दें (basal + tillering + panicle initiation) और neem-coated urea से ~20% कमी करें"
        action_te = "N ను <b>3 splits</b> లో (basal + tillering + panicle initiation) వేయండి, neem-coated urea తో ~20% తగ్గించండి"
        action_mr = "N <b>3 हप्त्यांत</b> द्या (basal + tillering + panicle initiation) आणि neem-coated urea वापरून ~20% कमी करा"
        if n_high:
            action_en = "Your N is high (>145 kg/ha) — first cut N to ~135 kg/ha, then split into 3 doses with neem-coated urea"
            action_hi = "आपका N ज़्यादा है (>145 kg/ha) — पहले N घटाकर ~135 kg/ha करें, फिर 3 बार में neem-coated urea से दें"
            action_te = "మీ N ఎక్కువగా ఉంది (>145 kg/ha) — ముందు N ను ~135 kg/ha కు తగ్గించి, 3 splits లో neem-coated urea తో వేయండి"
            action_mr = "तुमचा N जास्त आहे (>145 kg/ha) — प्रथम N कमी करून ~135 kg/ha करा, नंतर 3 हप्त्यांत neem-coated urea द्वारे द्या"
        paragraph = _t(
            f"{emoji} Your N₂O emission of <b>{value:,.3f} kg/ha</b> equals <b>{co2eq:,.1f} kg CO₂-eq</b> (N₂O is 273× stronger than CO₂). IPCC default: ~1% of applied N is lost as N₂O. {action_en}.",
            f"{emoji} आपका N₂O उत्सर्जन <b>{value:,.3f} kg/ha</b> = <b>{co2eq:,.1f} kg CO₂-eq</b> (N₂O CO₂ से 273× ज़्यादा शक्तिशाली)। IPCC default: लगाए गए N का ~1% N₂O के रूप में निकलता है। {action_hi}।",
            f"{emoji} మీ N₂O ఉద్గారం <b>{value:,.3f} kg/ha</b> = <b>{co2eq:,.1f} kg CO₂-eq</b> (N₂O అనేది CO₂ కంటే 273× శక్తివంతం). IPCC default: వేసిన N లో ~1% N₂O గా పోతుంది. {action_te}.",
            lang,
            mr=f"{emoji} तुमचे N₂O उत्सर्जन <b>{value:,.3f} kg/ha</b> = <b>{co2eq:,.1f} kg CO₂-eq</b> (N₂O CO₂ पेक्षा 273× जास्त प्रभावी). IPCC default: दिलेल्या N च्या ~1% N₂O म्हणून वायुमंडळात जातो. {action_mr}.")
        refs = cite("ipcc_ar6", "ipcc_2019")

    elif domain == "no3":
        mg_per_l = value / 1.0
        band, emoji, color = _band(mg_per_l, R["no3_bis_limit"] * 0.5, R["no3_bis_limit"])
        title = _t("💦 Nitrate (NO₃⁻)", "💦 नाइट्रेट (NO₃⁻)", "💦 నైట్రేట్ (NO₃⁻)", lang,
                   mr="💦 नायट्रेट (NO₃⁻)")
        if band == "high":
            warn_en = "⚠️ Above this risks <i>methaemoglobinaemia (\"blue-baby\")</i> in infants drinking groundwater nearby."
            warn_hi = "⚠️ इससे ऊपर पास के groundwater पीने वाले शिशुओं में <i>methaemoglobinaemia (\"blue-baby\")</i> का ख़तरा।"
            warn_te = "⚠️ ఇంతకన్నా ఎక్కువైతే చుట్టుపక్కల groundwater తాగే శిశువులకు <i>methaemoglobinaemia (\"blue-baby\")</i> ప్రమాదం."
            warn_mr = "⚠️ याच्यावर जवळचे groundwater पिणाऱ्या लहान बाळांमध्ये <i>methaemoglobinaemia (\"blue-baby\")</i> चा धोका."
        else:
            warn_en = "This stays below WHO/BIS limits, but cumulative N losses still reduce yield efficiency."
            warn_hi = "यह WHO/BIS सीमा से नीचे है, फिर भी कुल N नुक़सान yield efficiency कम करता है।"
            warn_te = "ఇది WHO/BIS పరిమితి కంటే తక్కువ, అయినా మొత్తం N నష్టం yield efficiency ను తగ్గిస్తుంది."
            warn_mr = "हे WHO/BIS मर्यादेखाली आहे, परंतु एकूण N नुकसान yield efficiency कमी करते."
        paragraph = _t(
            f"{emoji} Approximately <b>{mg_per_l:.1f} mg/L</b> could leach into local groundwater (1,000 m³/ha runoff proxy). WHO drinking-water limit: <b>50 mg/L</b>; BIS IS 10500: <b>45 mg/L</b>. {warn_en} Avoid topdressing N before heavy rain and maintain bunds & vegetative buffers — these practices can <b>cut NO₃⁻ leaching by 30–50%</b>, and switching to split-N + neem-coated urea adds another <b>~20%</b> reduction.",
            f"{emoji} लगभग <b>{mg_per_l:.1f} mg/L</b> स्थानीय groundwater में जा सकता है (1,000 m³/ha runoff अनुमान)। WHO सीमा: <b>50 mg/L</b>; BIS IS 10500: <b>45 mg/L</b>। {warn_hi} भारी बारिश से पहले N न डालें और मेड़ व वनस्पति buffer बनाएँ — ये उपाय <b>NO₃⁻ leaching 30–50% तक कम</b> करते हैं, और split-N + neem-coated urea से <b>~20%</b> और कमी होती है।",
            f"{emoji} సుమారు <b>{mg_per_l:.1f} mg/L</b> స్థానిక groundwater లోకి వెళ్ళవచ్చు (1,000 m³/ha runoff అంచనా). WHO పరిమితి: <b>50 mg/L</b>; BIS IS 10500: <b>45 mg/L</b>. {warn_te} భారీ వర్షానికి ముందు N వేయొద్దు, బండ్లు మరియు vegetative buffers నిర్వహించండి — ఈ చర్యలు <b>NO₃⁻ leaching ను 30–50% తగ్గిస్తాయి</b>, split-N + neem-coated urea తో మరో <b>~20%</b> తగ్గింపు.",
            lang,
            mr=f"{emoji} सुमारे <b>{mg_per_l:.1f} mg/L</b> स्थानिक groundwater मध्ये झिरपू शकते (1,000 m³/ha runoff अंदाज). WHO मर्यादा: <b>50 mg/L</b>; BIS IS 10500: <b>45 mg/L</b>. {warn_mr} मोठ्या पावसाआधी N टाकू नका आणि बांध व वनस्पती buffers टिकवा — हे उपाय <b>NO₃⁻ झिरप 30–50% कमी</b> करतात, split-N + neem-coated urea ने आणखी <b>~20%</b> कपात.")
        refs = cite("who_no3", "bis")

    elif domain == "nh3":
        band, emoji, color = _band(value, 5, 15)
        title = _t("🌬️ Ammonia (NH₃)", "🌬️ अमोनिया (NH₃)", "🌬️ अम्मोनिया (NH₃)", lang,
                   mr="🌬️ अमोनिया (NH₃)")
        paragraph = _t(
            f"{emoji} Your <b>{value:,.2f} kg NH₃/ha</b> volatilises into air, contributing to <b>PM₂.₅ formation</b> (linked to respiratory illness) and acidifies soil over time, reducing nutrient availability. Incorporate urea within 24 h, avoid hot/windy mid-day spreading, and use neem-coated urea to cut NH₃ loss by 10–15%.",
            f"{emoji} आपका <b>{value:,.2f} kg NH₃/ha</b> हवा में मिलकर <b>PM₂.₅</b> बनाता है (श्वसन रोगों से जुड़ा) और मिट्टी को धीरे-धीरे अम्लीय करता है, पोषक उपलब्धता घटाता है। यूरिया 24 घंटे में मिट्टी में मिलाएँ, गर्म/तेज़ हवा वाले दिनों में न डालें, और neem-coated urea से NH₃ नुक़सान 10–15% कम करें।",
            f"{emoji} मी <b>{value:,.2f} kg NH₃/ha</b> गालिलो कलिसि <b>PM₂.₅</b> ऐर्पडटानिकि दारितीस्तुंदि (श्वासकोश व्याधुलतो संबंधं), मट्टिनि क्रमंगा अम्लीयं चेस्तुंदि, पोषक लभ्यत तग्गुतुंदि. यूरियानु 24 गंटललो मट्टिलो कलपंडि, वेडि/गालि ऎक्कुव उन्न middays लो वेयॊद्दु, neem-coated urea तो NH₃ नष्टं 10–15% तग्गिंचंडि.",
            lang,
            mr=f"{emoji} तुमचे <b>{value:,.2f} kg NH₃/ha</b> हवेत मिसळून <b>PM₂.₅</b> तयार करते (श्वसन आजाराशी संबंधित) आणि माती हळूहळू आम्लधर्मी करते, पोषण कमी करते. युरिया 24 तासांत मातीत मिसळा, गरम/वार्याच्या दुपारी टाकू नका, neem-coated urea वापरून NH₃ नुकसान 10–15% कमी करा.")
        refs = cite("who_air", "salca")

    elif domain == "po4":
        band, emoji, color = _band(value, 0.05, 0.2)
        title = _t("💧 Phosphate (PO₄³⁻)", "💧 फ़ॉस्फेट (PO₄³⁻)", "💧 फास्फेट् (PO₄³⁻)", lang,
                   mr="💧 फॉस्फेट (PO₄³⁻)")
        paragraph = _t(
            f"{emoji} Your <b>{value:,.3f} kg PO₄/ha</b> runoff fuels <b>algal blooms</b> in ponds and irrigation tanks once concentrations exceed <b>0.1 mg P/L</b>, killing fish through oxygen depletion. Apply DAP only at recommended rates with <b>band placement (cuts P runoff by 40–60%)</b>, never just before predicted rainfall, and add 5 t/ha FYM to substitute <b>~25%</b> of synthetic P needs.",
            f"{emoji} आपका <b>{value:,.3f} kg PO₄/ha</b> runoff तालाबों में <b>algal blooms</b> पैदा करता है जब P >0.1 mg/L हो — मछलियाँ ऑक्सीजन की कमी से मरती हैं। DAP केवल अनुशंसित मात्रा में, <b>band placement (P runoff 40–60% कम)</b> से डालें, बारिश से पहले कभी नहीं, और 5 t/ha FYM से <b>~25%</b> सिंथेटिक P की जगह लें।",
            f"{emoji} मी <b>{value:,.3f} kg PO₄/ha</b> runoff चेरुवुल्लो <b>algal blooms</b> कु दारितीस्तुंदि (P >0.1 mg/L वद्द) — चेपलु आक्सिजन् लेक चनिपोतायि. DAP नु सिफारसु चेसिन मोतादुलो <b>band placement (P runoff 40–60% तग्गिंपु)</b> तो मात्रमे वेयंडि, वर्षानिकि मुंदु ऎप्पुडू वेयॊद्दु, 5 t/ha FYM तो <b>~25%</b> synthetic P नु भर्ती चेयंडि.",
            lang,
            mr=f"{emoji} तुमचा <b>{value:,.3f} kg PO₄/ha</b> runoff तळ्यांमध्ये P >0.1 mg/L झाल्यावर <b>शैवाळ फुलोरा (algal blooms)</b> निर्माण करतो — मासे ऑक्सिजनच्या कमतरतेने मरतात. DAP फक्त शिफारस केलेल्या प्रमाणात, <b>band placement (P runoff 40–60% कमी)</b> ने द्या; पावसाआधी कधीच टाकू नका, आणि 5 t/ha शेणखताने <b>~25%</b> synthetic P बदला.")
        refs = cite("who_no3", "salca")

    elif domain == "ecotox":
        band, emoji, color = _band(value, 5_000, 20_000)
        title = _t("☠️ Ecotoxicity", "☠️ इको-विषाक्तता (Ecotoxicity)", "☠️ ऎको-टाक्सिसिटी (Ecotoxicity)", lang,
                   mr="☠️ परिसंस्था-विषाक्तता (Ecotoxicity)")
        zn = ctx.get("zinc", 0)
        zn_note_en = f" Your Zn input is {zn:.0f} kg/ha — " + ("at safe levels." if zn <= 20 else "<b>above 20 kg/ha</b>; reduce unless soil-test confirms deficiency.")
        zn_note_hi = f" आपका Zn input {zn:.0f} kg/ha है — " + ("सुरक्षित स्तर पर।" if zn <= 20 else "<b>20 kg/ha से ऊपर</b>; soil-test से कमी की पुष्टि न हो तो घटाएँ।")
        zn_note_te = f" मी Zn input {zn:.0f} kg/ha — " + ("सुरक्षित स्थायिलो." if zn <= 20 else "<b>20 kg/ha कंटे ऎक्कुव</b>; soil-test लो deficiency निर्धारिस्ते तप्प तग्गिंचंडि.")
        zn_note_mr = f" तुमचा Zn input {zn:.0f} kg/ha — " + ("सुरक्षित पातळीवर." if zn <= 20 else "<b>20 kg/ha पेक्षा जास्त</b>; soil-test मध्ये कमतरता सिद्ध झाल्याशिवाय कमी करा.")
        paragraph = _t(
            f"{emoji} Your soil ecotoxicity score of <b>{value:,.0f} CTUe</b> is driven mostly by Zinc — Zn alone is <b>~120× more toxic per kg</b> than N, P, K (612.9 vs 2.7–5.2 CTUe/kg).{zn_note_en} Split Zn applications with FYM rather than concentrated dose — this can <b>cut ecotoxicity by 40–60%</b>, and substituting 5 t/ha FYM for synthetic Zn delivers another <b>~20%</b> reduction.",
            f"{emoji} आपका मिट्टी ecotoxicity स्कोर <b>{value:,.0f} CTUe</b> मुख्यतः Zinc से आता है — Zn प्रति kg <b>~120× अधिक विषाक्त</b> है N, P, K की तुलना में (612.9 vs 2.7–5.2 CTUe/kg)।{zn_note_hi} Zn को एक साथ नहीं, FYM के साथ split करके दें — इससे <b>ecotoxicity 40–60% कम</b> होती है, और 5 t/ha FYM से synthetic Zn बदलने पर <b>~20%</b> और कमी।",
            f"{emoji} मी मट्टि ecotoxicity स्कोर् <b>{value:,.0f} CTUe</b> मुख्यंगा Zinc वल्ल — Zn प्रति kg <b>~120× ऎक्कुव विषपूरितं</b> N, P, K कंटे (612.9 vs 2.7–5.2 CTUe/kg).{zn_note_te} Zn नु ओकेसारि काकुंडा FYM तो splits लो वेयंडि — दीनितो <b>ecotoxicity 40–60% तग्गुतुंदि</b>, 5 t/ha FYM तो synthetic Zn भर्ती चेस्ते मरो <b>~20%</b> तग्गिंपु.",
            lang,
            mr=f"{emoji} तुमच्या मातीचा ecotoxicity स्कोर <b>{value:,.0f} CTUe</b> मुख्यतः Zinc मुळे — Zn प्रति kg <b>~120× जास्त विषारी</b> आहे N, P, K च्या तुलनेत (612.9 vs 2.7–5.2 CTUe/kg).{zn_note_mr} Zn एकाच वेळी न देता शेणखतासोबत splits मध्ये द्या — यामुळे <b>ecotoxicity 40–60% कमी</b> होते, आणि 5 t/ha शेणखताने synthetic Zn बदलल्यास आणखी <b>~20%</b> कपात.")
        refs = cite("salca", "icar_inm")

    elif domain == "credits":
        inr_low = value * R["ccts_price_low"]
        inr_high = value * R["ccts_price_high"]
        bags = inr_high / (R["urea_price_inr_kg"] * 45)
        wages = inr_high / R["farm_wage_inr_day"]
        band, emoji, color = _band(value, 0.3, 1.0)
        title = _t("💎 Carbon Credit Potential", "💎 कार्बन क्रेडिट संभावना", "💎 కార్బన్ క్రెడిట్ సామర్థ్యం", lang,
                   mr="💎 कार्बन क्रेडिट क्षमता")
        if band == "low":
            tone_en = "is on the lower side. To reach a meaningful credit volume, raise FYM to ~10 t/ha and add 1.5–2 t/ha compost"
            tone_hi = "कम है। meaningful credits के लिए FYM बढ़ाकर ~10 t/ha करें और 1.5–2 t/ha compost जोड़ें"
            tone_te = "తక్కువగా ఉంది. meaningful credits కోసం FYM ను ~10 t/ha కు పెంచి, 1.5–2 t/ha compost జోడించండి"
            tone_mr = "कमी आहे. लक्षणीय credits साठी शेणखत ~10 t/ha पर्यंत वाढवा आणि 1.5–2 t/ha कंपोस्ट घाला"
        elif band == "mid":
            tone_en = "is moderate. Maintain a 3-season log of FYM/compost rates with photos and soil-test data"
            tone_hi = "मध्यम है। 3 सीज़न का FYM/compost log, फ़ोटो और soil-test data रखें"
            tone_te = "మధ్యస్థంగా ఉంది. 3 సీజన్ల FYM/compost log, ఫొటోలు, soil-test data నిర్వహించండి"
            tone_mr = "मध्यम आहे. 3 हंगामांचे शेणखत/कंपोस्टचे log, फोटो आणि soil-test data ठेवा"
        else:
            tone_en = "is strong. Register now with a verified aggregator (NCDEX / Verra / Gold Standard) to monetise this"
            tone_hi = "बहुत अच्छी है। अभी verified aggregator (NCDEX / Verra / Gold Standard) के साथ register करें"
            tone_te = "చాలా బలంగా ఉంది. వెంటనే verified aggregator (NCDEX / Verra / Gold Standard) తో register చేసుకోండి"
            tone_mr = "उत्तम आहे. आत्ताच verified aggregator (NCDEX / Verra / Gold Standard) सोबत नोंदणी करा"
        paragraph = _t(
            f"{emoji} Your potential of <b>{value:.3f} t CO₂-eq/ha</b> translates to <b>₹{inr_low:,.0f}–₹{inr_high:,.0f}/ha</b> at the CCTS price band of ₹600–₹900/t — equivalent to <b>~{bags:.1f} bags of urea</b> or <b>~{wages:.0f} days of farm wages</b>. This {tone_en}.",
            f"{emoji} आपकी संभावना <b>{value:.3f} t CO₂-eq/ha</b> = <b>₹{inr_low:,.0f}–₹{inr_high:,.0f}/ha</b> (CCTS दर ₹600–₹900/t) — यानी <b>~{bags:.1f} bags यूरिया</b> या <b>~{wages:.0f} दिन की मज़दूरी</b>। यह {tone_hi}।",
            f"{emoji} మీ సామర్థ్యం <b>{value:.3f} t CO₂-eq/ha</b> = <b>₹{inr_low:,.0f}–₹{inr_high:,.0f}/ha</b> (CCTS ధర ₹600–₹900/t) — అంటే <b>~{bags:.1f} bags యూరియా</b> లేదా <b>~{wages:.0f} రోజుల వేతనం</b>. ఇది {tone_te}.",
            lang,
            mr=f"{emoji} तुमची क्षमता <b>{value:.3f} t CO₂-eq/ha</b> = <b>₹{inr_low:,.0f}–₹{inr_high:,.0f}/ha</b> (CCTS दर ₹600–₹900/t) — म्हणजे <b>~{bags:.1f} bags युरिया</b> किंवा <b>~{wages:.0f} दिवसांची मजुरी</b>. ही {tone_mr}.")
        refs = cite("moefcc_ccts", "verra", "gold_awm")

    elif domain == "blend_savings":
        gwp_saved = ctx.get("gwp_saved", 0)
        cost_delta = ctx.get("cost_delta", 0)
        alpha = ctx.get("alpha", 0)
        title = _t("🌱 Blend Trade-off", "🌱 मिश्रण समझौता (Blend)", "🌱 Blend ట్రేడ్-ఆఫ్", lang,
                   mr="🌱 मिश्रण तडजोड (Blend)")
        if alpha == 0:
            color, emoji = "#0ea5e9", "ℹ️"
            paragraph = _t(
                "Move the slider above 0% to see how shifting toward organic affects emissions, cost, and credit potential.",
                "Slider को 0% से ऊपर ले जाएँ और देखें organic की ओर बढ़ने पर उत्सर्जन, लागत और credits कैसे बदलते हैं।",
                "Slider ను 0% పైన పెంచి organic వైపు మారితే ఉద్గారాలు, ఖర్చు మరియు credits ఎలా మారతాయో చూడండి.",
                lang,
                mr="Slider 0% च्या वर हलवा आणि organic कडे वळल्यावर उत्सर्जन, खर्च आणि credits कसे बदलतात ते पाहा.")
        elif gwp_saved <= 0:
            color, emoji = "#f59e0b", "⚠️"
            paragraph = _t(
                f"⚠️ This blend <b>increases</b> GWP compared to fully conventional (Δ = +{abs(gwp_saved):,.1f} kg CO₂-eq/ha). Reduce organic input volumes or check for over-application of FYM (>15 t/ha drives high CH₄).",
                f"⚠️ यह मिश्रण पूरी तरह conventional की तुलना में GWP <b>बढ़ाता</b> है (Δ = +{abs(gwp_saved):,.1f} kg CO₂-eq/ha)। organic input घटाएँ या FYM over-application जाँचें (>15 t/ha CH₄ बढ़ाता है)।",
                f"⚠️ ఈ blend పూర్తి conventional తో పోలిస్తే GWP ను <b>పెంచుతుంది</b> (Δ = +{abs(gwp_saved):,.1f} kg CO₂-eq/ha). organic input తగ్గించండి లేదా FYM over-application తనిఖీ చేయండి (>15 t/ha వద్ద CH₄ పెరుగుతుంది).",
                lang,
                mr=f"⚠️ हे मिश्रण पूर्णतः conventional च्या तुलनेत GWP <b>वाढवते</b> (Δ = +{abs(gwp_saved):,.1f} kg CO₂-eq/ha). organic input कमी करा किंवा शेणखताचा अतिवापर तपासा (>15 t/ha मुळे CH₄ वाढतो).")
        else:
            km = gwp_saved / R["co2_per_km_car"]
            trees = gwp_saved / R["co2_per_tree_year"]
            ratio = cost_delta / gwp_saved if gwp_saved else 0
            if cost_delta < 0:
                color, emoji = "#16a34a", "🌟"
                cost_phrase_en = f"AND <b>saves ₹{abs(cost_delta):,.0f}/ha</b>"
                cost_phrase_hi = f"और <b>₹{abs(cost_delta):,.0f}/ha बचाता</b> है"
                cost_phrase_te = f"మరియు <b>₹{abs(cost_delta):,.0f}/ha ఆదా</b> చేస్తుంది"
                cost_phrase_mr = f"आणि <b>₹{abs(cost_delta):,.0f}/ha वाचवते</b>"
            else:
                color, emoji = "#16a34a", "🌱"
                cost_phrase_en = f"at an extra <b>₹{cost_delta:,.0f}/ha</b> (₹{ratio:,.1f} per kg CO₂ avoided)"
                cost_phrase_hi = f"<b>₹{cost_delta:,.0f}/ha अतिरिक्त</b> लागत पर (₹{ratio:,.1f}/kg CO₂)"
                cost_phrase_te = f"<b>₹{cost_delta:,.0f}/ha అదనపు</b> ఖర్చుతో (₹{ratio:,.1f}/kg CO₂)"
                cost_phrase_mr = f"<b>₹{cost_delta:,.0f}/ha जादा</b> खर्चात (₹{ratio:,.1f}/kg CO₂)"
            paragraph = _t(
                f"{emoji} This {int(alpha*100)}% organic blend avoids <b>{gwp_saved:,.1f} kg CO₂-eq/ha</b> ({km:,.0f} km of car driving, ~{trees:.1f} trees of yearly absorption) {cost_phrase_en}. Recover any cost gap via CCTS soil-carbon credits (Tab 3) or PM-PRANAM organic incentives.",
                f"{emoji} यह {int(alpha*100)}% organic मिश्रण <b>{gwp_saved:,.1f} kg CO₂-eq/ha</b> बचाता है ({km:,.0f} किमी कार + ~{trees:.1f} पेड़/साल) {cost_phrase_hi}। लागत अंतर CCTS soil-carbon credits (Tab 3) या PM-PRANAM से वसूलें।",
                f"{emoji} ఈ {int(alpha*100)}% organic blend <b>{gwp_saved:,.1f} kg CO₂-eq/ha</b> ఆదా చేస్తుంది ({km:,.0f} కి.మీ. కారు + ~{trees:.1f} చెట్లు/ఏడాది) {cost_phrase_te}. ఖర్చు తేడాను CCTS soil-carbon credits (Tab 3) లేదా PM-PRANAM తో recover చేయండి.",
                lang,
                mr=f"{emoji} हे {int(alpha*100)}% organic मिश्रण <b>{gwp_saved:,.1f} kg CO₂-eq/ha</b> वाचवते ({km:,.0f} किमी कार + ~{trees:.1f} झाडे/वर्ष) {cost_phrase_mr}. खर्चातील फरक CCTS soil-carbon credits (Tab 3) किंवा PM-PRANAM द्वारे भरून काढा.")
        refs = cite("epa", "moefcc_ccts", "frontiers22")

    if not paragraph:
        return ""

    return f"""
    <div style='background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border-left: 4px solid {color}; border-radius: 12px;
                padding: 0.85rem 1.1rem; margin: 0.6rem 0;
                box-shadow: 0 4px 10px rgba(15,23,42,0.05);'>
        <div style='font-weight:700; color:#0f172a; margin-bottom:0.4rem; font-size:0.95rem;'>
            {title}
        </div>
        <div style='color:#334155; font-size:0.88rem; line-height:1.65;'>
            {paragraph}
        </div>
        <div style='margin-top:0.55rem; line-height:1.9;'>{refs}</div>
    </div>
    """


def render_inference_section(cards, key_suffix=""):
    """Render the bottom-of-tab inference section with English/Hindi/Telugu language tabs.

    cards: list of (domain, value, ctx) tuples to render.
    """
    if not cards:
        return
    st.markdown("---")
    st.markdown("### 📖 What This Means for Your Field · आपके खेत के लिए मतलब · మీ పొలానికి అర్థం · तुमच्या शेतासाठी अर्थ")
    lang_tabs = st.tabs(["🇬🇧 English", "🇮🇳 हिंदी", "🇮🇳 తెలుగు", "🇮🇳 मराठी"])
    for tab, lang in zip(lang_tabs, ["en", "hi", "te", "mr"]):
        with tab:
            for domain, value, ctx in cards:
                html = build_inference_card(domain, value, ctx, lang)
                if html:
                    st.markdown(html, unsafe_allow_html=True)
    render_references_expander()


def render_references_expander():
    with st.expander("📚 References & Data Sources"):
        st.markdown("All inferences and benchmarks above are grounded in:")
        for key, (name, url) in SOURCES.items():
            if url == "#":
                st.markdown(f"- **{name}**")
            else:
                st.markdown(f"- [{name}]({url})")
        st.caption("Indicative interpretations only — local soil, climate, and water conditions may shift exact thresholds.")


def calculate_soc_credits(manure, compost, buffer_pct):
    """Estimate soil-carbon-only credit potential from FYM and compost inputs."""
    c_to_co2 = 3.667

    # Stabilized carbon assumptions for amendments.
    fym_dm, fym_c, fym_h = 0.25, 0.25, 0.25
    compost_dm, compost_c, compost_h = 0.55, 0.25, 0.35

    soc_fym_kgc = manure * fym_dm * fym_c * fym_h
    soc_compost_kgc = compost * compost_dm * compost_c * compost_h
    soc_fym_tco2 = (soc_fym_kgc / 1000) * c_to_co2
    soc_compost_tco2 = (soc_compost_kgc / 1000) * c_to_co2

    soc_total_tco2 = soc_fym_tco2 + soc_compost_tco2
    soc_credits_tco2 = soc_total_tco2 * (1 - buffer_pct / 100)
    buffer_withheld_tco2 = soc_total_tco2 - soc_credits_tco2

    return {
        "fym_soc_tco2": soc_fym_tco2,
        "compost_soc_tco2": soc_compost_tco2,
        "fym_credits_tco2": soc_fym_tco2 * (1 - buffer_pct / 100),
        "compost_credits_tco2": soc_compost_tco2 * (1 - buffer_pct / 100),
        "soc_before_buffer_tco2": soc_total_tco2,
        "credits_tco2": soc_credits_tco2,
        "buffer_withheld_tco2": buffer_withheld_tco2,
        "value_low_inr": soc_credits_tco2 * 600,
        "value_high_inr": soc_credits_tco2 * 900,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    hero_left, hero_right = st.columns([2.4, 1.6], gap="small")
    with hero_left:
        st.markdown("""
        <div style='display:inline-flex; align-items:center; gap:0.45rem; background:rgba(16,185,129,0.12); color:#065f46;
                    padding:0.35rem 0.85rem; border-radius:999px; font-weight:600; font-size:0.82rem;
                    border:1px solid rgba(16,185,129,0.25); margin-bottom:0.85rem;'>
            <span style='width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 0 4px rgba(16,185,129,0.18);'></span>
            AI · LCA · Climate Smart
        </div>
        <h1 style='margin:0 0 0.35rem; font-size: clamp(1.8rem, 3.4vw, 2.6rem); color:#0f172a;'>
            Smart farming for a <span style='background:linear-gradient(90deg,#16a34a,#0f766e); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>sustainable Bharat</span>
        </h1>
        <p style='color:#334155; font-size:1.02rem; line-height:1.6; margin:0.4rem 0 0.6rem;'>
            Predict and compare the <b>full environmental impact</b> of fertiliser application in rice cultivation —
            including upstream production emissions and field-level emissions, powered by ISO 14040/44-compliant LCA.
        </p>
        <p style='color:#0f172a; font-weight:600; margin:0;'>
            Fast insights, actionable comparisons, and cost-aware recommendations for rice farmers and agronomists.
        </p>
        """, unsafe_allow_html=True)
    with hero_right:
        st.markdown("""
        <div style='position:relative; background: linear-gradient(135deg, #ecfdf5 0%, #bbf7d0 100%);
                    border-radius: 28px; padding: 2rem 1.6rem; text-align: center;
                    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
                    border: 1px solid rgba(16,185,129,0.25); overflow:hidden;'>
            <div style='position:absolute; top:-40px; right:-40px; width:160px; height:160px;
                        background: radial-gradient(circle, rgba(16,185,129,0.35), transparent 70%); border-radius:50%;'></div>
            <h2 style='color:#064e3b; margin:0; padding:0; line-height:1.15; font-size:1.9rem; position:relative;'>ClimateKrishi AI</h2>
            <h3 style='color:#065f46; margin:0.35rem 0 0.6rem; padding:0; line-height:1.2; font-size:1.05rem; font-weight:600; position:relative;'>A Climate Smart Agriculture LCA Tool</h3>
            <p style='color:#065f46; margin:0 0 0.9rem; font-size:0.92rem; line-height:1.45; position:relative;'>Enter your inputs and instantly compare the environmental cost of rice fertiliser choices.</p>
            <div style='font-size: 2.6rem; line-height: 1; position:relative;'>🌾🌾🌾🌾🌾🌾🌾🌾🌾</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
with st.container():
    card1, card2, card3, card4, card5 = st.columns(5)
    with card1:
        st.markdown("""
        <div class='ck-feature-card'>
            <h4 style='color:#047857;'>Real-time soil insights</h4>
            <p style='color:#334155;'>Understand the environmental footprint of each nutrient input.</p>
        </div>
        """, unsafe_allow_html=True)
    with card2:
        st.markdown("""
        <div class='ck-feature-card'>
            <h4 style='color:#047857;'>LCA Predictor</h4>
            <p style='color:#334155;'>Compare conventional and organic strategies at a glance.</p>
        </div>
        """, unsafe_allow_html=True)
    with card3:
        st.markdown("""
        <div class='ck-feature-card'>
            <h4 style='color:#047857;'>Field Emission Calculator</h4>
            <p style='color:#334155;'>Estimate CH₄, N₂O and nutrient emissions by irrigation type.</p>
        </div>
        """, unsafe_allow_html=True)
    with card4:
        st.markdown("""
        <div class='ck-feature-card'>
            <h4 style='color:#047857;'>Carbon Credit Potential</h4>
            <p style='color:#334155;'>Estimate soil-carbon credit potential and value per hectare.</p>
        </div>
        """, unsafe_allow_html=True)
    with card5:
        st.markdown("""
        <div class='ck-feature-card'>
            <h4 style='color:#047857;'>Model Validation</h4>
            <p style='color:#334155;'>Built with real-world rice cultivation datasets from ICAR-IIRR & KVK.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 LCA Impact Predictor",
    "🎚️ Organic–Conventional Gradient",
    "💲 Carbon Credit Potential",
    "🌱 Field Emission Calculator",
    "📊 Model Information",
])

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
                    st.caption("📊 **Impact Bar Chart** — magnitude of each of the 4 LCA impact categories produced by your selected NPK + Zn dose. Taller bar = larger environmental burden.")

                    # Visual gauges – your impact vs the upper-range benchmark (max recommended inputs)
                    bench = predict_conventional(150, 60, 40, 30)
                    st.markdown("#### 📉 Impact vs Upper-Range Benchmark")
                    st.caption("Each ring shows where your input lands compared to the maximum recommended Conventional dose.")
                    g1, g2, g3, g4 = st.columns(4)
                    with g1:
                        st.markdown(render_value_gauge_html("Global Warming", out[0], bench[0],
                                                           unit="kg CO₂-eq", icon="🌍", value_fmt="{:,.0f}"), unsafe_allow_html=True)
                    with g2:
                        st.markdown(render_value_gauge_html("Eutrophication", out[1], bench[1],
                                                           unit="kg P-eq", icon="💧", value_fmt="{:.4f}"), unsafe_allow_html=True)
                    with g3:
                        st.markdown(render_value_gauge_html("Acidification", out[2], bench[2],
                                                           unit="kg SO₂-eq", icon="🌫️", value_fmt="{:.2f}"), unsafe_allow_html=True)
                    with g4:
                        st.markdown(render_value_gauge_html("Ecotoxicity", out[3], bench[3],
                                                           unit="CTUe", icon="☠️", value_fmt="{:,.0f}"), unsafe_allow_html=True)

                    # Radar chart vs benchmark
                    st.markdown("#### 🕸️ Impact Profile (Radar)")
                    st.caption("Your input vs maximum-recommended dose, normalized 0–100 across all 4 categories.")
                    st.plotly_chart(build_radar_chart([
                        ("Your Input", list(out), "#1d4ed8"),
                        ("Max Dose Benchmark", list(bench), "#ef4444"),
                    ], title="Your Conventional Input vs Benchmark"), use_container_width=True)
                    st.caption("🕸️ **Radar Plot** — each axis is one impact category, normalized 0–100. The smaller the polygon, the lower the overall environmental footprint.")

                    # Bullet charts per nutrient – are inputs in recommended range?
                    st.markdown("#### 🎯 Input Levels vs Recommended Range (Bullet Charts)")
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        st.altair_chart(build_bullet_chart(sN, *CONV_RANGES['N'], "Nitrogen (N)", "kg/ha"),
                                        use_container_width=True)
                        st.altair_chart(build_bullet_chart(sK, *CONV_RANGES['K'], "Potassium (K)", "kg/ha"),
                                        use_container_width=True)
                    with bcol2:
                        st.altair_chart(build_bullet_chart(sP, *CONV_RANGES['P'], "Phosphorus (P)", "kg/ha"),
                                        use_container_width=True)
                        st.altair_chart(build_bullet_chart(sZn, *CONV_RANGES['Zn'], "Zinc (Zn)", "kg/ha"),
                                        use_container_width=True)
                    st.caption("🎯 **Bullet Charts** — the dark bar is your applied dose; the green band is the agronomically recommended range. Bars inside the band indicate balanced fertilisation.")

                    # Waffle: per-nutrient share of GWP based on IMPACT_DATA + actual dose
                    st.markdown("#### 🟩 GWP Composition by Nutrient (Waffle)")
                    parts = [
                        ("Nitrogen", sN * IMPACT_DATA["Global Warming (kg CO₂-eq)"][0], "#1f77b4"),
                        ("Phosphorus", sP * IMPACT_DATA["Global Warming (kg CO₂-eq)"][1], "#ff7f0e"),
                        ("Potassium", sK * IMPACT_DATA["Global Warming (kg CO₂-eq)"][2], "#2ca02c"),
                        ("Zinc", sZn * IMPACT_DATA["Global Warming (kg CO₂-eq)"][3], "#d62728"),
                    ]
                    st.markdown(render_waffle_html(parts, total_label="Upstream GWP"), unsafe_allow_html=True)
                    st.caption("🟩 **Waffle Chart** — each of the 100 squares represents 1% of upstream GWP. Color shows which nutrient (N/P/K/Zn) is the dominant climate hotspot in your blend.")

                    # Farmer-friendly inferences
                    render_inference_section([
                        ("gwp_total", out[0], {"irrigation": "", "amendments_used": False}),
                        ("ecotox",   out[3], {"zinc": sZn}),
                    ], key_suffix="t1conv")
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
                    st.caption("📊 **Impact Bar Chart** — the 4 LCA scores produced by your selected manure + compost rates. Lower bars indicate a lighter environmental footprint per hectare.")

                    # Visual gauges – your impact vs the upper-range benchmark (max recommended inputs)
                    bench = predict_organic(15000, 2000)
                    st.markdown("#### 📉 Impact vs Upper-Range Benchmark")
                    st.caption("Each ring shows where your input lands compared to the maximum recommended Organic dose.")
                    g1, g2, g3, g4 = st.columns(4)
                    with g1:
                        st.markdown(render_value_gauge_html("Global Warming", out[0], bench[0],
                                                           unit="kg CO₂-eq", icon="🌍", value_fmt="{:,.0f}"), unsafe_allow_html=True)
                    with g2:
                        st.markdown(render_value_gauge_html("Eutrophication", out[1], bench[1],
                                                           unit="kg P-eq", icon="💧", value_fmt="{:.4f}"), unsafe_allow_html=True)
                    with g3:
                        st.markdown(render_value_gauge_html("Acidification", out[2], bench[2],
                                                           unit="kg SO₂-eq", icon="🌫️", value_fmt="{:.2f}"), unsafe_allow_html=True)
                    with g4:
                        st.markdown(render_value_gauge_html("Ecotoxicity", out[3], bench[3],
                                                           unit="CTUe", icon="☠️", value_fmt="{:,.0f}"), unsafe_allow_html=True)

                    # Radar chart
                    st.markdown("#### 🕸️ Impact Profile (Radar)")
                    st.caption("Your input vs maximum-recommended dose, normalized 0–100 across all 4 categories.")
                    st.plotly_chart(build_radar_chart([
                        ("Your Input", list(out), "#15803d"),
                        ("Max Dose Benchmark", list(bench), "#ef4444"),
                    ], title="Your Organic Input vs Benchmark"), use_container_width=True)
                    st.caption("🕸️ **Radar Plot** — visualizes your organic-input footprint across all 4 impact categories (normalized 0–100) vs the maximum-dose benchmark.")

                    # Bullet charts per amendment
                    st.markdown("#### 🎯 Input Levels vs Recommended Range (Bullet Charts)")
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        st.altair_chart(build_bullet_chart(sManure, *ORG_RANGES['Manure'],
                                                           "Farm Yard Manure", "kg/ha"),
                                        use_container_width=True)
                    with bcol2:
                        st.altair_chart(build_bullet_chart(sCompost, *ORG_RANGES['Compost'],
                                                           "Compost", "kg/ha"),
                                        use_container_width=True)
                    st.caption("🎯 **Bullet Charts** — your applied FYM and Compost rates against ICAR-recommended ranges. Aim to keep both bars within the green band for balanced organic nutrition.")

                    # Farmer-friendly inferences
                    render_inference_section([
                        ("gwp_total", out[0], {"irrigation": "", "amendments_used": True}),
                        ("ecotox",   out[3], {"zinc": 0}),
                    ], key_suffix="t1org")
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
                    col1, col2, col3 = st.columns([2, 2, 1.15])
                    with col1:
                        st.metric(f"{cat} — A", f"{formats[i].format(outA[i])} {units[i]}")
                    with col2:
                        st.metric(f"{cat} — B", f"{formats[i].format(outB[i])} {units[i]}")
                    with col3:
                        st.markdown(build_winner_badge(outA[i], outB[i]), unsafe_allow_html=True)
                
                compare_df = build_comparison_dataframe(outA, outB)
                st.altair_chart(build_comparison_chart(compare_df), use_container_width=True)
                st.caption("📊 **Side-by-side Comparison Bars** — each impact category plotted for Combination A and B together. Shorter bars = lower environmental impact.")

                # Visual gauges – B as % of A (lower than 100% means B has less impact than A)
                st.markdown("#### 📉 Combination B vs A (lower is better)")
                st.caption("Each ring shows B's impact as a percentage of A. <100% (green) means B is greener.")
                cmp_g = st.columns(4)
                cmp_data = [
                    ("Global Warming", "🌍", outB[0], outA[0], "vs A", "{:,.0f}"),
                    ("Eutrophication", "💧", outB[1], outA[1], "vs A", "{:.4f}"),
                    ("Acidification", "🌫️", outB[2], outA[2], "vs A", "{:.2f}"),
                    ("Ecotoxicity", "☠️", outB[3], outA[3], "vs A", "{:,.0f}"),
                ]
                for col, (lbl, ic, val, base, u, fmt) in zip(cmp_g, cmp_data):
                    with col:
                        st.markdown(render_value_gauge_html(lbl, val, base, unit=u, icon=ic, value_fmt=fmt),
                                    unsafe_allow_html=True)

                # Radar A vs B
                st.markdown("#### 🕸️ Profile Comparison (Radar)")
                st.plotly_chart(build_radar_chart([
                    ("Combination A", list(outA), "#3b82f6"),
                    ("Combination B", list(outB), "#f97316"),
                ], title="A vs B — Impact Profile"), use_container_width=True)
                st.caption("🕸️ **Radar Comparison** — overlapping polygons of A and B; the smaller polygon dominates the larger across the four impact categories.")

                # Diff arrows
                st.markdown("#### 🔀 Side-by-Side Difference")
                diff_unit_fmt = [("kg CO₂-eq", "{:,.1f}"), ("kg P-eq", "{:.4f}"),
                                 ("kg SO₂-eq", "{:.3f}"), ("CTUe", "{:,.0f}")]
                for i, cat in enumerate(IMPACT_NAMES):
                    u, f = diff_unit_fmt[i]
                    st.markdown(render_diff_arrow_html(outA[i], outB[i], unit=u,
                                label=cat, fmt=f), unsafe_allow_html=True)
                st.markdown(render_diff_arrow_html(costA, costB, unit="₹/ha",
                            label="Input Cost", fmt="{:,.0f}"), unsafe_allow_html=True)
                st.caption("🔀 **Difference Arrows** — green arrow means B is lower (better); red means B is higher than A. Magnitude shows the absolute change in each metric.")

                st.markdown("---")
                st.markdown("**Cost vs GWP Trade-off**")
                st.altair_chart(build_cost_vs_impact_scatter(outA, outB, costA, costB, 
                                "Comb. A", "Comb. B"), use_container_width=True)
                st.caption("💰 **Cost vs GWP Scatter** — X-axis is climate impact, Y-axis is input cost. Bottom-left is the sweet spot: cheaper AND greener.")

                # Farmer-friendly inferences (interpret combination B)
                render_inference_section([
                    ("gwp_total", outB[0], {"irrigation": "", "amendments_used": False}),
                    ("ecotox",    outB[3], {"zinc": ZnB}),
                ], key_suffix="t1cmp_conv")
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
                    col1, col2, col3 = st.columns([2, 2, 1.15])
                    with col1:
                        st.metric(f"{cat} — A", f"{formats[i].format(outA[i])} {units[i]}")
                    with col2:
                        st.metric(f"{cat} — B", f"{formats[i].format(outB[i])} {units[i]}")
                    with col3:
                        st.markdown(build_winner_badge(outA[i], outB[i]), unsafe_allow_html=True)

                compare_df = build_comparison_dataframe(outA, outB)
                st.altair_chart(build_comparison_chart(compare_df), use_container_width=True)
                st.caption("📊 **Side-by-side Comparison Bars** — each impact category plotted for Combination A and B together. Shorter bars = lower environmental impact.")

                # Visual gauges – B as % of A
                st.markdown("#### 📉 Combination B vs A (lower is better)")
                st.caption("Each ring shows B's impact as a percentage of A. <100% (green) means B is greener.")
                cmp_g = st.columns(4)
                cmp_data = [
                    ("Global Warming", "🌍", outB[0], outA[0], "vs A", "{:,.0f}"),
                    ("Eutrophication", "💧", outB[1], outA[1], "vs A", "{:.4f}"),
                    ("Acidification", "🌫️", outB[2], outA[2], "vs A", "{:.2f}"),
                    ("Ecotoxicity", "☠️", outB[3], outA[3], "vs A", "{:,.0f}"),
                ]
                for col, (lbl, ic, val, base, u, fmt) in zip(cmp_g, cmp_data):
                    with col:
                        st.markdown(render_value_gauge_html(lbl, val, base, unit=u, icon=ic, value_fmt=fmt),
                                    unsafe_allow_html=True)

                # Radar A vs B
                st.markdown("#### 🕸️ Profile Comparison (Radar)")
                st.plotly_chart(build_radar_chart([
                    ("Combination A", list(outA), "#3b82f6"),
                    ("Combination B", list(outB), "#f97316"),
                ], title="A vs B — Organic Impact Profile"), use_container_width=True)
                st.caption("🕸️ **Radar Comparison** — overlapping polygons of A and B; the smaller polygon dominates the larger across the four impact categories.")

                # Diff arrows
                st.markdown("#### 🔀 Side-by-Side Difference")
                diff_unit_fmt = [("kg CO₂-eq", "{:,.1f}"), ("kg P-eq", "{:.4f}"),
                                 ("kg SO₂-eq", "{:.3f}"), ("CTUe", "{:,.0f}")]
                for i, cat in enumerate(IMPACT_NAMES):
                    u, f = diff_unit_fmt[i]
                    st.markdown(render_diff_arrow_html(outA[i], outB[i], unit=u,
                                label=cat, fmt=f), unsafe_allow_html=True)
                st.markdown(render_diff_arrow_html(costA, costB, unit="₹/ha",
                            label="Amendment Cost", fmt="{:,.0f}"), unsafe_allow_html=True)
                st.caption("🔀 **Difference Arrows** — green arrow means B is lower (better); red means B is higher than A. Magnitude shows the absolute change in each metric.")

                st.markdown("---")
                st.markdown("**Cost vs GWP Trade-off**")
                st.altair_chart(build_cost_vs_impact_scatter(outA, outB, costA, costB,
                                "Comb. A", "Comb. B"), use_container_width=True)
                st.caption("💰 **Cost vs GWP Scatter** — X-axis is climate impact, Y-axis is amendment cost. Bottom-left is the sweet spot: cheaper AND greener.")

                # Farmer-friendly inferences (interpret combination B)
                render_inference_section([
                    ("gwp_total", outB[0], {"irrigation": "", "amendments_used": True}),
                    ("ecotox",    outB[3], {"zinc": 0}),
                ], key_suffix="t1cmp_org")

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

        st.subheader("📊 Environmental Impact at Selected Blend")
        st.caption("Includes upstream fertiliser production + field-level emissions (N₂O, NO₃, NH₃, PO₄)")

        # Summary gauges (visual % change vs Conventional)
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        gwp_reduction = ((blend_out[0] - conv_out[0]) / conv_out[0] * 100) if conv_out[0] != 0 else 0
        cost_delta_pct = ((blend_cost - conv_cost) / conv_cost * 100) if conv_cost != 0 else 0
        acidif_reduction = ((blend_out[2] - conv_out[2]) / conv_out[2] * 100) if conv_out[2] != 0 else 0
        eutroph_reduction = ((blend_out[1] - conv_out[1]) / conv_out[1] * 100) if conv_out[1] != 0 else 0
        with col_kpi1:
            st.markdown(render_gauge_html("Global Warming", gwp_reduction, icon="🌍"), unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(render_gauge_html("Input Cost", cost_delta_pct, icon="💰", inverse=True), unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(render_gauge_html("Acidification", acidif_reduction, icon="🌫️"), unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(render_gauge_html("Eutrophication", eutroph_reduction, icon="💧"), unsafe_allow_html=True)

        # Plotly speedometer gauges (alternative visualisation, animates with slider)
        st.markdown("##### 🏎️ Speedometer View (Plotly)")
        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            st.plotly_chart(build_speedometer(gwp_reduction, "GWP %"), use_container_width=True)
        with sp2:
            st.plotly_chart(build_speedometer(cost_delta_pct, "Cost %", inverse=True), use_container_width=True)
        with sp3:
            st.plotly_chart(build_speedometer(acidif_reduction, "Acid. %"), use_container_width=True)
        with sp4:
            st.plotly_chart(build_speedometer(eutroph_reduction, "Eutro. %"), use_container_width=True)
        st.caption("🏎️ **Speedometer Gauges** — the needle position shows percentage change of each metric vs the conventional baseline. Greener zones (right) mean better outcomes; for cost, left/lower is cheaper.")

        st.markdown(" ")
        scenario_col1, scenario_col2, scenario_col3 = st.columns(3, gap="medium")
        with scenario_col1:
            st.markdown("""
                <div class='ck-scenario-card'>
                    <div class='ck-scenario-head'>
                        <div class='ck-scenario-accent' style='background:#1d4ed8;'></div>
                        <div>
                            <h4 style='margin:0; color:#1d4ed8;'>Conventional</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Current baseline synthetic system.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {conv_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{conv_cost:,.0f}/ha</p>
                </div>
            """.format(conv_gwp=conv_out[0], conv_cost=conv_cost), unsafe_allow_html=True)
        with scenario_col2:
            st.markdown("""
                <div class='ck-scenario-card'>
                    <div class='ck-scenario-head'>
                        <div class='ck-scenario-accent' style='background:#f97316;'></div>
                        <div>
                            <h4 style='margin:0; color:#b45309;'>Blend</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Current organic/conventional transition.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {blend_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{blend_cost:,.0f}/ha</p>
                </div>
            """.format(blend_gwp=blend_out[0], blend_cost=blend_cost), unsafe_allow_html=True)
        with scenario_col3:
            st.markdown("""
                <div class='ck-scenario-card'>
                    <div class='ck-scenario-head'>
                        <div class='ck-scenario-accent' style='background:#16a34a;'></div>
                        <div>
                            <h4 style='margin:0; color:#15803d;'>Organic</h4>
                            <p style='margin:0.35rem 0 0; color:#475569;'>Full organic amendment values.</p>
                        </div>
                    </div>
                    <p style='margin:0.85rem 0 0.25rem; color:#0f172a; font-weight:600;'>GWP: {org_gwp:.1f} kg CO₂-eq</p>
                    <p style='margin:0; color:#0f172a; font-weight:600;'>Cost: ₹{org_cost:,.0f}/ha</p>
                </div>
            """.format(org_gwp=org_out[0], org_cost=org_cost), unsafe_allow_html=True)
        st.caption(f"Snapshots update instantly for the selected blend ({int(alpha * 100)}% organic).")

        blend_df = pd.concat([
            build_impact_dataframe(conv_out, "Conventional"),
            build_impact_dataframe(blend_out, "Blend"),
            build_impact_dataframe(org_out, "Organic")
        ])
        
        st.markdown("---")
        st.subheader("� Detailed Impact by Scenario")
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
        st.subheader("� Impact Breakdown")
        st.altair_chart(build_gradient_impact_chart(blend_df), use_container_width=True)
        st.caption("📊 **Grouped Impact Bars** — every impact category plotted side-by-side for Conventional, Blend, and Organic. Lets you spot which category benefits most from a transition.")

        st.markdown("---")
        st.subheader("💰 Input Cost Analysis")
        cost_delta = blend_cost - conv_cost
        cost_col1, cost_col2, cost_col3 = st.columns(3)
        with cost_col1:
            st.metric("🧪 Conventional Cost", f"₹{conv_cost:,.0f}/ha")
        with cost_col2:
            st.metric("🎚️ Blend Cost", f"₹{blend_cost:,.0f}/ha",
                      delta=f"₹{cost_delta:+,.0f} vs Conv.")
        with cost_col3:
            st.metric("🌿 Organic Cost", f"₹{org_cost:,.0f}/ha")
        st.altair_chart(build_cost_comparison_chart(conv_cost, blend_cost, org_cost, alpha),
                        use_container_width=True)
        st.caption("💰 **Cost Comparison Bars** — per-hectare input cost for the three scenarios. Helps decide if the green premium of organic blend is affordable for your farm.")

        st.markdown("---")
        st.subheader("📈 Impact Trend Across Organic Blend")
        st.altair_chart(build_blend_chart(conv_out, org_out), use_container_width=True)
        st.caption("📈 **Blend Trend Curves** — each line traces how one impact metric changes as the organic share rises from 0% to 100%. Sharp drops indicate strong climate benefit from organic substitution.")

        # Animated marker that moves with the slider
        st.markdown("##### 🎯 Same trend with live slider marker")
        st.altair_chart(build_blend_trend_with_marker(conv_out, org_out, alpha),
                        use_container_width=True)
        st.caption("🎯 **Live Marker Trend** — the dot moves along the curve as you change the slider, so you can see where your current blend lies on the conventional→organic continuum.")

        # Confidence band
        st.markdown("##### 〰️ GWP with ±10% Confidence Band")
        st.altair_chart(build_confidence_band_chart(conv_out, org_out, band_pct=0.10),
                        use_container_width=True)
        st.caption("〰️ **Confidence Band** — shaded area shows a ±10% uncertainty envelope around the predicted GWP, accounting for model variance and on-farm variability.")

        # Streamgraph: composition of impacts across blend
        st.markdown("##### 🌊 Impact Composition Streamgraph")
        st.altair_chart(build_blend_streamgraph(conv_out, org_out),
                        use_container_width=True)
        st.caption("🌊 **Streamgraph** — ribbons show how the four impacts re-balance their relative shares as the blend shifts from fully conventional to fully organic.")

        st.markdown("---")
        st.subheader("🕸️ Three-Scenario Radar")
        st.caption("Conventional vs Blend vs Organic — compare profiles at a glance.")
        st.plotly_chart(build_radar_chart([
            ("Conventional", list(conv_out), "#1d4ed8"),
            (f"Blend ({int(alpha*100)}% Org)", list(blend_out), "#f97316"),
            ("Organic", list(org_out), "#16a34a"),
        ], title="Conv vs Blend vs Organic"), use_container_width=True)
        st.caption("🕸️ **Three-Scenario Radar** — overlapping polygons for Conventional, Blend, and Organic make it easy to spot which scenario dominates each environmental dimension.")

        st.markdown("---")
        st.subheader("🧭 Blend Decision Charts")
        frontier_col, delta_col = st.columns(2, gap="medium")
        with frontier_col:
            st.altair_chart(
                build_blend_frontier_chart(conv_out, org_out, conv_cost, org_cost, alpha),
                use_container_width=True,
            )
            st.caption("🧭 **Cost–GWP Frontier** — each point is one possible blend; your chosen point is highlighted. Curve bends toward bottom-left for win-win blends.")
        with delta_col:
            st.altair_chart(build_impact_delta_chart(conv_out, blend_out), use_container_width=True)
            st.caption("📊 **Impact Delta Bars** — percent change of each impact category for your current blend vs the conventional baseline. Negative bars (green) = improvement.")

        # Pareto frontier with iso-trade-off lines
        st.markdown("##### ⚖️ Pareto Frontier with Iso-trade-off Lines")
        st.altair_chart(
            build_pareto_with_isolines(conv_out, org_out, conv_cost, org_cost, alpha),
            use_container_width=True,
        )
        st.caption("⚖️ **Pareto with Iso-lines** — dashed lines connect blends with equal cost-to-GWP trade-off. Moving along a line costs nothing extra; crossing onto a lower line means a strictly better deal.")

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

        # Farmer-friendly inferences for the blend
        render_inference_section([
            ("gwp_total",      blend_out[0], {"irrigation": "", "amendments_used": alpha > 0}),
            ("blend_savings",  None, {
                "gwp_saved":  conv_out[0] - blend_out[0],
                "cost_delta": blend_cost - conv_cost,
                "alpha":      alpha,
            }),
        ], key_suffix="t2")
    else:
        st.error("❌ Required models not fully available. Need both conventional and organic models for gradient analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CARBON CREDIT POTENTIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Carbon Credit Potential (CCTS)")
    st.caption("Soil-carbon credits only (1 ha basis), shown in t CO2-eq/ha")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        cc_manure = st.number_input(
            "Farm Yard Manure (kg/ha)",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key="cc_manure",
        )
    with col2:
        cc_compost = st.number_input(
            "Compost (kg/ha)",
            min_value=0.0,
            value=1500.0,
            step=50.0,
            key="cc_compost",
        )

    cc_warn = validate_organic(cc_manure, cc_compost)
    if cc_warn:
        st.warning("⚠️ Inputs outside recommended organic ranges:\n\n" + "\n".join(f"- {w}" for w in cc_warn))

    buffer_pct = st.slider(
        "Permanence Buffer (%)",
        min_value=10,
        max_value=30,
        value=20,
        step=1,
        help="Applied to account for uncertainty and reversal risk.",
        key="cc_buffer",
    )

    result = calculate_soc_credits(cc_manure, cc_compost, buffer_pct)

    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.metric("🌱 Soil Carbon Credits", f"{result['credits_tco2']:.3f} t CO2-eq/ha")
    with kpi2:
        st.metric("📦 SOC Stored (Before Buffer)", f"{result['soc_before_buffer_tco2']:.3f} t CO2-eq/ha")

    # Visual gauges – credits & value vs upper-range benchmark (max manure & compost)
    bench = calculate_soc_credits(15000, 2000, buffer_pct)
    st.markdown("#### 📉 Credit Potential vs Upper-Range Benchmark")
    st.caption("Each ring shows your potential relative to the maximum recommended FYM + compost dose.")
    cg1, cg2, cg3, cg4 = st.columns(4)
    with cg1:
        st.markdown(render_value_gauge_html("Credits Earned", result["credits_tco2"], bench["credits_tco2"],
                                           unit="t CO₂/ha", icon="🌱", value_fmt="{:.3f}"), unsafe_allow_html=True)
    with cg2:
        st.markdown(render_value_gauge_html("SOC Stored", result["soc_before_buffer_tco2"], bench["soc_before_buffer_tco2"],
                                           unit="t CO₂/ha", icon="📦", value_fmt="{:.3f}"), unsafe_allow_html=True)
    with cg3:
        st.markdown(render_value_gauge_html("FYM Contribution", result["fym_credits_tco2"], bench["fym_credits_tco2"],
                                           unit="t CO₂/ha", icon="🐄", value_fmt="{:.3f}"), unsafe_allow_html=True)
    with cg4:
        st.markdown(render_value_gauge_html("Compost Contribution", result["compost_credits_tco2"], bench["compost_credits_tco2"],
                                           unit="t CO₂/ha", icon="🍂", value_fmt="{:.3f}"), unsafe_allow_html=True)

    st.markdown("##### Estimated Value (per ha, CCTS)")
    st.info(
        f"₹{result['value_low_inr']:,.0f} - ₹{result['value_high_inr']:,.0f}\n\n"
        "Based on ₹600-₹900 per t CO2-eq."
    )
    st.caption(
        "Credits are based on stabilized soil carbon from FYM and compost inputs, adjusted using the permanence buffer."
    )

    st.markdown("---")
    st.markdown("#### Visual Insights")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.altair_chart(build_ccts_source_chart(result), use_container_width=True)
        st.caption("🌿 **Carbon Source Breakdown** — split of soil-carbon sequestration credit between FYM and Compost amendments. Helps you see which input is doing the heavy lifting on credits.")
    with chart_col2:
        st.altair_chart(build_ccts_buffer_chart(result), use_container_width=True)
        st.caption("🛡️ **Permanence Buffer** — portion of gross sequestration retained as a non-permanence reserve (per CCTS rules) vs net credits actually issuable.")

    st.altair_chart(build_ccts_value_chart(result), use_container_width=True)
    st.caption("💵 **Credit Monetary Value Range** — estimated rupee value of your net credits at the low (₹600/tCO₂) and high (₹900/tCO₂) market price brackets.")

    # Farmer-friendly inferences
    render_inference_section([
        ("credits", result['credits_tco2'], {}),
    ], key_suffix="t3")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FIELD EMISSION CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔥 Methane (CH₄)", f"{emissions['CH4']:,} kg/ha/season")
            st.metric("⚡ Nitrous Oxide (N₂O)", f"{emissions['N2O']:,} kg/ha/season")
        with col2:
            st.metric("💧 Phosphate (PO₄)", f"{emissions['PO4']:,} kg/ha/season")
            st.metric("💦 Nitrate (NO₃)", f"{emissions['NO3']:,} kg/ha/season")
        with col3:
            st.metric("🌬️ Ammonia (NH₃)", f"{emissions['NH3']:,} kg/ha/season")

        # Visual gauges – emissions vs Fully-Flooded upper-bound benchmark (max N=150, max P=60)
        bench = compute_field_emissions(150, 60, "Fully Flooded", "None", "None")
        st.markdown("#### 📉 Emissions vs Fully-Flooded Upper-Bound Benchmark")
        st.caption("Each ring shows your emission relative to the maximum-input, fully-flooded baseline.")
        g1, g2, g3, g4, g5 = st.columns(5)
        gauges = [
            ("CH₄", "🔥", emissions["CH4"], bench["CH4"], "kg/ha", "{:,.1f}"),
            ("N₂O", "⚡", emissions["N2O"], bench["N2O"], "kg/ha", "{:,.3f}"),
            ("NO₃", "💦", emissions["NO3"], bench["NO3"], "kg/ha", "{:,.2f}"),
            ("NH₃", "🌬️", emissions["NH3"], bench["NH3"], "kg/ha", "{:,.2f}"),
            ("PO₄", "💧", emissions["PO4"], bench["PO4"], "kg/ha", "{:,.2f}"),
        ]
        for col, (lbl, ic, val, mx, u, fmt) in zip([g1, g2, g3, g4, g5], gauges):
            with col:
                st.markdown(render_value_gauge_html(lbl, val, mx, unit=u, icon=ic, value_fmt=fmt),
                            unsafe_allow_html=True)

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
            st.caption("📊 **Emission Magnitude Bars** — absolute kg/ha emission of gas or leached this season. Useful to spot the largest single source.")
        
        with col_chart2:
            st.markdown("**Emission Share (%)**")
            st.altair_chart(build_emission_pie_chart(emissions), use_container_width=True)
            st.caption("🥧 **Emission Share Pie** — same emissions expressed as percentages so you instantly see which one dominates the field-level footprint.")

        st.markdown("---")
        st.subheader("🔗 Inputs → Emissions → Impacts (Sankey)")
        st.caption("Follow how each fertiliser/amendment flows into specific emissions and ultimately into climate, eutrophication and acidification impacts.")
        st.plotly_chart(build_sankey_chart(emissions, synthetic_n, synthetic_p, amendment_1, amendment_2),
                        use_container_width=True)
        st.caption("🔗 **Sankey Flow Diagram** — ribbon thickness shows mass flow from each fertiliser/amendment input through specific emissions and onward to the LCA impact categories they cause.")

        st.markdown("---")
        st.subheader("🌳 Emission Treemap")
        st.caption("Hierarchical view: Climate / Eutrophication / Acidification → individual emissions, sized by magnitude.")
        st.plotly_chart(build_emission_treemap(emissions), use_container_width=True)
        st.caption("🌳 **Treemap** — each rectangle area is proportional to the emission's mass. Quickly reveals whether climate, eutrophication, or acidification dominates your footprint.")

        # Farmer-friendly inferences for each emission
        render_inference_section([
            ("ch4", emissions["CH4"], {"irrigation": irrigation}),
            ("n2o", emissions["N2O"], {"synthetic_n": synthetic_n}),
            ("no3", emissions["NO3"], {}),
            ("nh3", emissions["NH3"], {}),
            ("po4", emissions["PO4"], {}),
        ], key_suffix="t4")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL VALIDATION & INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
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
    st.subheader("🔥 Per-kg Footprint Heatmap")
    st.caption("Color encodes log₁₀ of the impact value. Reveals at a glance where each input dominates.")
    st.altair_chart(build_impact_heatmap(), use_container_width=True)
    st.caption("🔥 **Per-kg Footprint Heatmap** — rows are inputs (N, P, K, Zn), columns are LCA impact categories. Darker cells = higher impact per kg. Zinc's ecotoxicity row is the standout hotspot.")

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
        - `Organic evaluate.py` — Organic model evaluation
        - `Organic LCA script.py` — Organic OpenLCA data generation
        - `predict.py` — Batch prediction utilities
        - `visualise.py` — Visualization and analysis tools
        """)

st.markdown("---")
st.caption("© 2026 CLIMATEKRISHI AI | Data from ICAR-IIRR & KVK Medak")

import base64 as _b64
def _img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return _b64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

_logo_files = [
    ("media/footer_logos/iith_logo.jpeg", "IIT Hyderabad"),
    ("media/footer_logos/278969668_371139835029083_611259122675019803_n.jpg", "ICAR-IIRR"),
    ("media/footer_logos/about-removebg-preview.png", "Krishi Vigyan Kendra"),
    ("media/footer_logos/olcalogo.png", "openLCA"),
]
_logo_html = ""
for _p, _alt in _logo_files:
    if os.path.exists(_p):
        _ext = "png" if _p.lower().endswith(".png") else "jpeg"
        _b = _img_to_b64(_p)
        _logo_html += (
            f"<div style='flex:1; display:flex; align-items:center; justify-content:center; "
            f"height:110px; padding:0.4rem;'>"
            f"<img src='data:image/{_ext};base64,{_b}' alt='{_alt}' "
            f"style='max-height:100%; max-width:100%; object-fit:contain;'/>"
            f"</div>"
        )

st.markdown(
    f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                gap:1.5rem; flex-wrap:nowrap; padding:0.75rem 0 1.25rem;'>
        {_logo_html}
    </div>
    """,
    unsafe_allow_html=True,
)
