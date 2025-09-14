import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Space Textile Selector", layout="wide")

# -----------------------------
# Defaults & constants
# -----------------------------
COMPACT_DEFAULT = True  # good for mobile
AUTO_CSV_PATH = "/space_textile_dataset_AI_ready_ULTRA_v3.csv"  # auto-load if present

SCHEMA_ORDER = [
    "Material_ID","Material_Name","Material_Class","Base_Fiber","Coating_Type","Weave_Pattern",
    "Test_Environment","Areal_Density_gm2","Thickness_mm",
    "Outgassing_TML_percent","Outgassing_CVCM_percent","Flammability_Rating",
    "Thermal_Conductivity_Through_WmK","Thermal_Conductivity_InPlane_WmK",
    "Tensile_Strength_MPa","Tensile_Modulus_GPa","Bulk_Density_kgm3",
    "Max_Service_Temp_C","k_-100C_WmK","k_0C_WmK","k_23C_WmK","k_150C_WmK","k_200C_WmK",
    "Notes","Data_Source"
]

NUMERIC_COLS_DEFAULT = [
    "Areal_Density_gm2","Thickness_mm",
    "Outgassing_TML_percent","Outgassing_CVCM_percent",
    "Thermal_Conductivity_Through_WmK","Thermal_Conductivity_InPlane_WmK",
    "Tensile_Strength_MPa","Tensile_Modulus_GPa","Bulk_Density_kgm3",
    "Max_Service_Temp_C","k_-100C_WmK","k_0C_WmK","k_23C_WmK","k_150C_WmK","k_200C_WmK"
]

# Subset recommended for compact mobile view
MOBILE_CORE_COLS = [
    "Material_Name","Material_Class","Base_Fiber","Weave_Pattern","Test_Environment",
    "Thermal_Conductivity_Through_WmK","Tensile_Strength_MPa","Bulk_Density_kgm3",
    "Outgassing_TML_percent","Outgassing_CVCM_percent","Max_Service_Temp_C"
]

RANK_METHODS = {
    "Distance to targets (L2)": "l2",
    "Weighted absolute error (L1)": "l1",
    "Weighted desirability score (0-1)": "desirability"
}

# -----------------------------
# Data loading helpers
# -----------------------------
def _load_sample():
    # Try to load the sandbox path if present
    try:
        df = pd.read_csv(AUTO_CSV_PATH)
        if len(df) > 0:
            st.sidebar.success("Loaded default ULTRA v3 dataset from sandbox.")
            return df
    except Exception:
        pass
    # Minimal fallback dataset if no CSV provided and sandbox path missing
    data = [
        {"Material_ID":"HP-KEVLAR29-PL-200","Material_Name":"Kevlar 29 fabric","Material_Class":"Textile","Base_Fiber":"Kevlar 29","Coating_Type":"","Weave_Pattern":"Plain","Test_Environment":"Air","Areal_Density_gm2":150,"Thickness_mm":0.2,"Outgassing_TML_percent":np.nan,"Outgassing_CVCM_percent":np.nan,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.0812,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":3000,"Tensile_Modulus_GPa":70,"Bulk_Density_kgm3":1440,"Max_Service_Temp_C":177,"k_23C_WmK":0.0812,"Notes":"Sample row","Data_Source":"DuPont/NIST"},
        {"Material_ID":"AR-NOMEX410-PA-500","Material_Name":"Nomex 410 paper","Material_Class":"Textile","Base_Fiber":"Nomex 410","Coating_Type":"","Weave_Pattern":"Paper","Test_Environment":"Air","Areal_Density_gm2":200,"Thickness_mm":0.5,"Outgassing_TML_percent":2.44,"Outgassing_CVCM_percent":0.05,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.139,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1100,"Max_Service_Temp_C":220,"k_150C_WmK":0.139,"Notes":"Sample row","Data_Source":"DuPont/NASA"},
        {"Material_ID":"FILM-MYLAR-AL-12","Material_Name":"Mylar film, aluminized","Material_Class":"Film","Base_Fiber":"Mylar","Coating_Type":"Aluminized","Weave_Pattern":"Film","Test_Environment":"Vacuum","Areal_Density_gm2":np.nan,"Thickness_mm":0.012,"Outgassing_TML_percent":0.26,"Outgassing_CVCM_percent":0.0,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.155,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1390,"Max_Service_Temp_C":150,"k_23C_WmK":0.155,"Notes":"Sample row","Data_Source":"NASA"},
    ]
    st.sidebar.warning("Default ULTRA v3 not found in sandbox — using small built-in sample.")
    return pd.DataFrame(data)


def load_data():
    st.sidebar.header("1) Load your dataset")
    # Auto-load default if available
    try:
        df_auto = pd.read_csv(AUTO_CSV_PATH)
        if len(df_auto) > 0:
            st.sidebar.info("Auto-loaded: ULTRA v3 (sandbox)")
            default_df = df_auto
        else:
            default_df = None
    except Exception:
        default_df = None

    upl = st.sidebar.file_uploader("Upload CSV (AI-ready schema)", type=["csv"])
    if upl is not None:
        try:
            df = pd.read_csv(upl)
            st.sidebar.success("Uploaded CSV loaded.")
            return df
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            return _load_sample()
    else:
        if default_df is not None:
            return default_df
        else:
            return _load_sample()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS_DEFAULT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def desirability(x, target, higher_is_better=True, tol=None):
    if pd.isna(x) or pd.isna(target):
        return np.nan
    if tol is None:
        tol = 0.2 * abs(target) if target != 0 else 0.2
    if higher_is_better:
        # 0 if far below, 1 if >= target, smooth ramp
        return np.clip((x - (target - tol)) / (tol), 0, 1)
    else:
        # 1 if <= target, decays as it exceeds target
        return np.clip((target + tol - x) / (tol), 0, 1)


def rank(df: pd.DataFrame, selections: list, targets: dict, weights: dict, senses: dict, method: str) -> pd.DataFrame:
    scored = df.copy()
    for col in selections:
        t = targets.get(col, np.nan)
        w = weights.get(col, 1.0)
        sense = senses.get(col, "minimize")  # or "maximize"
        if col not in scored.columns:
            scored[col] = np.nan
        x = scored[col].astype(float)
        if method == "l2":
            err = (x - t)**2
            if sense == "maximize":
                err = (np.maximum(0, t - x))**2  # penalize only if below target
            scored[f"score__{col}"] = -w * err
        elif method == "l1":
            err = np.abs(x - t)
            if sense == "maximize":
                err = np.maximum(0, t - x)
            scored[f"score__{col}"] = -w * err
        else:  # desirability
            hib = (sense == "maximize")
            d = x.apply(lambda v: desirability(v, t, higher_is_better=hib))
            scored[f"score__{col}"] = w * d

    score_cols = [c for c in scored.columns if c.startswith("score__")]
    if method in ("l1","l2"):
        scored["Score"] = scored[score_cols].sum(axis=1, skipna=True)
    else:
        filled = scored[score_cols].fillna(0)
        scored["Score"] = filled.sum(axis=1) / np.maximum((~scored[score_cols].isna()).sum(axis=1), 1)

    scored = scored.sort_values("Score", ascending=False)
    return scored

# -----------------------------
# UI
# -----------------------------
st.title("🛰️ Space Textile & Material Selector")
st.caption("Rank materials by your target properties. Upload the AI-ready CSV or use the default if present.")

# Load
df = load_data()
df = coerce_numeric(df)

with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(25), use_container_width=True)

# Filters
st.sidebar.header("2) Filter (optional)")
col_filter1, col_filter2 = st.sidebar.columns(2)

classes = sorted(df["Material_Class"].dropna().unique().tolist()) if "Material_Class" in df.columns else []
envs = sorted(df["Test_Environment"].dropna().unique().tolist()) if "Test_Environment" in df.columns else []

selected_classes = col_filter1.multiselect("Material_Class", classes, default=classes)
selected_envs = col_filter2.multiselect("Test_Environment", envs, default=envs)

mask = pd.Series([True] * len(df))
if "Material_Class" in df.columns and selected_classes:
    mask &= df["Material_Class"].isin(selected_classes)
if "Test_Environment" in df.columns and selected_envs:
    mask &= df["Test_Environment"].isin(selected_envs)

df_f = df[mask].copy()

# Mobile compact mode toggle
compact = st.sidebar.checkbox("Mobile compact mode", value=COMPACT_DEFAULT)

# Property selection
st.sidebar.header("3) Select properties & targets")
num_cols_available = [c for c in NUMERIC_COLS_DEFAULT if c in df_f.columns]
props = st.sidebar.multiselect("Properties of interest", num_cols_available, default=["Thermal_Conductivity_Through_WmK","Bulk_Density_kgm3"]) 

# Targets/weights/sense per property
targets, weights, senses = {}, {}, {}
for p in props:
    with st.sidebar.expander(f"➡️ {p}", expanded=False):
        default_target = float(np.nanmedian(df_f[p])) if p in df_f.columns else 0.0
        t = st.number_input(f"Target for {p}", value=float(default_target), key=f"t_{p}")
        w = st.slider(f"Weight for {p}", 0.0, 5.0, 1.0, 0.1, key=f"w_{p}")
        s = st.selectbox(f"Optimization sense for {p}", ["minimize","maximize"], index=0, key=f"s_{p}")
        targets[p] = t
        weights[p] = w
        senses[p] = s

# Method
st.sidebar.header("4) Ranking method")
method_label = st.sidebar.selectbox("Method", list(RANK_METHODS.keys()), index=0)
method = RANK_METHODS[method_label]

# Run ranking
st.sidebar.header("5) Run")
if st.sidebar.button("Generate ranked list", type="primary"):
    if len(props) == 0:
        st.warning("Please select at least one property to rank.")
    else:
        scored = rank(df_f, props, targets, weights, senses, method)
        # Choose columns to display
        show_cols = [c for c in SCHEMA_ORDER if c in scored.columns]
        show_cols = ["Score"] + [c for c in show_cols if c != "Score"]

        st.subheader("Ranked materials")
        if compact:
            # Show a condensed table
            compact_cols = ["Score"] + [c for c in MOBILE_CORE_COLS if c in scored.columns]
            st.dataframe(scored[compact_cols].head(200), use_container_width=True)
            # Card-style view for top 25
            st.markdown("---")
            st.markdown("### Top 25 (cards)")
            top = scored.head(25)
            for _, r in top.iterrows():
                st.markdown(f"**{r.get('Material_Name','(Unnamed)')}**  ")
                st.caption(f"Score: {r.get('Score',np.nan):.3f} | Class: {r.get('Material_Class','')} | Env: {r.get('Test_Environment','')}")
                cols = st.columns(3)
                cols[0].markdown(f"**k_through**: {r.get('Thermal_Conductivity_Through_WmK',np.nan)} W/m·K  ")
                cols[0].markdown(f"**Density**: {r.get('Bulk_Density_kgm3',np.nan)} kg/m³  ")
                cols[1].markdown(f"**Tensile**: {r.get('Tensile_Strength_MPa',np.nan)} MPa  ")
                cols[1].markdown(f"**Modulus**: {r.get('Tensile_Modulus_GPa',np.nan)} GPa  ")
                cols[2].markdown(f"**TML/CVCM**: {r.get('Outgassing_TML_percent',np.nan)}% / {r.get('Outgassing_CVCM_percent',np.nan)}%  ")
                st.markdown("---")
        else:
            st.dataframe(scored[show_cols].head(200), use_container_width=True)

        # Download CSV
        out = BytesIO()
        scored.to_csv(out, index=False)
        st.download_button("Download ranked results (CSV)", data=out.getvalue(), file_name="ranked_materials.csv", mime="text/csv")
    

with st.expander("📘 Tips: Properties, Sources & How to Adjust", expanded=False):
    st.markdown(
        """
        ### Properties (with typical sources)
        **Identity & categorization**
        - **Material_ID** – unique row ID. 
        - **Material_Name** – readable name (e.g., “Kevlar 29 fabric”).  
          *Source:* Vendor datasheets / standards names.
        - **Material_Class** – category (Textile, Film, Ceramic Textile, Carbon Textile, Insulation, Polymer/Insulation).  
          *Source:* Vendor literature; textile engineering handbooks.
        - **Base_Fiber** – base material (Kevlar 29, Nomex 410, E-glass, PET, …).  
          *Source:* Vendor datasheets (DuPont, 3M, AGY, DSM, Toyobo, Kuraray, etc.).
        - **Coating_Type** – surface treatment (e.g., Aluminized, PTFE).  
          *Source:* Vendor process notes; MLI blanket specs (NASA/ESA).
        - **Weave_Pattern** – Plain, Twill, Satin, Basket, Ripstop, UD, Nonwoven, Paper, Knitted, Film.  
          *Source:* Textile handbooks; vendor fabric catalogs.
        - **Test_Environment** – Air or Vacuum (where data applies).  
          *Source:* Test reports; NASA/ESA materials & processes (M&P) docs.

        **Geometric / physical**
        - **Areal_Density_gm2** – mass/area (g/m²).  
          *Source:* Vendor datasheets; **ASTM D3776 / ISO 3801**.
        - **Thickness_mm** – thickness (mm).  
          *Source:* Vendor datasheets; **ASTM D1777** (textiles); film gauges from datasheets.
        - **Bulk_Density_kgm3** – material density (kg/m³).  
          *Source:* Vendor datasheets; **MatWeb**; materials handbooks.

        **Thermal**
        - **Thermal_Conductivity_Through_WmK** – through-thickness k (W/m·K).  
          *Source:* Vendor/literature; **Thermtest knowledge base**; NASA thermal reports; polymer datasheets (e.g., Kapton, PET, PTFE); **3M Nextel** data.
        - **Thermal_Conductivity_InPlane_WmK** – in-plane k (anisotropic materials).  
          *Source:* Carbon fiber (PAN/pitch) vendor data; composites literature (K1100-class, PAN CF).
        - **k_-100C_WmK**, **k_0C_WmK**, **k_23C_WmK**, **k_150C_WmK**, **k_200C_WmK** – k at specific temperatures.  
          *Source:* Temperature-dependent curves in vendor datasheets; **NIST**/NASA publications for aramids & films; MLI effective k from NASA.

        **Mechanical**
        - **Tensile_Strength_MPa** – tensile strength (MPa).  
          *Source:* Vendor datasheets; **ASTM D5035/D5034** (textile tensile), **ASTM D3822** (single fibers); film standards.
        - **Tensile_Modulus_GPa** – modulus (GPa).  
          *Source:* Vendor datasheets; composites handbooks; ASTM methods above.

        **Stability / compliance**
        - **Outgassing_TML_percent** – Total Mass Loss % (NASA outgassing).  
          *Source:* **NASA GSFC Outgassing Database**; **ASTM E595** test method.
        - **Outgassing_CVCM_percent** – Condensables %.  
          *Source:* **NASA GSFC Outgassing Database**; **ASTM E595**.
        - **Flammability_Rating** – Pass / Fail / Self-extinguishing / Non-flammable.  
          *Source:* **NASA-STD-6001 (flammability)**; **UL 94** for many polymers/films.
        - **Max_Service_Temp_C** – recommended max service temperature (°C).  
          *Source:* Vendor datasheets (DuPont Nomex/Kevlar/Kapton; 3M Nextel; Aspen Aerogels; DSM Dyneema; Toyobo Zylon; Kuraray Vectran).

        ### For each selected property, you can adjust in the sidebar
        - **Target** – quantitative goal.
        - **Weight** – importance to your ranking (0–5). Higher weight = stronger influence on rank.
        - **Sense** – **minimize** or **maximize** (direction of optimization).

"""
    )
