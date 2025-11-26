import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO  

st.set_page_config(page_title="Space Textile Selector", layout="wide")

# -----------------------------
# Defaults & constants
# -----------------------------
COMPACT_DEFAULT = True  # good for mobile
AUTO_CSV_PATH = "data/space_textile_dataset_AI_ready_ULTRA_v3.csv"  # path inside your GitHub repo  # auto-load if present

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
    # Minimal fallback dataset if nothing is provided
    data = [
        {"Material_ID":"HP-KEVLAR29-PL-200","Material_Name":"Kevlar 29 fabric","Material_Class":"Textile","Base_Fiber":"Kevlar 29","Coating_Type":"","Weave_Pattern":"Plain","Test_Environment":"Air","Areal_Density_gm2":150,"Thickness_mm":0.2,"Outgassing_TML_percent":np.nan,"Outgassing_CVCM_percent":np.nan,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.0812,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":3000,"Tensile_Modulus_GPa":70,"Bulk_Density_kgm3":1440,"Max_Service_Temp_C":177,"k_23C_WmK":0.0812,"Notes":"Sample row","Data_Source":"DuPont/NIST"},
        {"Material_ID":"AR-NOMEX410-PA-500","Material_Name":"Nomex 410 paper","Material_Class":"Textile","Base_Fiber":"Nomex 410","Coating_Type":"","Weave_Pattern":"Paper","Test_Environment":"Air","Areal_Density_gm2":200,"Thickness_mm":0.5,"Outgassing_TML_percent":2.44,"Outgassing_CVCM_percent":0.05,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.139,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1100,"Max_Service_Temp_C":220,"k_150C_WmK":0.139,"Notes":"Sample row","Data_Source":"DuPont/NASA"},
        {"Material_ID":"FILM-MYLAR-AL-12","Material_Name":"Mylar film, aluminized","Material_Class":"Film","Base_Fiber":"Mylar","Coating_Type":"Aluminized","Weave_Pattern":"Film","Test_Environment":"Vacuum","Areal_Density_gm2":np.nan,"Thickness_mm":0.012,"Outgassing_TML_percent":0.26,"Outgassing_CVCM_percent":0.0,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.155,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1390,"Max_Service_Temp_C":150,"k_23C_WmK":0.155,"Notes":"Sample row","Data_Source":"NASA"},
    ]
    return pd.DataFrame(data)



@st.cache_data(show_spinner=False)
def _load_csv_from_url(url: str) -> pd.DataFrame:
    import requests, io
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(show_spinner=False)
def _load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def _load_csv_from_url(url: str) -> pd.DataFrame:
    import requests, io
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(show_spinner=False)
def _load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_data():
    st.sidebar.header("1) Load your dataset")

    source = st.sidebar.radio(
        "CSV source",
        ["Upload", "GitHub RAW URL", "Bundled file"],
        index=2,
        help="Choose where to load the dataset from"
    )

    # Option 1: upload
    if source == "Upload":
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
            st.sidebar.info("No file uploaded yet. Using small built-in sample.")
            return _load_sample()

    # Option 2: GitHub RAW URL
    elif source == "GitHub RAW URL":
        default_url = st.session_state.get("csv_url", "")
        url = st.sidebar.text_input(
            "Paste RAW CSV URL",
            value=default_url,
            placeholder="https://raw.githubusercontent.com/<user>/<repo>/main/data/your_file.csv",
        )
        st.session_state["csv_url"] = url
        if url:
            try:
                df = _load_csv_from_url(url)
                st.sidebar.success("Loaded from GitHub RAW URL")
                return df
            except Exception as e:
                st.sidebar.error(f"Failed to load from URL: {e}")
                return _load_sample()
        else:
            st.sidebar.warning("Provide a RAW URL or switch source.")
            return _load_sample()

    # Option 3: bundled file in the repo
    else:  # "Bundled file"
        try:
            df = _load_csv_from_path(AUTO_CSV_PATH)
            st.sidebar.success(f"Loaded bundled file: {AUTO_CSV_PATH}")
            return df
        except Exception as e:
            st.sidebar.error(f"Bundled file not found at {AUTO_CSV_PATH}: {e}")
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

# -----------------------------
# Stack builder (beta)
# -----------------------------
st.markdown("---")
st.header("🧱 Stack builder (beta)")
st.caption("Combine multiple layers (outside → inside) and compute effective properties, then rank stacks with the same targets/weights.")

# Helpers for stack calculations
def _compute_layer_areal_density(row, thickness_mm: float | None) -> float | float:
    """Return areal density in g/m² using either row value or ρ·t if needed."""
    ad = row.get("Areal_Density_gm2", np.nan)
    if not pd.isna(ad):
        return float(ad)
    rho = row.get("Bulk_Density_kgm3", np.nan)
    t = thickness_mm if thickness_mm is not None else row.get("Thickness_mm", np.nan)
    if pd.isna(rho) or pd.isna(t):
        return np.nan
    # rho [kg/m³], t [mm] → convert t to m, then kg/m² → g/m²
    t_m = float(t) / 1000.0
    return float(rho) * t_m * 1000.0


def _compute_stack_from_layers(df_source: pd.DataFrame, layers: list[dict]) -> dict:
    """Compute stack properties from a list of layer dicts: {id, thickness_mm}.
    df_source must have unique Material_ID.
    """
    if not layers:
        return {}

    # Collect layer rows
    rows = []
    for lyr in layers:
        mat_id = lyr["id"]
        override_t = lyr.get("thickness_mm", 0.0)
        sub = df_source[df_source["Material_ID"] == mat_id]
        if sub.empty:
            continue
        base = sub.iloc[0]
        t = float(override_t) if override_t and override_t > 0 else float(base.get("Thickness_mm", np.nan))
        rows.append((base, t))
    if not rows:
        return {}

    # Aggregate basics
    t_tot = 0.0
    ad_tot = 0.0
    rho_num = 0.0
    rho_den = 0.0

    # Effective through-thickness k
    R_perp = 0.0
    k_perp_missing = False

    # In-plane bounds
    num_k_par = 0.0
    denom_k_par = 0.0
    k_par_missing = False

    # Outgassing, treated mass-weighted by areal density
    tml_num = 0.0
    cvc_num = 0.0
    ad_mass_den = 0.0

    # Max service temp, flammability
    maxT_list = []
    flam_list = []

    # Tensile ROM & conservative
    E_num = 0.0
    sig_rom_num = 0.0
    sig_lb_list = []

    for base, t in rows:
        if pd.isna(t) or t <= 0:
            continue
        t_tot += t

        rho = base.get("Bulk_Density_kgm3", np.nan)
        if not pd.isna(rho):
            rho_num += float(rho) * t
            rho_den += t

        # areal density
        ad = _compute_layer_areal_density(base, t)
        if not pd.isna(ad):
            ad_tot += ad
            ad_mass_den += ad

        # through-thickness k
        k_perp = base.get("Thermal_Conductivity_Through_WmK", np.nan)
        if pd.isna(k_perp) or k_perp <= 0:
            k_perp_missing = True
        else:
            # convert t from mm to m in resistance
            R_perp += (t / 1000.0) / float(k_perp)

        # in-plane k bounds
        k_par = base.get("Thermal_Conductivity_InPlane_WmK", np.nan)
        if pd.isna(k_par) or k_par <= 0:
            k_par_missing = True
        else:
            num_k_par += float(k_par) * t
            denom_k_par += t / float(k_par)

        # outgassing
        tml = base.get("Outgassing_TML_percent", np.nan)
        cvc = base.get("Outgassing_CVCM_percent", np.nan)
        if not pd.isna(ad) and not pd.isna(tml):
            tml_num += ad * float(tml)
        if not pd.isna(ad) and not pd.isna(cvc):
            cvc_num += ad * float(cvc)

        # max service temp
        Tmax = base.get("Max_Service_Temp_C", np.nan)
        if not pd.isna(Tmax):
            maxT_list.append(float(Tmax))

        # flammability rating
        flam = base.get("Flammability_Rating", None)
        if isinstance(flam, str) and flam.strip():
            flam_list.append(flam.strip())

        # mechanical
        E = base.get("Tensile_Modulus_GPa", np.nan)
        sig = base.get("Tensile_Strength_MPa", np.nan)
        if not pd.isna(E):
            E_num += float(E) * t
        if not pd.isna(sig):
            sig_rom_num += float(sig) * t
            sig_lb_list.append(float(sig))

    if t_tot <= 0:
        return {}

    # Effective properties
    rho_eff = rho_num / rho_den if rho_den > 0 else np.nan
    k_perp_eff = np.nan
    if not k_perp_missing and R_perp > 0:
        k_perp_eff = (t_tot / 1000.0) / R_perp

    k_par_ub = np.nan
    k_par_lb = np.nan
    if not k_par_missing and t_tot > 0 and denom_k_par > 0:
        k_par_ub = num_k_par / t_tot
        k_par_lb = t_tot / denom_k_par

    tml_stack = np.nan
    cvc_stack = np.nan
    if ad_mass_den > 0:
        tml_stack = tml_num / ad_mass_den
        cvc_stack = cvc_num / ad_mass_den

    Tmax_stack = min(maxT_list) if maxT_list else np.nan

    # Flammability: worst-case
    flam_stack = None
    if flam_list:
        # simple ordering: Fail > Self-extinguishing > Pass/Non-flammable
        joined = " ".join([f.upper() for f in flam_list])
        if "FAIL" in joined:
            flam_stack = "Fail (worst layer)"
        elif "SELF" in joined:
            flam_stack = "Self-extinguishing (limited by layer)"
        elif any(x in joined for x in ["PASS", "NON-FLAMMABLE"]):
            flam_stack = "Pass (all layers)"
        else:
            flam_stack = flam_list[0]

    E_eff = E_num / t_tot if t_tot > 0 and E_num > 0 else np.nan
    sig_rom = sig_rom_num / t_tot if t_tot > 0 and sig_rom_num > 0 else np.nan
    sig_lb = min(sig_lb_list) if sig_lb_list else np.nan

    # Build a single-row dict using the main schema
    names = [r[0].get("Material_Name", "") for r in rows]
    mat_name = "STACK: " + " / ".join(names)
    mat_id = "STACK-" + "-".join([r[0].get("Material_ID", "?") for r in rows])[:60]

    stack_row = {
        "Material_ID": mat_id,
        "Material_Name": mat_name,
        "Material_Class": "Stack",
        "Base_Fiber": "Mixed",
        "Coating_Type": "Mixed",
        "Weave_Pattern": "Stack",
        "Test_Environment": rows[0][0].get("Test_Environment", "Mixed"),
        "Areal_Density_gm2": ad_tot if ad_tot > 0 else np.nan,
        "Thickness_mm": t_tot,
        "Outgassing_TML_percent": tml_stack,
        "Outgassing_CVCM_percent": cvc_stack,
        "Flammability_Rating": flam_stack,
        "Thermal_Conductivity_Through_WmK": k_perp_eff,
        "Thermal_Conductivity_InPlane_WmK": k_par_ub,
        "Tensile_Strength_MPa": sig_rom,
        "Tensile_Modulus_GPa": E_eff,
        "Bulk_Density_kgm3": rho_eff,
        "Max_Service_Temp_C": Tmax_stack,
        # keep other thermal-temperature columns blank/NaN for now
        "k_-100C_WmK": np.nan,
        "k_0C_WmK": np.nan,
        "k_23C_WmK": k_perp_eff,
        "k_150C_WmK": np.nan,
        "k_200C_WmK": np.nan,
        "Notes": "Synthetic stack from layers: " + ", ".join(names),
        "Data_Source": "Rule-based stack combination"
    }
    return stack_row


# UI state for stack builder
if "stack_layers" not in st.session_state:
    st.session_state["stack_layers"] = []
if "stacks_catalog" not in st.session_state:
    st.session_state["stacks_catalog"] = pd.DataFrame(columns=SCHEMA_ORDER)

# 1) Layer selection
st.markdown("### 1) Select layers (outside → inside)")

if "Material_ID" in df_f.columns:
    df_palette = df_f.copy()
    df_palette_display = df_palette[[c for c in ["Material_ID","Material_Name","Material_Class","Test_Environment","Thickness_mm","Areal_Density_gm2","Thermal_Conductivity_Through_WmK","Bulk_Density_kgm3"] if c in df_palette.columns]]
    with st.expander("Available materials palette", expanded=False):
        st.dataframe(df_palette_display.head(200), use_container_width=True)

    options = [f"{row.Material_Name} [{row.Material_Class}] :: {row.Material_ID}" for _, row in df_palette.iterrows()]
    selected = st.selectbox("Choose a material to add as a layer", options) if options else None

    thickness_override = st.number_input("Override thickness for this layer [mm] (0 = use DB value)", min_value=0.0, value=0.0, step=0.05)

    if selected and st.button("Add layer (outside → inside)"):
        # parse Material_ID from selected string
        try:
            mat_id = selected.split("::")[-1].strip()
            st.session_state["stack_layers"].append({"id": mat_id, "thickness_mm": thickness_override})
        except Exception:
            st.warning("Could not parse selected material ID.")

# Show current stack
layers = st.session_state["stack_layers"]
if layers:
    st.markdown("### 2) Current stack order")
    # Build small table for display
    disp_rows = []
    for idx, lyr in enumerate(layers):
        sub = df_f[df_f["Material_ID"] == lyr["id"]]
        if sub.empty:
            continue
        base = sub.iloc[0]
        disp_rows.append({
            "Order (outside→inside)": idx + 1,
            "Material_ID": base.get("Material_ID",""),
            "Material_Name": base.get("Material_Name",""),
            "Class": base.get("Material_Class",""),
            "Env": base.get("Test_Environment",""),
            "DB_Thickness_mm": base.get("Thickness_mm", np.nan),
            "Override_Thickness_mm": lyr.get("thickness_mm", 0.0),
        })
    if disp_rows:
        st.dataframe(pd.DataFrame(disp_rows), use_container_width=True)

    col_a, col_b = st.columns(2)
    if col_a.button("Clear stack"):
        st.session_state["stack_layers"] = []
        layers = []
    # Recompute after potential clear

# Compute stack properties if layers present
if st.session_state["stack_layers"]:
    st.markdown("### 3) Computed stack properties")
    stack_row = _compute_stack_from_layers(df_f, st.session_state["stack_layers"])
    if stack_row:
        stack_df = pd.DataFrame([stack_row])
        show_cols_stack = [c for c in SCHEMA_ORDER if c in stack_df.columns]
        st.dataframe(stack_df[show_cols_stack], use_container_width=True)

        if st.button("Save this stack to catalog"):
            st.session_state["stacks_catalog"] = pd.concat([
                st.session_state["stacks_catalog"],
                stack_df[SCHEMA_ORDER]
            ], ignore_index=True)
            st.success("Stack saved. It can now be ranked like other materials.")
    else:
        st.info("No valid layers to compute. Check that chosen materials exist and have thickness.")

# If we have any saved stacks, rank them using same props/targets
if not st.session_state["stacks_catalog"].empty and props:
    st.markdown("### 4) Ranked stacks (using current targets/weights)")
    stacks_ranked = rank(st.session_state["stacks_catalog"], props, targets, weights, senses, method)
    show_cols = [c for c in SCHEMA_ORDER if c in stacks_ranked.columns]
    show_cols = ["Score"] + [c for c in show_cols if c != "Score"]
    st.dataframe(stacks_ranked[show_cols].head(100), use_container_width=True)


st.caption("© Your Company — AI selector for astronaut/military textiles")

 
# st.set_page_config(page_title="Space Textile Selector", layout="wide")

# # -----------------------------
# # Defaults & constants
# # -----------------------------
# COMPACT_DEFAULT = True  # good for mobile
# AUTO_CSV_PATH = "/space_textile_dataset_AI_ready_ULTRA_v3.csv"  # auto-load if present

# SCHEMA_ORDER = [
#     "Material_ID","Material_Name","Material_Class","Base_Fiber","Coating_Type","Weave_Pattern",
#     "Test_Environment","Areal_Density_gm2","Thickness_mm",
#     "Outgassing_TML_percent","Outgassing_CVCM_percent","Flammability_Rating",
#     "Thermal_Conductivity_Through_WmK","Thermal_Conductivity_InPlane_WmK",
#     "Tensile_Strength_MPa","Tensile_Modulus_GPa","Bulk_Density_kgm3",
#     "Max_Service_Temp_C","k_-100C_WmK","k_0C_WmK","k_23C_WmK","k_150C_WmK","k_200C_WmK",
#     "Notes","Data_Source"
# ]

# NUMERIC_COLS_DEFAULT = [
#     "Areal_Density_gm2","Thickness_mm",
#     "Outgassing_TML_percent","Outgassing_CVCM_percent",
#     "Thermal_Conductivity_Through_WmK","Thermal_Conductivity_InPlane_WmK",
#     "Tensile_Strength_MPa","Tensile_Modulus_GPa","Bulk_Density_kgm3",
#     "Max_Service_Temp_C","k_-100C_WmK","k_0C_WmK","k_23C_WmK","k_150C_WmK","k_200C_WmK"
# ]

# # Subset recommended for compact mobile view
# MOBILE_CORE_COLS = [
#     "Material_Name","Material_Class","Base_Fiber","Weave_Pattern","Test_Environment",
#     "Thermal_Conductivity_Through_WmK","Tensile_Strength_MPa","Bulk_Density_kgm3",
#     "Outgassing_TML_percent","Outgassing_CVCM_percent","Max_Service_Temp_C"
# ]

# RANK_METHODS = {
#     "Distance to targets (L2)": "l2",
#     "Weighted absolute error (L1)": "l1",
#     "Weighted desirability score (0-1)": "desirability"
# }

# # -----------------------------
# # Data loading helpers
# # -----------------------------
# def _load_sample():
#     # Try to load the sandbox path if present
#     try:
#         df = pd.read_csv(AUTO_CSV_PATH)
#         if len(df) > 0:
#             st.sidebar.success("Loaded default ULTRA v3 dataset from sandbox.")
#             return df
#     except Exception:
#         pass
#     # Minimal fallback dataset if no CSV provided and sandbox path missing
#     data = [
#         {"Material_ID":"HP-KEVLAR29-PL-200","Material_Name":"Kevlar 29 fabric","Material_Class":"Textile","Base_Fiber":"Kevlar 29","Coating_Type":"","Weave_Pattern":"Plain","Test_Environment":"Air","Areal_Density_gm2":150,"Thickness_mm":0.2,"Outgassing_TML_percent":np.nan,"Outgassing_CVCM_percent":np.nan,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.0812,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":3000,"Tensile_Modulus_GPa":70,"Bulk_Density_kgm3":1440,"Max_Service_Temp_C":177,"k_23C_WmK":0.0812,"Notes":"Sample row","Data_Source":"DuPont/NIST"},
#         {"Material_ID":"AR-NOMEX410-PA-500","Material_Name":"Nomex 410 paper","Material_Class":"Textile","Base_Fiber":"Nomex 410","Coating_Type":"","Weave_Pattern":"Paper","Test_Environment":"Air","Areal_Density_gm2":200,"Thickness_mm":0.5,"Outgassing_TML_percent":2.44,"Outgassing_CVCM_percent":0.05,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.139,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1100,"Max_Service_Temp_C":220,"k_150C_WmK":0.139,"Notes":"Sample row","Data_Source":"DuPont/NASA"},
#         {"Material_ID":"FILM-MYLAR-AL-12","Material_Name":"Mylar film, aluminized","Material_Class":"Film","Base_Fiber":"Mylar","Coating_Type":"Aluminized","Weave_Pattern":"Film","Test_Environment":"Vacuum","Areal_Density_gm2":np.nan,"Thickness_mm":0.012,"Outgassing_TML_percent":0.26,"Outgassing_CVCM_percent":0.0,"Flammability_Rating":"Pass","Thermal_Conductivity_Through_WmK":0.155,"Thermal_Conductivity_InPlane_WmK":np.nan,"Tensile_Strength_MPa":np.nan,"Tensile_Modulus_GPa":np.nan,"Bulk_Density_kgm3":1390,"Max_Service_Temp_C":150,"k_23C_WmK":0.155,"Notes":"Sample row","Data_Source":"NASA"},
#     ]
#     st.sidebar.warning("Default ULTRA v3 not found in sandbox — using small built-in sample.")
#     return pd.DataFrame(data)


# def load_data():
#     st.sidebar.header("1) Load your dataset")
#     # Auto-load default if available
#     try:
#         df_auto = pd.read_csv(AUTO_CSV_PATH)
#         if len(df_auto) > 0:
#             st.sidebar.info("Auto-loaded: ULTRA v3 (sandbox)")
#             default_df = df_auto
#         else:
#             default_df = None
#     except Exception:
#         default_df = None

#     upl = st.sidebar.file_uploader("Upload CSV (AI-ready schema)", type=["csv"])
#     if upl is not None:
#         try:
#             df = pd.read_csv(upl)
#             st.sidebar.success("Uploaded CSV loaded.")
#             return df
#         except Exception as e:
#             st.sidebar.error(f"Could not read CSV: {e}")
#             return _load_sample()
#     else:
#         if default_df is not None:
#             return default_df
#         else:
#             return _load_sample()


# def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
#     for c in NUMERIC_COLS_DEFAULT:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors='coerce')
#     return df


# def desirability(x, target, higher_is_better=True, tol=None):
#     if pd.isna(x) or pd.isna(target):
#         return np.nan
#     if tol is None:
#         tol = 0.2 * abs(target) if target != 0 else 0.2
#     if higher_is_better:
#         # 0 if far below, 1 if >= target, smooth ramp
#         return np.clip((x - (target - tol)) / (tol), 0, 1)
#     else:
#         # 1 if <= target, decays as it exceeds target
#         return np.clip((target + tol - x) / (tol), 0, 1)


# def rank(df: pd.DataFrame, selections: list, targets: dict, weights: dict, senses: dict, method: str) -> pd.DataFrame:
#     scored = df.copy()
#     for col in selections:
#         t = targets.get(col, np.nan)
#         w = weights.get(col, 1.0)
#         sense = senses.get(col, "minimize")  # or "maximize"
#         if col not in scored.columns:
#             scored[col] = np.nan
#         x = scored[col].astype(float)
#         if method == "l2":
#             err = (x - t)**2
#             if sense == "maximize":
#                 err = (np.maximum(0, t - x))**2  # penalize only if below target
#             scored[f"score__{col}"] = -w * err
#         elif method == "l1":
#             err = np.abs(x - t)
#             if sense == "maximize":
#                 err = np.maximum(0, t - x)
#             scored[f"score__{col}"] = -w * err
#         else:  # desirability
#             hib = (sense == "maximize")
#             d = x.apply(lambda v: desirability(v, t, higher_is_better=hib))
#             scored[f"score__{col}"] = w * d

#     score_cols = [c for c in scored.columns if c.startswith("score__")]
#     if method in ("l1","l2"):
#         scored["Score"] = scored[score_cols].sum(axis=1, skipna=True)
#     else:
#         filled = scored[score_cols].fillna(0)
#         scored["Score"] = filled.sum(axis=1) / np.maximum((~scored[score_cols].isna()).sum(axis=1), 1)

#     scored = scored.sort_values("Score", ascending=False)
#     return scored

# # -----------------------------
# # UI
# # -----------------------------
# st.title("🛰️ Space Textile & Material Selector")
# st.caption("Rank materials by your target properties. Upload the AI-ready CSV or use the default if present.")

# # Load
# df = load_data()
# df = coerce_numeric(df)

# with st.expander("Preview dataset", expanded=False):
#     st.dataframe(df.head(25), use_container_width=True)

# # Filters
# st.sidebar.header("2) Filter (optional)")
# col_filter1, col_filter2 = st.sidebar.columns(2)

# classes = sorted(df["Material_Class"].dropna().unique().tolist()) if "Material_Class" in df.columns else []
# envs = sorted(df["Test_Environment"].dropna().unique().tolist()) if "Test_Environment" in df.columns else []

# selected_classes = col_filter1.multiselect("Material_Class", classes, default=classes)
# selected_envs = col_filter2.multiselect("Test_Environment", envs, default=envs)

# mask = pd.Series([True] * len(df))
# if "Material_Class" in df.columns and selected_classes:
#     mask &= df["Material_Class"].isin(selected_classes)
# if "Test_Environment" in df.columns and selected_envs:
#     mask &= df["Test_Environment"].isin(selected_envs)

# df_f = df[mask].copy()

# # Mobile compact mode toggle
# compact = st.sidebar.checkbox("Mobile compact mode", value=COMPACT_DEFAULT)

# # Property selection
# st.sidebar.header("3) Select properties & targets")
# num_cols_available = [c for c in NUMERIC_COLS_DEFAULT if c in df_f.columns]
# props = st.sidebar.multiselect("Properties of interest", num_cols_available, default=["Thermal_Conductivity_Through_WmK","Bulk_Density_kgm3"]) 

# # Targets/weights/sense per property
# targets, weights, senses = {}, {}, {}
# for p in props:
#     with st.sidebar.expander(f"➡️ {p}", expanded=False):
#         default_target = float(np.nanmedian(df_f[p])) if p in df_f.columns else 0.0
#         t = st.number_input(f"Target for {p}", value=float(default_target), key=f"t_{p}")
#         w = st.slider(f"Weight for {p}", 0.0, 5.0, 1.0, 0.1, key=f"w_{p}")
#         s = st.selectbox(f"Optimization sense for {p}", ["minimize","maximize"], index=0, key=f"s_{p}")
#         targets[p] = t
#         weights[p] = w
#         senses[p] = s

# # Method
# st.sidebar.header("4) Ranking method")
# method_label = st.sidebar.selectbox("Method", list(RANK_METHODS.keys()), index=0)
# method = RANK_METHODS[method_label]

# # Run ranking
# st.sidebar.header("5) Run")
# if st.sidebar.button("Generate ranked list", type="primary"):
#     if len(props) == 0:
#         st.warning("Please select at least one property to rank.")
#     else:
#         scored = rank(df_f, props, targets, weights, senses, method)
#         # Choose columns to display
#         show_cols = [c for c in SCHEMA_ORDER if c in scored.columns]
#         show_cols = ["Score"] + [c for c in show_cols if c != "Score"]

#         st.subheader("Ranked materials")
#         if compact:
#             # Show a condensed table
#             compact_cols = ["Score"] + [c for c in MOBILE_CORE_COLS if c in scored.columns]
#             st.dataframe(scored[compact_cols].head(200), use_container_width=True)
#             # Card-style view for top 25
#             st.markdown("---")
#             st.markdown("### Top 25 (cards)")
#             top = scored.head(25)
#             for _, r in top.iterrows():
#                 st.markdown(f"**{r.get('Material_Name','(Unnamed)')}**  ")
#                 st.caption(f"Score: {r.get('Score',np.nan):.3f} | Class: {r.get('Material_Class','')} | Env: {r.get('Test_Environment','')}")
#                 cols = st.columns(3)
#                 cols[0].markdown(f"**k_through**: {r.get('Thermal_Conductivity_Through_WmK',np.nan)} W/m·K  ")
#                 cols[0].markdown(f"**Density**: {r.get('Bulk_Density_kgm3',np.nan)} kg/m³  ")
#                 cols[1].markdown(f"**Tensile**: {r.get('Tensile_Strength_MPa',np.nan)} MPa  ")
#                 cols[1].markdown(f"**Modulus**: {r.get('Tensile_Modulus_GPa',np.nan)} GPa  ")
#                 cols[2].markdown(f"**TML/CVCM**: {r.get('Outgassing_TML_percent',np.nan)}% / {r.get('Outgassing_CVCM_percent',np.nan)}%  ")
#                 st.markdown("---")
#         else:
#             st.dataframe(scored[show_cols].head(200), use_container_width=True)

#         # Download CSV
#         out = BytesIO()
#         scored.to_csv(out, index=False)
#         st.download_button("Download ranked results (CSV)", data=out.getvalue(), file_name="ranked_materials.csv", mime="text/csv")
    

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
