import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# PAGE + SESSION STATE
st.set_page_config(page_title="From vacancy to housing")

if "manual_buildings" not in st.session_state:
    st.session_state["manual_buildings"] = []
if "show_add_dialog" not in st.session_state:
    st.session_state["show_add_dialog"] = False
if "top_n" not in st.session_state:
    st.session_state["top_n"] = 5

df = pd.read_csv("VB_set_full.csv")

if st.session_state["manual_buildings"]:
    df_manual = pd.DataFrame(st.session_state["manual_buildings"])
    df = pd.concat([df, df_manual], ignore_index=True)

st.title("Decision-support tool: *From vacancy to housing*")
st.text("This tool supports the early-stage identification of vacant buildings with potential for temporary student housing. Buildings are scored based on a set of spatial, technical, functional, and regulatory criteria. The results provide an initial screening and are not intended as a substitute for detailed feasibility studies.")

# CONSTANTS
CITY_AREA_OPTIONS = ["Stadsdeel Centrum", "Stadsdeel Stratum", "Stadsdeel Tongelre", "Stadsdeel Woensel-Zuid", "Stadsdeel Woensel-Noord", "Stadsdeel Strijp", "Stadsdeel Gestel", ]
AHP_WEIGHTS = {"cr_layout": 0.209, "cr_technical": 0.279, "cr_typology": 0.092, "cr_edu_access": 0.106, "cr_amenities": 0.070, "cr_regulatory": 0.244,}
NEEDED_SC = ["sc_ufa", "sc_height", "sc_year", "sc_energy", "sc_dom_use", "sc_parcel", "sc_travel_edu", "sc_amenity", "sc_monument", "sc_landuse",]
ENERGY_LABELS = ["", "A++++", "A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
DOMINANT_USES = ["", "industrial", "retail", "educational / healthcare", "office", "residential"]
LAND_USE_OPTIONS = ["", "residential", "other"]

missing = [c for c in NEEDED_SC if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# HELPERS
def clip01(x):
    return np.clip(x, 0.0, 1.0)

def make_full_address(df_: pd.DataFrame) -> pd.Series:
    street = df_.get("street", "").fillna("").astype(str).str.strip()
    number = df_.get("number", "").fillna("").astype(str).str.strip()
    addition = df_.get("addition", "").fillna("").astype(str).str.strip()
    full = (street + " " + number).str.strip()
    full = full + addition.apply(lambda a: f" {a}" if a else "")
    return full.str.strip()

df["full_address"] = make_full_address(df)

def emphasize_known_scores(s: pd.Series, m: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = pd.Series(clip01(s.to_numpy()), index=s.index)
    out = 1.0 - np.power((1.0 - s.to_numpy()), m)
    return pd.Series(clip01(out), index=s.index)

def emphasize_with_placeholder_rule(s: pd.Series, m: float, placeholder: float = 0.5) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = pd.Series(clip01(s.to_numpy()), index=s.index)
    out = s.copy()
    mask_unknown = pd.Series(np.isclose(s.to_numpy(), placeholder, atol=1e-9), index=s.index)
    mask_known = ~mask_unknown
    out.loc[mask_known] = emphasize_known_scores(out.loc[mask_known], m)
    out.loc[mask_unknown] = clip01(placeholder * m) if m < 1 else placeholder
    return out

def emphasis_label(v: float) -> str:
    if v < 0.9:
        return "Reduced emphasis"
    if v < 1.1:
        return "Neutral emphasis"
    return "Increased emphasis"

# SCORING FUNCTIONS
def score_ufa_m2(ufa_m2):
    if ufa_m2 is None or pd.isna(ufa_m2):
        return 0.5
    ufa_m2 = float(ufa_m2)
    if ufa_m2 <= 500:
        return 0.2
    if ufa_m2 < 1000:
        return 0.6
    return 1.0

def score_building_height(floors):
    if floors is None or pd.isna(floors):
        return 0.5
    floors = int(floors)
    if floors <= 1:
        return 0.6
    if 2 <= floors <= 6:
        return 1.0
    return 0.6

def score_construction_year(year):
    if year is None or pd.isna(year):
        return 0.5
    year = int(year)
    if year <= 1960:
        return 0.2
    if year < 1990:
        return 0.6
    return 1.0

def score_energy_label(label):
    if label is None or str(label).strip() == "":
        return 0.5
    label = str(label).strip().upper()
    if label in ["F", "G"]:
        return 0.2
    if label in ["C", "D", "E"]:
        return 0.6
    if label in ["A", "B", "A+", "A++", "A+++", "A++++"]:
        return 1.0
    return 0.5

def score_dominant_use(use):
    if use is None or str(use).strip() == "":
        return 0.5
    u = str(use).strip().lower()
    if "industrial" in u:
        return 0.2
    if "retail" in u:
        return 0.4
    if "educational" in u or "healthcare" in u:
        return 0.6
    if "office" in u:
        return 0.8
    if "residential" in u:
        return 1.0
    return 0.5

def score_parcel_m2(parcel_m2):
    if parcel_m2 is None or pd.isna(parcel_m2):
        return 0.5
    parcel_m2 = float(parcel_m2)
    if parcel_m2 <= 250:
        return 0.2
    if parcel_m2 < 600:
        return 0.6
    return 1.0

def score_travel_education(minutes):
    if minutes is None or pd.isna(minutes):
        return 0.5
    minutes = float(minutes)
    if minutes >= 20:
        return 0.2
    if minutes > 10:
        return 0.6
    return 1.0

def score_amenity_accessibility(raw_0_100):
    if raw_0_100 is None or pd.isna(raw_0_100):
        return 0.5
    v = float(raw_0_100) / 100.0
    return float(np.clip(v, 0.0, 1.0))

def score_monument_status(is_monument):
    if is_monument is None:
        return 0.5
    return 0.2 if bool(is_monument) else 1.0

def score_land_use(land_use):
    if land_use is None or str(land_use).strip() == "":
        return 0.5
    lu = str(land_use).strip().lower()
    return 1.0 if "residential" in lu else 0.6

# SIDEBAR
st.sidebar.header("Policy alignment controls")
st.sidebar.subheader("Indicator emphasis")
st.sidebar.caption("Emphasis increases/decreases indicator influence but scores are capped at 1.00 (a building that already scores 1.00 cannot exceed it). Unknown values can be reduced but never boosted.")

m_ufa = st.sidebar.slider("Usable floor area (UFA)", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption(f"Higher = larger UFA benefits more. Current: **{emphasis_label(m_ufa)}** ({m_ufa:.2f})")

m_year = st.sidebar.slider("Construction year", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption(f"Higher = newer buildings benefit more. Current: **{emphasis_label(m_year)}** ({m_year:.2f})")

m_energy = st.sidebar.slider("Energy performance", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption(f"Higher = better energy labels benefit more. Current: **{emphasis_label(m_energy)}** ({m_energy:.2f})")

#SPATIAL STEERING
st.sidebar.subheader("Spatial steering")
steering_level = st.sidebar.radio("Steering level", ["City area", "Neighbourhood"], horizontal=True)
area_col = "city_area" if steering_level == "City area" else "neighbourhood"

if "neighbourhood" in df.columns:
    neighbourhood_options = sorted(
        [n for n in df["neighbourhood"].dropna().astype(str).unique() if n.strip() != ""]
    )
else:
    neighbourhood_options = []

if area_col in df.columns:
    area_options = sorted(
        [x for x in df[area_col].dropna().astype(str).unique() if x.strip() != ""]
    )
else:
    area_options = []

st.sidebar.caption("Assign positive or negative influence to selected areas (overlap is not allowed).")

pos_areas = st.sidebar.multiselect("Positive areas", options=area_options, default=[])
neg_options = [a for a in area_options if a not in pos_areas]
neg_areas = st.sidebar.multiselect("Negative areas", options=neg_options, default=[])
pos_boost = st.sidebar.slider("Positive multiplier", 1.00, 1.25, 1.00, 0.01)
neg_penalty = st.sidebar.slider("Negative multiplier", 0.75, 1.00, 1.00, 0.01)

show_only_focus = st.sidebar.checkbox("Show only buildings in selected focus areas", value=False)
exclude_negative = st.sidebar.checkbox("Exclude negative areas entirely", value=False)

# BASE CRITERION SCORES
df["cr_layout_base"] = 0.5 * df["sc_ufa"] + 0.5 * df["sc_height"]
df["cr_technical_base"] = 0.5 * df["sc_year"] + 0.5 * df["sc_energy"]
df["cr_typology_base"] = 0.5 * df["sc_dom_use"] + 0.5 * df["sc_parcel"]
df["cr_edu_access_base"] = df["sc_travel_edu"]
df["cr_amenities_base"] = df["sc_amenity"]
df["cr_regulatory_base"] = 0.5 * df["sc_monument"] + 0.5 * df["sc_landuse"]

# ADJUSTED INDICATOR SCORES
df["sc_ufa_adj"] = emphasize_with_placeholder_rule(df["sc_ufa"], m_ufa)
df["sc_year_adj"] = emphasize_with_placeholder_rule(df["sc_year"], m_year)
df["sc_energy_adj"] = emphasize_with_placeholder_rule(df["sc_energy"], m_energy)

# ADJUSTED CRITERION SCORES
df["cr_layout_adj"] = 0.5 * df["sc_ufa_adj"] + 0.5 * df["sc_height"]
df["cr_technical_adj"] = 0.5 * df["sc_year_adj"] + 0.5 * df["sc_energy_adj"]
df["cr_typology_adj"] = df["cr_typology_base"]
df["cr_edu_access_adj"] = df["cr_edu_access_base"]
df["cr_amenities_adj"] = df["cr_amenities_base"]
df["cr_regulatory_adj"] = df["cr_regulatory_base"]

# WEIGHTED SUITABILITY SCORE
def weighted_sum(prefix: str) -> pd.Series:
    return (
        AHP_WEIGHTS["cr_layout"] * df[f"cr_layout_{prefix}"] +
        AHP_WEIGHTS["cr_technical"] * df[f"cr_technical_{prefix}"] +
        AHP_WEIGHTS["cr_typology"] * df[f"cr_typology_{prefix}"] +
        AHP_WEIGHTS["cr_edu_access"] * df[f"cr_edu_access_{prefix}"] +
        AHP_WEIGHTS["cr_amenities"] * df[f"cr_amenities_{prefix}"] +
        AHP_WEIGHTS["cr_regulatory"] * df[f"cr_regulatory_{prefix}"]
    )

df["suitability_base"] = weighted_sum("base")
df["suitability_user"] = weighted_sum("adj")

# FOCUS BOOST
df["steer_multiplier"] = 1.0

if area_col in df.columns:
    if pos_areas:
        df.loc[df[area_col].astype(str).isin(pos_areas), "steer_multiplier"] *= pos_boost

    if neg_areas:
        if exclude_negative:
            # Mark for exclusion later
            df["exclude_flag"] = False
            df.loc[df[area_col].astype(str).isin(neg_areas), "exclude_flag"] = True
        else:
            df.loc[df[area_col].astype(str).isin(neg_areas), "steer_multiplier"] *= neg_penalty
else:
    df["exclude_flag"] = False

df["suitability_focus"] = clip01(df["suitability_user"] * df["steer_multiplier"])

n_affected = (df["steer_multiplier"] != 1.0).sum()
st.sidebar.info(f"Spatial steering applied to {n_affected} buildings (both positive and negative)")

df_display = df.copy()
if exclude_negative and "exclude_flag" in df_display.columns:
    df_display = df_display[~df_display["exclude_flag"]]

df_sorted = df_display.sort_values("suitability_focus", ascending=False)

# MAP
st.subheader("Spatial distribution of buildings (hover over points for details)")
map_df = df_sorted.dropna(subset=["lat", "lon"]).copy()

s = pd.to_numeric(map_df["suitability_focus"], errors="coerce").fillna(0.0)
s_min, s_max = float(s.min()), float(s.max())

map_df["suitability_focus_str"] = map_df["suitability_focus"].map(lambda x: f"{x:.2f}")
map_df["suitability_base_str"] = map_df["suitability_base"].map(lambda x: f"{x:.2f}")

tooltip_fields = {"Address": "full_address", "City area": "city_area", "Score (scenario)": "suitability_focus_str", "Score (base)": "suitability_base_str"}

tooltip_html = "<b>Building information</b><br/>"
for label, col in tooltip_fields.items():
    if col in map_df.columns:
        tooltip_html += f"{label}: {{{col}}}<br/>"

view_state = pdk.ViewState(
    latitude=float(map_df["lat"].mean()) if len(map_df) else 52.0,
    longitude=float(map_df["lon"].mean()) if len(map_df) else 5.0,
    zoom=12,
    pitch=0,
)

# SCALE SCORES TO 0-1 FOR COLOURING
if s_max > s_min:
    score_scaled = (s - s_min) / (s_max - s_min)
else:
    score_scaled = s * 0.0

map_df["score_scaled"] = score_scaled

def score_to_rgb(x: float):
    if x <= 0.5:
        return [255, int(round(510 * x)), 0]
    return [int(round(510 * (1 - x))), 255, 0]

map_df["fill_color"] = map_df["score_scaled"].apply(score_to_rgb)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[lon, lat]',
    get_fill_color="fill_color",
    get_radius=45,
    pickable=True,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": tooltip_html},
    map_style=pdk.map_styles.CARTO_LIGHT,
)

st.caption(f"Colors indicate relative suitability within the displayed set (min = {s_min:.2f}, max = {s_max:.2f}).")
st.pydeck_chart(deck, use_container_width=True)

# TOP BUILDINGS
st.subheader("Top buildings (with current settings)")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Show 5 more"):
        st.session_state["top_n"] = min(st.session_state["top_n"] + 5, len(df_sorted))
with col2:
    if st.button("Reset to top 5"):
        st.session_state["top_n"] = 5

top_n = st.session_state["top_n"]
topN = df_sorted.head(top_n).copy()

topN_disp = topN[["full_address", "suitability_focus"]].copy()
topN_disp["suitability_focus"] = topN_disp["suitability_focus"].map(lambda x: f"{x:.2f}")
topN_disp = topN_disp.rename(columns={"full_address": "Address", "suitability_focus": "Suitability score"})

st.markdown(f"Showing top **{top_n}** buildings:")
st.dataframe(topN_disp, use_container_width=True, hide_index=True)

# INSPECT BUILDING
st.subheader("Inspect a building")

mode = st.radio("Select mode", [f"Select from top {top_n}", "Search by address"], horizontal=True,)
selected_row = None

if mode == f"Select from top {top_n}":
    topN["label"] = topN["full_address"].astype(str)
    selected_label = st.selectbox(
        "Select a building from the shortlist",
        options=topN["label"].tolist(),
        key="shortlist_select",
    )
    selected_row = topN[topN["label"] == selected_label].iloc[0]

else:
    query = st.text_input("Type (part of) an address", value="", key="search_query")
    if query.strip():
        hits = df_sorted[df_sorted["full_address"].astype(str).str.contains(query, case=False, na=False)].copy()
        if hits.empty:
            st.warning("No matching addresses found in the current dataset/filter.")
        else:
            hits = hits.head(50)
            hits["label"] = hits["full_address"].astype(str)
            selected_label = st.selectbox("Select a result", options=hits["label"].tolist(), key="search_select")
            selected_row = hits[hits["label"] == selected_label].iloc[0]
    else:
        st.info("Start typing an address to search within the current dataset/filter.")

if selected_row is not None:
    st.markdown(f"### {selected_row['full_address']}")
    st.write(
        "**Suitability (base / scenario):**",
        f"{float(selected_row['suitability_base']):.2f} / {float(selected_row['suitability_focus']):.2f}",
    )

    INFO_FIELDS = [
        ("BAG ID", "bag_id"),
        ("Address", "full_address"),
        ("Construction year", "construction_year"),
        ("Energy label", "energy_label"),
        ("Current function", "dominant_use"),
        ("Usable floor area (m²)", "ufa_m2"),
        ("Neighbourhood", "neighbourhood"),
        ("City area", "city_area"),
    ]

    st.subheader("Building information")
    info_rows = []
    for label, col in INFO_FIELDS:
        if col in selected_row.index:
            val = selected_row[col]
            if pd.isna(val):
                val = "—"
            elif isinstance(val, (float, int, np.floating, np.integer)):
                val = round(float(val), 2)
            info_rows.append({"Attribute": label, "Value": val})
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    # CRITERION BARS
    CR_LABELS = {
        "cr_layout_adj": "Internal layout",
        "cr_technical_adj": "Technical feasibility",
        "cr_typology_adj": "Building typology",
        "cr_edu_access_adj": "Access to education",
        "cr_amenities_adj": "Access to amenities",
        "cr_regulatory_adj": "Regulatory flexibility",
    }
    chart_df = pd.DataFrame({
        "Criterion": [CR_LABELS[c] for c in CR_LABELS.keys()],
        "Score": [float(selected_row[c]) for c in CR_LABELS.keys()],
    }).sort_values("Score", ascending=True)

    st.subheader("Criterion scores for selected building")
    st.bar_chart(chart_df.set_index("Criterion"), use_container_width=True)

    with st.expander("Show more attributes (raw)"):
        st.dataframe(
            pd.DataFrame(selected_row).reset_index().rename(columns={"index": "Field", 0: "Value"}),
            use_container_width=True,
            hide_index=True,
        )

# ADD / DOWNLOAD / CLEAR
st.subheader("Options")

@st.dialog("Add building", width="large")
def add_building_dialog():
    with st.form("manual_building_form", clear_on_submit=True):
        st.caption("Enter attributes. These will be converted to indicator scores using the same class-based rules as the base dataset. Leave unknown fields empty; they will be treated as unknown (score = 0.50).")
        
        st.markdown("**Location & identification**")
        c1, c2, c3 = st.columns(3)
        with c1:
            street = st.text_input("Street*")
            city_area = st.selectbox("City area (optional)", ["—"] + CITY_AREA_OPTIONS)
            neighbourhood = st.selectbox("Neighbourhood (optional)", ["—"] + neighbourhood_options)

        with c2:
            number = st.text_input("Number*")
            lat = st.number_input("Latitude*", format="%.6f")
        with c3:
            addition = st.text_input("Addition (optional)")
            lon = st.number_input("Longitude*", format="%.6f")

        st.markdown("**Building characteristics**")
        c1, c2, c3 = st.columns(3)
        with c1:
            ufa_m2 = st.number_input("Usable floor area (m²)", min_value=0.0, step=10.0, value=0.0)
            parcel_m2 = st.number_input("Parcel size (m²)", min_value=0.0, step=10.0, value=0.0)
            energy_label = st.selectbox("Energy label", ["", "A++++", "A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"])
        with c2:
            floors = st.number_input("Building height (floors)", min_value=0, step=1, value=0)
            travel_education = st.number_input("Cycling time to higher education (minutes)", min_value=0.0, step=1.0, value=0.0)
            dominant_use = st.selectbox("Dominant building use", ["", "industrial", "retail", "educational / healthcare", "office", "residential"])
        with c3:
            construction_year = st.number_input("Construction year", min_value=0, step=1, value=0)
            amenity_accessibility = st.number_input("Amenity accessibility score (0–100)", min_value=0.0, max_value=100.0, step=1.0, value=0.0)
            land_use = st.selectbox("Land use / zoning", ["", "residential", "other"])         
        monument_status = st.checkbox("This is a monument", value=False)
        st.markdown("**PLEASE NOTE:** The building is only added to the current session. After reopening the tool it will no longer be visible in the dataset. Download the .csv file to save your results.")

        submitted = st.form_submit_button("Add building")
        if not submitted:
            return

        if not street.strip() or not number.strip():
            st.error("Street and number are required.")
            return
        if (lat == 0.0 and lon == 0.0) or pd.isna(lat) or pd.isna(lon):
            st.error("Please provide valid latitude and longitude.")
            return

        # INTERPRET 0 AS UNKNOWN
        ufa_val = np.nan if ufa_m2 <= 0 else float(ufa_m2)
        floors_val = np.nan if floors <= 0 else int(floors)
        year_val = np.nan if construction_year <= 0 else int(construction_year)
        parcel_val = np.nan if parcel_m2 <= 0 else float(parcel_m2)
        travel_val = np.nan if travel_education <= 0 else float(travel_education)
        amenity_val = amenity_accessibility

        new_row = {
            "street": street.strip(),
            "number": number.strip(),
            "addition": addition.strip(),
            "city_area": None if city_area == "—" else city_area,
            "lat": float(lat),
            "lon": float(lon),

            "ufa_m2": ufa_val,
            "building_height": floors_val,
            "construction_year": year_val,
            "energy_label": energy_label,
            "dominant_use": dominant_use,
            "parcel_m2": parcel_val,
            "travel_education": travel_val,
            "amenity_accessibility": amenity_val,
            "monument_status": bool(monument_status),
            "land_use": land_use,
            "neighbourhood": None if neighbourhood == "—" else neighbourhood,

            "sc_ufa": score_ufa_m2(ufa_val),
            "sc_height": score_building_height(floors_val),
            "sc_year": score_construction_year(year_val),
            "sc_energy": score_energy_label(energy_label),
            "sc_dom_use": score_dominant_use(dominant_use),
            "sc_parcel": score_parcel_m2(parcel_val),
            "sc_travel_edu": score_travel_education(travel_val),
            "sc_amenity": score_amenity_accessibility(amenity_val),
            "sc_monument": score_monument_status(monument_status),
            "sc_landuse": score_land_use(land_use),
        }

        st.session_state["manual_buildings"].append(new_row)
        st.success("Building added. (It will appear after the app reruns.)")

util1, util2, util3 = st.columns([1, 1, 1])

with util1:
    if st.button("➕ Add building manually"):
        st.session_state["show_add_dialog"] = True
with util2:
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results (CSV)",
        data=csv_bytes,
        file_name="screening_results_current_settings.csv",
        mime="text/csv",
        use_container_width=True,)
with util3:
    if st.button("🗑️ Clear manual buildings"):
        st.session_state["manual_buildings"] = []
        st.success("Manual buildings cleared.")

if st.session_state.get("show_add_dialog", False):
    add_building_dialog()
    st.session_state["show_add_dialog"] = False

# FULL DATASET TABLE
with st.expander("Show full dataset (current filter)"):
    st.write(f"Showing **{len(df_sorted)}** buildings.")
    st.dataframe(df_sorted, use_container_width=True)


# python -m streamlit run ds_app_clean.py
# cd "OneDrive - TU Eindhoven\Graduation project\Interface" 