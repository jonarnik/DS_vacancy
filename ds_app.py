import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

st.title("Decision-support tool: *From vacancy to housing*")

df = pd.read_csv("VB_set_full.csv")

# ---------- Helpers ----------
def clip01(arr):
    return np.clip(arr, 0.0, 1.0)

def renormalize(weights: dict) -> dict:
    s = float(sum(weights.values()))
    return {k: (v / s if s else 0.0) for k, v in weights.items()}

def make_full_address(df_: pd.DataFrame) -> pd.Series:
    street = df_.get("street", "").fillna("").astype(str).str.strip()
    number = df_.get("number", "").fillna("").astype(str).str.strip()
    addition = df_.get("addition", "").fillna("").astype(str).str.strip()

    full = (street + " " + number).str.strip()
    full = full + addition.apply(lambda x: f" {x}" if x else "")
    return full.str.strip()

# Add address early so it’s available for tooltips + tables
df["full_address"] = make_full_address(df)

# ---------- Sidebar (policy steering instead of hard filters) ----------
st.sidebar.header("Policy alignment controls")

# AHP baseline weights (sum to 1.0)
AHP_WEIGHTS = {
    "cr_layout": 0.209,
    "cr_technical": 0.279,
    "cr_typology": 0.092,
    "cr_edu_access": 0.106,
    "cr_amenities": 0.070,
    "cr_regulatory": 0.244,
}

# Indicator emphasis
st.sidebar.subheader("Indicator emphasis")
st.sidebar.caption(
    "Emphasis increases/decreases indicator influence but scores are capped at 1.00 "
    "(a building that already scores 1.00 cannot exceed it). It does not change the scoring rules themselves, and does not exclude buildings."
)

m_ufa = st.sidebar.slider("Usable floor area (UFA)", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption("Higher = larger UFA benefits more.")

m_year = st.sidebar.slider("Construction year", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption("Higher = newer buildings benefit more.")

m_energy = st.sidebar.slider("Energy performance", 0.5, 2.0, 1.0, 0.05)
st.sidebar.caption("Higher = better energy labels benefit more.")

# Focus areas using city_area (soft boost so all buildings remain visible)
st.sidebar.subheader("Focus areas")
city_area_options = [
    "Stadsdeel Centrum",
    "Stadsdeel Stratum",
    "Stadsdeel Tongelre",
    "Stadsdeel Woensel-Zuid",
    "Stadsdeel Woensel-Noord",
    "Stadsdeel Strijp",
    "Stadsdeel Gestel",
]

st.sidebar.caption(
    "Selecting focus areas applies a limited score boost to buildings located within these areas."
)

focus_areas = st.sidebar.multiselect("Select focus areas", city_area_options, default=[])
focus_boost = st.sidebar.slider("Focus boost", 1.00, 1.25, 1.10, 0.01)

# Optional “visibility” toggles
st.sidebar.subheader("Visibility options")
show_only_focus = st.sidebar.checkbox("Show only buildings in selected focus areas", value=False)

# ---------- Data prep: ensure sc_* columns numeric in [0,1] ----------
needed_sc = [
    "sc_ufa","sc_height","sc_year","sc_energy","sc_dom_use","sc_parcel",
    "sc_travel_edu","sc_amenity","sc_monument","sc_landuse"
]
missing = [c for c in needed_sc if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

for c in needed_sc:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[c] = clip01(df[c].to_numpy())

# ---------- Baseline criterion scores from indicators ----------
df["cr_layout_base"]     = 0.5*df["sc_ufa"] + 0.5*df["sc_height"]
df["cr_technical_base"]  = 0.5*df["sc_year"] + 0.5*df["sc_energy"]
df["cr_typology_base"]   = 0.5*df["sc_dom_use"] + 0.5*df["sc_parcel"]
df["cr_edu_access_base"] = df["sc_travel_edu"]
df["cr_amenities_base"]  = df["sc_amenity"]
df["cr_regulatory_base"] = 0.5*df["sc_monument"] + 0.5*df["sc_landuse"]

# ---------- Indicator emphasis (examples): modify only chosen indicators ----------
df["sc_ufa_adj"]    = clip01(df["sc_ufa"] * m_ufa)
df["sc_year_adj"]   = clip01(df["sc_year"] * m_year)
df["sc_energy_adj"] = clip01(df["sc_energy"] * m_energy)

# Recompute only affected criteria with adjusted indicators
df["cr_layout_adj"]    = 0.5*df["sc_ufa_adj"] + 0.5*df["sc_height"]  # height unchanged
df["cr_technical_adj"] = 0.5*df["sc_year_adj"] + 0.5*df["sc_energy_adj"]

# Other criteria unchanged (baseline)
df["cr_typology_adj"]   = df["cr_typology_base"]
df["cr_edu_access_adj"] = df["cr_edu_access_base"]
df["cr_amenities_adj"]  = df["cr_amenities_base"]
df["cr_regulatory_adj"] = df["cr_regulatory_base"]

# ---------- Baseline suitability (AHP default) ----------
df["suitability_base"] = (
    AHP_WEIGHTS["cr_layout"]     * df["cr_layout_base"] +
    AHP_WEIGHTS["cr_technical"]  * df["cr_technical_base"] +
    AHP_WEIGHTS["cr_typology"]   * df["cr_typology_base"] +
    AHP_WEIGHTS["cr_edu_access"] * df["cr_edu_access_base"] +
    AHP_WEIGHTS["cr_amenities"]  * df["cr_amenities_base"] +
    AHP_WEIGHTS["cr_regulatory"] * df["cr_regulatory_base"]
)

# ---------- Suitability with fixed AHP weights (indicator emphasis only) ----------
df["suitability_user"] = (
    AHP_WEIGHTS["cr_layout"]     * df["cr_layout_adj"] +
    AHP_WEIGHTS["cr_technical"]  * df["cr_technical_adj"] +
    AHP_WEIGHTS["cr_typology"]   * df["cr_typology_adj"] +
    AHP_WEIGHTS["cr_edu_access"] * df["cr_edu_access_adj"] +
    AHP_WEIGHTS["cr_amenities"]  * df["cr_amenities_adj"] +
    AHP_WEIGHTS["cr_regulatory"] * df["cr_regulatory_adj"]
)

# ---------- Focus boost by city_area (soft, keeps all buildings visible) ----------
df["focus_multiplier"] = 1.0
if focus_areas and "city_area" in df.columns:
    df.loc[df["city_area"].isin(focus_areas), "focus_multiplier"] = focus_boost

df["suitability_focus"] = clip01(df["suitability_user"] * df["focus_multiplier"])

# Optional: show only focus (visibility choice, not recommended as default)
df_display = df.copy()
if show_only_focus and focus_areas and "city_area" in df.columns:
    df_display = df_display[df_display["city_area"].isin(focus_areas)]

# ---------- Sort by scenario score ----------
df_sorted = df_display.sort_values("suitability_focus", ascending=False)



# ---------- Output map ----------
map_df = df_sorted.dropna(subset=["lat", "lon"]).copy()

map_df["suitability_focus_str"] = map_df["suitability_focus"].map(lambda x: f"{x:.2f}")
map_df["suitability_base_str"]  = map_df["suitability_base"].map(lambda x: f"{x:.2f}")






# ---------- Pydeck map ----------
tooltip_fields = {
    "Address": "full_address",
    "City area": "city_area",
    "Score (scenario)": "suitability_focus_str",
    "Score (base)": "suitability_base_str",
}


tooltip_html = "<b>Building information</b><br/>"
for label, col in tooltip_fields.items():
    if col in map_df.columns:
        tooltip_html += f"{label}: " + "{" + f"{col}" + "}" + "<br/>"

view_state = pdk.ViewState(
    latitude=float(map_df["lat"].mean()) if len(map_df) else 52.0,
    longitude=float(map_df["lon"].mean()) if len(map_df) else 5.0,
    zoom=12,
    pitch=0,
)

# Color gradients for suitability_focus
s = pd.to_numeric(map_df["suitability_focus"], errors="coerce").fillna(0.0)
s_min = float(s.min())
s_max = float(s.max())

if s_max > s_min:
    s_scaled = (s - s_min) / (s_max - s_min)
else:
    s_scaled = s * 0

map_df["score_scaled"] = s_scaled

def score_to_rgb(x: float):
    if x <= 0.5:
        r = 255
        g = int(round(510 * x))
        b = 0
    else:
        r = int(round(510 * (1 - x)))
        g = 255
        b = 0
    return [r, g, b]

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

st.subheader("Spatial distribution of buildings (hover over points for details)")

st.caption(
    f"Colors indicate relative suitability within the displayed set "
    f"(min = {s_min:.2f}, max = {s_max:.2f})."
)
st.pydeck_chart(deck, use_container_width=True)


# top 5 buildings selection

st.subheader("Top 5 buildings (with current settings)")

top_n = 5
top5 = df_sorted.sort_values("suitability_focus", ascending=False).head(top_n).copy()

# Choose what to show (adjust columns to your dataset)
cols = [c for c in ["full_address", "suitability_focus"] if c in top5.columns]
st.dataframe(top5[cols], use_container_width=True)
st.markdown("**Inspect a shortlisted building:**")

# Use a stable ID if you have one (bag_id is ideal). Otherwise full_address.
id_col = "bag_id" if "bag_id" in top5.columns else "full_address"

# Build label shown in the dropdown
top5["label"] = top5["full_address"].astype(str)

selected_label = st.selectbox(
    "Select a building from the Top 5",
    options=top5["label"].tolist()
)

selected_row = top5[top5["label"] == selected_label].iloc[0]

st.markdown(f"### {selected_row['full_address']}")
st.write("**Total suitability:**", float(selected_row["suitability_total"]))

#Building information table

INFO_FIELDS = [
    ("BAG ID", "bag_id"),
    ("Address", "full_address"),
    ("Construction year", "construction_year"),
    ("Energy label", "energy_label"),
    ("Current function", "dominant_use"),
    ("Usable floor area (m²)", "ufa_m2"),
    ("Neighbourhood", "neighbourhood"),
    ("City area", "city_area")
]

st.subheader("Building information")

info_rows = []
for label, col in INFO_FIELDS:
    if col in top5.columns:
        val = selected_row[col]
        # Clean up NaN/None for display
        if pd.isna(val):
            val = "—"
        info_rows.append({"Attribute": label, "Value": val})

info_df = pd.DataFrame(info_rows)
st.dataframe(info_df, use_container_width=True, hide_index=True)

with st.expander("Show more attributes (raw)"):
    st.dataframe(pd.DataFrame(selected_row).reset_index().rename(columns={"index":"Field", 0:"Value"}))

CR_LABELS = {
    "cr_layout_adj": "Internal layout",
    "cr_technical_adj": "Technical feasibility",
    "cr_typology_adj": "Building typology",
    "cr_edu_access_adj": "Access to education",
    "cr_amenities_adj": "Access to amenities",
    "cr_regulatory_adj": "Regulatory flexibility",
}

import pandas as pd

cr_cols = list(CR_LABELS.keys())

chart_df = pd.DataFrame({
    "Criterion": [CR_LABELS[c] for c in cr_cols],
    "Score": [float(selected_row[c]) for c in cr_cols],
}).sort_values("Score", ascending=True)  # horizontal bars look nicer this way

st.subheader("Criterion scores for selected building")
st.bar_chart(
    chart_df.set_index("Criterion"),
    use_container_width=True
)



st.subheader("Full dataset of buildings on map")

st.write(f"Showing **{len(df_sorted)}** buildings.")
st.dataframe(
    df_sorted,
    use_container_width=True
)

# python -m streamlit run ds_app.py
# cd "OneDrive - TU Eindhoven\Graduation project\Interface"