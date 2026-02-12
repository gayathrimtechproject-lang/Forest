import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# --------------------------------------------------
# DASHBOARD CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Telangana Tree Cover Dashboard",
    page_icon="🌳",
    layout="wide"
)

# Custom Styling
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

st.title("🌳 Telangana Tree Cover Loss Analysis (2001–2024)")
st.markdown("Analysis based on Global Forest Watch (GFW) Subnational Data.")

# --------------------------------------------------
# FILE UPLOAD & PRE-PROCESSING
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload 'Subnational_2_tree_cover_loss.csv'", 
    type=["csv"],
    help="Upload the GFW CSV file containing tree cover loss data."
)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # Standardize Column Names
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    
    # Filter for Telangana, India
    if 'country' in df_raw.columns and 'subnational1' in df_raw.columns:
        df = df_raw[
            (df_raw["country"].astype(str).str.lower() == "india") & 
            (df_raw["subnational1"].astype(str).str.lower() == "telangana")
        ].copy()
    else:
        st.error("Missing location columns ('country' or 'subnational1').")
        st.stop()

    if df.empty:
        st.warning("No data found for Telangana. Please check your CSV file.")
        st.stop()

    # --------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------
    st.sidebar.header("Filter Controls")
    
    # 1. Threshold Selection (Prevents double counting rows)
    if 'threshold' in df.columns:
        available_thresholds = sorted(df['threshold'].unique())
        selected_threshold = st.sidebar.selectbox(
            "Canopy Cover Threshold (%)", 
            available_thresholds, 
            index=min(5, len(available_thresholds)-1)
        )
        df = df[df['threshold'] == selected_threshold]
    
    # 2. District Selection
    districts = sorted(df["subnational2"].unique())
    selected_district = st.sidebar.selectbox("🌍 Select District", districts)

    # --------------------------------------------------
    # YEAR COLUMN DETECTION
    # --------------------------------------------------
    # Specifically target 'tc_loss_ha_YYYY' pattern
    year_map = {}
    for col in df.columns:
        match = re.search(r'tc_loss_ha_(\d{4})', col)
        if match:
            year = int(match.group(1))
            if 2001 <= year <= 2024:
                year_map[col] = year

    year_cols = sorted(year_map.keys(), key=lambda x: year_map[x])
    
    if not year_cols:
        st.error("No loss data columns detected (e.g., tc_loss_ha_2001).")
        st.stop()

    # --------------------------------------------------
    # DATA AGGREGATION
    # --------------------------------------------------
    dist_df = df[df["subnational2"] == selected_district]
    
    # Create Trend DataFrame
    trend_data = []
    for col in year_cols:
        trend_data.append({
            "Year": year_map[col],
            "Loss": dist_df[col].sum()
        })
    trend_df = pd.DataFrame(trend_data)

    # Metrics
    total_loss_ha = trend_df["Loss"].sum()
    max_loss_year = trend_df.loc[trend_df["Loss"].idxmax(), "Year"]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Loss (2001-2024)", f"{total_loss_ha:,.1f} Ha")
    m2.metric("Peak Loss Year", int(max_loss_year))
    m3.metric("Threshold Filter", f"{selected_threshold}%")

    # --------------------------------------------------
    # CHART 1: ANNUAL LOSS BAR CHART
    # --------------------------------------------------
    st.subheader(f"📊 Annual Tree Cover Loss - {selected_district.title()}")
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=trend_df, x="Year", y="Loss", color="teal", ax=ax1)
    ax1.set_ylabel("Loss (Hectares)")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # --------------------------------------------------
    # CHART 2: STABILIZED YOY % CHANGE
    # --------------------------------------------------
    st.subheader("📈 Year-over-Year Growth Rate")
    
    trend_df['YoY %'] = trend_df['Loss'].pct_change() * 100
    # Clean data: Replace infinity and clip extremes for visualization
    yoy_clean = trend_df.replace([np.inf, -np.inf], np.nan).dropna()
    yoy_clean['Color'] = np.where(yoy_clean['YoY %'] >= 0, '#e74c3c', '#27ae60') # Red for increase, Green for decrease

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(yoy_clean["Year"], yoy_clean["YoY %"], color=yoy_clean['Color'])
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel("% Change in Loss")
    ax2.set_xlabel("Year")
    st.pyplot(fig2)

    # --------------------------------------------------
    # CHART 3: DISTRICT COMPARISON
    # --------------------------------------------------
    st.subheader("📊 District Comparison: Total Loss (Ha)")
    
    comp_data = []
    for d in districts:
        d_rows = df[df["subnational2"] == d]
        total = d_rows[year_cols].sum().sum()
        comp_data.append({"District": d.title(), "Total Loss": total})
    
    comp_df = pd.DataFrame(comp_data).sort_values("Total Loss", ascending=True)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comp_df, y="District", x="Total Loss", palette="viridis", ax=ax3)
    ax3.set_xlabel("Cumulative Loss (Hectares)")
    st.pyplot(fig3)

    # --------------------------------------------------
    # CHART 4: HEATMAP
    # --------------------------------------------------
    st.subheader("🔥 Temporal Loss Heatmap by District")
    
    heatmap_data = df.groupby("subnational2")[year_cols].sum()
    heatmap_data.columns = [year_map[c] for c in heatmap_data.columns]
    heatmap_data.index = [i.title() for i in heatmap_data.index]

    fig4, ax4 = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, cmap="YlOrRd", cbar_kws={'label': 'Loss (Ha)'}, ax=ax4)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("District")
    st.pyplot(fig4)

    # Raw Data View
    if st.checkbox("Show Raw Data Table"):
        st.write(dist_df)

else:
    st.info("👋 Welcome! Please upload the 'Subnational_2_tree_cover_loss.csv' file from the sidebar to start the analysis.")