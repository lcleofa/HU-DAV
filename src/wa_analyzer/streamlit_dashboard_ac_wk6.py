# streamlit_stylometry_dashboard.py
"""
Streamlit Dashboard for Stylometric Analysis of WhatsApp Messages (Exclamation Marks)
-----------------------------------------------------------------------------------
Allows residents to explore writing-style clusters interactively.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from loguru import logger
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler
from apartment_community_wk6 import ExclamationStylometry  # your class file name

sns.set_theme(style="whitegrid")

# ====================================================
# --- Streamlit App ---
# ====================================================

st.title("Stylometric Analysis: Exclamation Mark (!) Messages")

# --- Load config ---
CONFIG_PATH = "config.toml"
config_path = Path(CONFIG_PATH).resolve()
config = ConfigLoader(config_path).load()

# --- Sidebar Controls ---
st.sidebar.header("🔧 Parameters")
st.sidebar.markdown("Adjust analysis and visualization parameters.")

xmin = st.sidebar.slider("xmin", min_value=0, max_value=100000, value=config["Stylometry"]["xmin"], step=500)
xmax = st.sidebar.slider("xmax", min_value=0, max_value=100000, value=config["Stylometry"]["xmax"], step=500)
ymin = st.sidebar.slider("ymin", min_value=-20000, max_value=20000, value=config["Stylometry"]["ymin"], step=100)
ymax = st.sidebar.slider("ymax", min_value=-20000, max_value=20000, value=config["Stylometry"]["ymax"], step=100)

outlier_threshold_x = st.sidebar.selectbox(
    "Outlier Threshold (PCA1)",
    options=[20000, 40000, 60000, 80000, 100000, 120000, 140000],
    index=[20000, 40000, 60000, 80000, 100000, 120000, 140000].index(
        config["Stylometry"]["outlier_threshold_x"]
    ),
)

sample_n = st.sidebar.selectbox(
    "Number of sample messages per author",
    options=[5, 10, 15, 20],
    index=[5, 10, 15, 20].index(config["Stylometry"]["sample_n"]),
)

# --- Load and preprocess data ---
st.sidebar.markdown("### 📂 Data Loading")
data_handler = DataHandler(config)
df, _ = data_handler.load_data()

img_dir = Path(config["Images"]["imgdir"]).resolve()
img_dir.mkdir(parents=True, exist_ok=True)
analysis = ExclamationStylometry(df, img_dir, config)

df_ex = analysis.filter_exclamation_messages()
author_texts = analysis.compute_stylometry(df_ex)
author_texts = author_texts[author_texts["x"] <= outlier_threshold_x]

# --- PCA Cluster Plot ---
st.subheader("💬 Stylometric Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=author_texts,
    x="x",
    y="y",
    hue="author",
    palette="tab10",
    s=80,
    edgecolor="black",
    legend=False,  # ❌ hide legend for main plot
    ax=ax,
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Stylometric Clustering of Exclamation (!) Messages per Author")
sns.despine()
st.pyplot(fig)

# --- Cluster Region Selection ---
st.subheader("🔍 Inspect Cluster Region")
st.markdown("Use the sliders in the sidebar to define a focus region (xmin/xmax/ymin/ymax).")

cluster_authors_df = author_texts[
    (author_texts["x"] > xmin)
    & (author_texts["x"] < xmax)
    & (author_texts["y"] > ymin)
    & (author_texts["y"] < ymax)
].copy()

if cluster_authors_df.empty:
    st.warning("No authors found in this coordinate range.")
else:
    st.success(f"{len(cluster_authors_df)} authors found in this region.")
    st.dataframe(cluster_authors_df[["author", "x", "y"]])

    # --- Zoomed-in Plot with Legend ---
    fig_zoom, ax_zoom = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=author_texts,
        x="x",
        y="y",
        hue="author",
        alpha=0.3,
        legend=False,  # no legend for background points
        ax=ax_zoom,
    )
    sns.scatterplot(
        data=cluster_authors_df,
        x="x",
        y="y",
        hue="author",
        s=120,
        edgecolor="black",
        legend=True,  # ✅ show legend for zoomed authors
        ax=ax_zoom,
    )
    plt.xlim(xmin - 1000, xmax + 1000)
    plt.ylim(ymin - 500, ymax + 500)
    plt.title("Zoomed-in Cluster Region (Selected Authors Only)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Make the legend clearer and outside the plot
    plt.legend(
        title="Authors",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
        title_fontsize=10,
    )

    st.pyplot(fig_zoom)
