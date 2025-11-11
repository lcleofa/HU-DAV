# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from config_loader import ConfigLoader
from apartment_community_wk5 import DataHandler_merge_meta, ElevatorAnalysis  # your classes

sns.set_theme(style="whitegrid")

# --- Dashboard title ---
st.title("WhatsApp Chat Analyse per Verdieping in Flatgebouw")

# --- Load config ---
CONFIG_PATH = "config.toml"
config_path = Path(CONFIG_PATH).resolve()
config = ConfigLoader(config_path).load()

# --- Load & merge data ---
data_handler = DataHandler_merge_meta(config)
df, author_info_df = data_handler.load_data()

# --- Sidebar for user selection ---
st.sidebar.header("Selecteer Keyword")

# Load keywords dynamically from config
allowed_keywords = config["Analysis"]["keywords"]

selected_keyword = st.sidebar.radio(
    "Selecteer een keyword:",
    allowed_keywords,
    index=0
)

# Optional: allow image directory from config
img_dir = Path(config["Images"]["imgdir"]).resolve()

# --- Analysis ---
analysis = ElevatorAnalysis(df, selected_keyword, img_dir)
analysis.preprocess()
analysis.aggregate_by_floor()

if analysis.floor_stats.empty:
    st.warning("Geen gegevens beschikbaar voor dit keyword.")
else:
    # --- Plot ---
    # st.subheader(f"Aantal berichten met '{selected_keyword}' per verdieping")
    st.subheader(f"Hogere verdiepingen praten vaker over de 'lift' in een flatgebouw chat")

    # Correlation info
    corr_msgs, corr_length = analysis.correlation_analysis()
    st.markdown(f"**Correlatie Etage vs Aantal berichten met '{selected_keyword}':** {corr_msgs:.2f}")
    st.markdown(f"**Correlatie Etage vs Gemiddelde berichtlengte:** {corr_length:.2f}")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.regplot(
        data=analysis.floor_stats,
        x="Floor_nr",
        y="keyword_msgs",
        scatter_kws={"s": 80},
        line_kws={"color": "red"},
        ax=ax
    )
    ax.set_xlabel("Etage nummer")
    ax.set_ylabel(f"Aantal berichten met '{selected_keyword}'")
    ax.set_title(f"Verdiepingen vs berichten met '{selected_keyword}'", fontsize=14, fontweight="bold")
    ax.set_ylim(0)
    sns.despine()
    st.pyplot(fig)
