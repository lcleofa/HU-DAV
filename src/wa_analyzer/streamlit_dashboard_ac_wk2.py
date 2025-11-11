# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from apartment_community_wk2 import DataHandler, BoardFunctionAnalysis
from config_loader import ConfigLoader

sns.set_theme(style="whitegrid", palette="muted")

# --- Dashboard title ---
st.title("WhatsApp Chat Analyse per Bestuursfunctie in VVE Flatgebouw")

# --- Load config ---
CONFIG_PATH = "config.toml"
config_path = Path(CONFIG_PATH).resolve()
config = ConfigLoader(config_path).load()

# --- Load & merge data ---
df_merged = DataHandler(config).load_data()

# --- Sidebar for user selection ---
st.sidebar.header("Selecteer Parameters")

# --- Exit button ---
exit_app = st.sidebar.button("Sluit Applicatie")
if exit_app:
    st.warning("De applicatie is afgesloten. Vernieuw de pagina om opnieuw te starten.")
    st.stop()  # stops the rest of the script cleanly


# Metric selection
metric = st.sidebar.selectbox(
    "Selecteer te visualiseren maatstaf",
    ("emoji", "length"),
    format_func=lambda x: "Aantal Emojies" if x == "emoji" else "Berichtlengte",
    index=0
)

# --- Plotting ---
st.subheader(f"{'Aantal Emojies' if metric=='emoji' else 'Berichtlengte'} per Bestuursfunctie")

analysis = BoardFunctionAnalysis(df_merged, img_dir=Path("tmp"), metric=metric)

# Compute metric
metric_df = analysis.compute_metric()
if metric_df.empty:
    st.warning("Geen gegevens beschikbaar voor deze maatstaf.")
else:
    top_per_function = metric_df.loc[metric_df.groupby("Board_function")["metric_value"].idxmax()].reset_index(drop=True)
    top_sorted = top_per_function.sort_values("metric_value", ascending=False)
    max_idx = top_sorted["metric_value"].idxmax()
    max_function = top_sorted.loc[max_idx, "Board_function"]

    colors = ["red" if bf == max_function else sns.color_palette("muted")[0] for bf in top_sorted["Board_function"]]

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=top_sorted, x="Board_function", y="metric_value", palette=colors, ax=ax)

    ylabel = "Aantal Emojies" if metric=="emoji" else "Gemiddelde Berichtlengte"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Bestuursfunctie")
    title = (
        "Emojigebruik per bestuursfunctie: Commissielid domineert!"
        if metric=="emoji"
        else "Gemiddelde berichtlengte per bestuursfunctie: Algemeen bestuurslid schrijft het langst"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    sns.despine()
    st.pyplot(fig)

