# streamlit_dashboard_qna.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler
from apartment_community_wk4 import QuestionLengthAnalysis  # import your class

sns.set_theme(style="whitegrid", palette="muted")

# --- Dashboard title ---
st.title("Vraagbericht Lengte Analyse per Geslacht")

# --- Load config ---
CONFIG_PATH = "config.toml"
config_path = Path(CONFIG_PATH).resolve()
config = ConfigLoader(config_path).load()

# --- Load data ---
data_handler = DataHandler(config)
df, author_info_df = data_handler.load_data()

# --- Sidebar ---
st.sidebar.header("Selecteer Parameters")

# Slider for top N authors
top_n = st.sidebar.slider("Selecteer Top N auteurs", min_value=1, max_value=50, value=10, step=1)

# Gender selection
gender_option = st.sidebar.radio(
    "Selecteer distributie",
    ("Both", "Male", "Female"),
    format_func=lambda x: "Beide Geslachten" if x=="Both" else ("Mannelijk" if x=="Male" else "Vrouwelijk")
)

# --- Run analysis ---
# Use a dummy img_dir since we do not save in Streamlit
# Fetch max_question_length from config
max_length = config["Analysis"]["max_question_length"]  # no hardcoded 800

# Pass it to the class
analysis = QuestionLengthAnalysis(df, author_info_df, img_dir=Path("tmp"), top_n=top_n, max_length=max_length)
analysis.prepare_data()

# Filter by gender if needed
df_plot = analysis.df_question.copy()
if gender_option == "Male":
    df_plot = df_plot[df_plot["author_gender"] == "Male"]
elif gender_option == "Female":
    df_plot = df_plot[df_plot["author_gender"] == "Female"]

# --- Plot ---
st.subheader("Wie stelt de langste vraag in een flatgebouw chatgroep? Hint: meestal niet de mannen…")

fig, ax = plt.subplots(figsize=(12,6))
colors = {"Male":"blue", "Female":"red"}

# Plot histograms
for gender in df_plot["author_gender"].unique():
    data = df_plot[df_plot["author_gender"]==gender]["msg_length"]
    sns.histplot(
        data,
        bins=30,
        color=colors.get(gender, "gray"),
        alpha=0.5,
        ax=ax
    )

# Calculate statistics
male_data = df_plot[df_plot["author_gender"]=="Male"]["msg_length"] if "Male" in df_plot["author_gender"].unique() else pd.Series(dtype=float)
female_data = df_plot[df_plot["author_gender"]=="Female"]["msg_length"] if "Female" in df_plot["author_gender"].unique() else pd.Series(dtype=float)

male_mean, male_std = male_data.mean() if not male_data.empty else 0, male_data.std() if not male_data.empty else 0
female_mean, female_std = female_data.mean() if not female_data.empty else 0, female_data.std() if not female_data.empty else 0

# Add mean lines
if not male_data.empty:
    ax.axvline(male_mean, color="blue", linestyle="--", linewidth=2)
if not female_data.empty:
    ax.axvline(female_mean, color="red", linestyle="--", linewidth=2)

# Create legend like the original script
legend_labels = [
    f"Geslacht",
    f"Gem. Mannen (μ={male_mean:.1f}, σ={male_std:.1f})",
    f"Gem. Vrouwen (μ={female_mean:.1f}, σ={female_std:.1f})",
    "Mannelijke Auteurs",
    "Vrouwelijke Auteurs",
]

ax.legend(legend_labels[1:], title=legend_labels[0], loc="upper right")

# Titles and labels
ax.set_xlabel("Lengte Bericht (tekens)")
ax.set_ylabel("Aantal Berichten")
ax.set_title(f"Vraagbericht Lengte Distributie - Top {top_n} Auteurs")
sns.despine()

# Display figure in Streamlit
st.pyplot(fig)
