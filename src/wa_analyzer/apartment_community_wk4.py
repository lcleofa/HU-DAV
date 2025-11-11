# %%
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import tomllib
from loguru import logger
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from logger_setup import LoggerSetup
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler


# ====================================================
# --- Question Message Length Analysis ---
# ====================================================
class QuestionLengthAnalysis:
    """Analyze and visualize message length distributions for male vs. female authors.

    This class performs a gender-based analysis of question message lengths.
    It filters the dataset to include only the top N authors, enriches the data with
    author metadata (name and gender), and generates visualizations comparing
    male and female authors based on the length of their question messages.

    Attributes:
        df (pd.DataFrame): The main dataframe containing message data.
        author_info_df (pd.DataFrame): Dataframe containing author metadata.
        img_dir (Path): Directory where plots will be saved.
        top_n (int): Number of top authors to analyze.
    """

    def __init__(self, df: pd.DataFrame, author_info_df: pd.DataFrame, img_dir: Path, top_n: int, max_length: int):
        self.df = df.copy()
        self.author_info_df = author_info_df.copy()
        self.top_n = top_n
        self.max_length = max_length  # store max_length
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)


    def prepare_data(self):
        """Prepare and clean the dataset for question length analysis.

        Steps:
            1. Select the top N authors based on message count.
            2. Map author metadata (name and gender) into the dataset.
            3. Filter only messages that end with a question mark ("?").
            4. Remove outliers with message lengths greater than 800.
            5. Add a log-transformed message length column for distribution analysis.

        Returns:
            None: Modifies the instance variable `self.df_question` with filtered data.
        """
        logger.info(f"Selecting top {self.top_n} authors by message count...")

        # --- Top N authors ---
        top_authors = self.df["author"].value_counts().head(self.top_n).index.tolist()
        df_top = self.df[self.df["author"].isin(top_authors)].copy()

        # --- Map metadata ---
        author_map = dict(zip(self.author_info_df["author"], self.author_info_df["Name"]))
        gender_map = dict(zip(self.author_info_df["author"], self.author_info_df["Gender"]))

        df_top["author_name"] = df_top["author"].map(author_map)
        df_top["author_name"].fillna(df_top["author"], inplace=True)

        df_top["author_gender"] = df_top["author"].map(gender_map)
        df_top["author_gender"].fillna("Unknown", inplace=True)

        # --- Keep only messages that end with a question mark ---
        df_question = df_top[df_top["message"].str.endswith("?")].copy()
        logger.info(f"Filtered {len(df_question)} question messages.")

        # --- Filter extreme lengths ---
        df_question = df_question[df_question["msg_length"] <= self.max_length]
        df_question["log_msg_length"] = df_question["msg_length"].apply(lambda x: np.log(x + 1))

        self.df_question = df_question

    def plot_hist_by_gender(self):
        """Plot and save a histogram of question message lengths by gender with mean & std in legend."""
        if not hasattr(self, "df_question"):
            raise ValueError("Data not prepared. Run prepare_data() first.")

        logger.info("Plotting question message length distribution by gender...")
        plt.figure(figsize=(14, 8))

        # --- Filter male & female ---
        df_male = self.df_question[self.df_question["author_gender"] == "Male"]
        df_female = self.df_question[self.df_question["author_gender"] == "Female"]

        # --- Calculate stats ---
        male_mean = df_male["msg_length"].mean()
        male_std = df_male["msg_length"].std()
        female_mean = df_female["msg_length"].mean()
        female_std = df_female["msg_length"].std()

        # --- Plot histograms ---
        sns.histplot(df_male["msg_length"], bins=30, color="blue", stat="count", alpha=0.5)
        sns.histplot(df_female["msg_length"], bins=30, color="red", stat="count", alpha=0.5)

        # --- Add mean lines ---
        plt.axvline(male_mean, color="blue", linestyle="--", linewidth=2)
        plt.axvline(female_mean, color="red", linestyle="--", linewidth=2)

        #--- Create legend with stats ---
        legend_labels = [
            f"Geslacht (Top {self.top_n} auteurs)",
            f"Gem. Mannen (μ={male_mean:.1f}, σ={male_std:.1f})",
            f"Gem. Vrouwen (μ={female_mean:.1f}, σ={female_std:.1f})",
            "Mannelijke Auteurs",
            "Vrouwelijke Auteurs",
        ]

        plt.legend(
            legend_labels[1:],  # skip the title from being treated as a label
            title=legend_labels[0],
            loc="upper right"
        )

        # --- Titles & labels ---
        plt.title(
            "Wie stelt de langste vraag in een flatgebouw chatgroep? Hint: meestal niet de mannen… ",
            fontsize=16
        )
        plt.xlabel("Lengte Bericht (aantal tekens)")
        plt.ylabel("Aantal Berichten")
        plt.tight_layout()

        # --- Save plot ---
        save_path = self.img_dir / "wk4_distributie_vraaglengtes_man_vrouw.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


def main():
    """Main entry point for the question length analysis script."""
    # --- Load config first ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- Get top_n from config (must exist there) ---
    top_n = config["Analysis"]["top_n"]

    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Analyze question message lengths by gender")
    parser.add_argument("--top", type=int, help="Number of top authors to analyze (overrides config)")
    args = parser.parse_args()

    # --- Determine final top_n ---
    if args.top is not None:
        top_n = args.top

    # --- Setup logger ---
    log_filename = "wk4_question_length.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # --- Load data ---
    data_handler = DataHandler(config)
    logger.info(f"Loading data from {data_handler.datafile}")
    df, author_info_df = data_handler.load_data()

    # --- Prepare image directory from config ---
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Image directory set to: {img_dir}")

    # --- Run analysis ---
    logger.info(f"Running analysis for top_n={top_n}")

    # Fetch max_question_length from config
    max_length = config["Analysis"]["max_question_length"]  

    # Pass it to the class
    analysis = QuestionLengthAnalysis(df, author_info_df, img_dir, top_n=top_n, max_length=max_length)

    analysis.prepare_data()
    analysis.plot_hist_by_gender()

    logger.info("Question length analysis complete.")



if __name__ == "__main__":
    main()
