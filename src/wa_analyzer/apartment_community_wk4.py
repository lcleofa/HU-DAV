# %%
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tomllib
import json
from loguru import logger
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from logger_setup import LoggerSetup
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler


class MessageAnalysis:
    def __init__(self, df: pd.DataFrame, author_info_df: pd.DataFrame, img_dir: Path, top_n: int):
        self.df = df.copy()
        self.author_info_df = author_info_df.copy()
        self.top_n = top_n
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)  # make sure folder exists

    def prepare_data(self):
        # Compute top N authors
        top_authors = self.df['author'].value_counts().head(self.top_n).index.tolist()
        self.df_top = self.df[self.df['author'].isin(top_authors)].copy()

        # Map author names and gender
        author_map = dict(zip(self.author_info_df['author'], self.author_info_df['Name']))
        gender_map = dict(zip(self.author_info_df['author'], self.author_info_df['Gender']))

        self.df_top['author_name'] = self.df_top['author'].map(author_map)
        self.df_top['author_name'].fillna(self.df_top['author'], inplace=True)

        self.df_top['author_gender'] = self.df_top['author'].map(gender_map)
        self.df_top['author_gender'].fillna('Unknown', inplace=True)

        # Filter extreme message lengths
        self.df_top = self.df_top[self.df_top['msg_length'] <= 800]
        self.df_top['log_msg_length'] = self.df_top['msg_length'].apply(lambda x: np.log(x + 1))

    def plot_kde_by_gender(self):
        plt.figure(figsize=(14, 8))

        gender_colors = {'Male': 'blue', 'Female': 'red', 'Unknown': 'gray'}
        gender_linestyles = {'Male': '--', 'Female': '-', 'Unknown': '--'}
        gender_labels_dutch = {'Male': 'Man', 'Female': 'Vrouw', 'Unknown': 'Onbekend'}

        legend_added = set()
        for author in self.df_top['author_name'].unique():
            author_data = self.df_top[self.df_top['author_name'] == author]
            gender = author_data['author_gender'].iloc[0]
            color = gender_colors.get(gender, 'gray')
            linestyle = gender_linestyles.get(gender, '--')
            label = gender_labels_dutch.get(gender, 'Onbekend') if gender not in legend_added else None
            if label:
                legend_added.add(gender)

            sns.kdeplot(
                data=author_data,
                x='log_msg_length',
                fill=False,
                common_norm=False,
                linestyle=linestyle,
                linewidth=2,
                color=color,
                label=label
            )

        plt.legend(title="Geslacht", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Distributie berichtlengtes van top {self.top_n} auteurs: mannelijke auteurs versturen gemiddeld iets langere berichten.", fontsize=16)
        plt.xlabel("Log(Lengte bericht)")
        plt.ylabel("Dichtheid")
        plt.tight_layout()
        # plt.show()

                # --- Save plot ---
        save_path = self.img_dir / f"wk4_log_berichtlengtes_distributions.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


def main():

    # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # Setup logger from utilities (with custom filename)
    log_filename = "wk4_distributions.log"
    logger_obj = LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")


    # Command-line arguments
    parser = argparse.ArgumentParser(description="Analyze top authors by message length")
    parser.add_argument("--top", type=int, default=10, help="Number of top authors to analyze")
    args = parser.parse_args()

    # --- Load data and metadata ---
    data_handler = DataHandler(config)
    df, author_info_df = data_handler.load_data()

    img_dir = Path.cwd() / "img"

    # Message analysis
    analysis = MessageAnalysis(df, author_info_df, img_dir, top_n=args.top)
    analysis.prepare_data()
    analysis.plot_kde_by_gender()
    logger.info("Done!")


if __name__ == "__main__":
    main()
