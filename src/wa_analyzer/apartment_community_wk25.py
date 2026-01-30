"""
WhatsApp Chat Analysis – Emoji Usage per Board Function
======================================================

This script computes the percentage of WhatsApp messages
containing emojis per board function and visualizes the result.

Output:
- Bar chart:
  img/wk2_percentage_messages_with_emojis_by_board_function.png
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from logger_setup import LoggerSetup
from config_loader import ConfigLoader

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
sns.set_theme(style="whitegrid")


# ====================================================
# --- Utility functions ---
# ====================================================
def clean_message(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ====================================================
# --- Data Handler ---
# ====================================================
class DataHandler:
    """Load and merge WhatsApp data with resident metadata."""

    def __init__(self, config: dict):
        self.datafile = Path(config["processed"]) / config["current"]
        self.meta_file = Path(config["meta"]) / config["resident_metadata"]

    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.datafile)
        logger.info(f"Loaded WhatsApp data: {len(df)} messages")

        if "has_emoji" not in df.columns:
            df["has_emoji"] = df["message"].str.contains(
                r"[\U0001F300-\U0001FAFF]", regex=True
            ).astype(int)
            logger.info("'has_emoji' column created")

        with open(self.meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        author_info_df = (
            pd.DataFrame.from_dict(metadata, orient="index")
            .reset_index()
            .rename(columns={"index": "author"})
        )

        df_merged = df.merge(author_info_df, on="author", how="left")
        logger.info(
            f"Merged dataset: {len(df_merged)} messages, "
            f"{df_merged['author'].nunique()} authors"
        )

        return df_merged


# ====================================================
# --- Analysis ---
# ====================================================
class BoardFunctionEmojiAnalysis:
    """Compute and visualize emoji usage per board function."""

    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df.copy()
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def compute_percentage_per_board_function(self) -> pd.DataFrame:
        df = self.df[self.df["Board_function"].str.lower() != "not_applicable"]

        df_grouped = (
            df.groupby("Board_function", as_index=False)
            .agg(
                total_messages=("message", "count"),
                messages_with_emojis=("has_emoji", "sum"),
            )
        )

        df_grouped["percentage_messages_with_emojis"] = (
            df_grouped["messages_with_emojis"]
            / df_grouped["total_messages"]
            * 100
        ).round(2)

        df_grouped.sort_values(
            "percentage_messages_with_emojis",
            ascending=False,
            inplace=True,
        )

        logger.info("Computed emoji percentage per board function")
        return df_grouped

    def plot(self):
        df_plot = self.compute_percentage_per_board_function()
        if df_plot.empty:
            logger.warning("No data available for plotting")
            return

        # Identify highest board function
        max_value = df_plot["percentage_messages_with_emojis"].max()
        max_function = df_plot.loc[
            df_plot["percentage_messages_with_emojis"] == max_value,
            "Board_function",
        ].iloc[0]

        colors = [
            "red" if bf == max_function else "navy"
            for bf in df_plot["Board_function"]
        ]

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            data=df_plot,
            x="Board_function",
            y="percentage_messages_with_emojis",
            palette=colors,
        )

        ax.set_title(
            "Percentage berichten met emoji’s per bestuursfunctie in WhatsApp flatgebouw",
            fontsize=14,
        )
        ax.set_xlabel("Bestuursfunctie")
        ax.set_ylabel("Percentage berichten met emoji’s (%)")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = (
            self.img_dir
            / "wk2_percentage_berichten_met_emojis_per_bestuursfunctie.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved to {output_path}")


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze emoji usage per board function"
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config file"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = ConfigLoader(config_path).load()

    log_filename = "wk2_percentage_messages_with_emojis_by_board_function.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized")

    df_merged = DataHandler(config).load_data()
    img_dir = Path(config["Images"]["imgdir"]).resolve()

    analysis = BoardFunctionEmojiAnalysis(df_merged, img_dir)
    analysis.plot()

    logger.info("Emoji percentage analysis by board function complete")


if __name__ == "__main__":
    main()
