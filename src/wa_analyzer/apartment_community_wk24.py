"""
WhatsApp Chat Analysis by Board Member
======================================

This script analyzes WhatsApp chat data to compute the
**percentage of messages containing emojis per board member**.

Steps:
1. Load preprocessed WhatsApp message data.
2. Load resident metadata.
3. Merge both datasets on author.
4. Clean message text.
5. Compute emoji usage per author.
6. Visualize results as a bar plot, highlighting the highest category.

Outputs:
- Bar chart image: img/wk2_percentage_messages_with_emojis_by_board_member.png
- Log file: wk2_percentage_messages_with_emojis_by_board_member.log

Usage:
    python emoji_percentage_by_board_member.py --config config.toml
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
    """
    Clean message text by removing links, newlines and excess whitespace.

    Args:
        text (str): Raw message text.

    Returns:
        str: Cleaned, lowercased message.
    """
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
        # Load WhatsApp data
        df = pd.read_parquet(self.datafile)
        logger.info(f"Loaded WhatsApp data: {len(df)} messages")

        # Ensure emoji flag exists
        if "has_emoji" not in df.columns:
            df["has_emoji"] = df["message"].str.contains(
                r"[\U0001F300-\U0001FAFF]", regex=True
            ).astype(int)
            logger.info("'has_emoji' column created")

        # Load metadata
        with open(self.meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        author_info_df = (
            pd.DataFrame.from_dict(metadata, orient="index")
            .reset_index()
            .rename(columns={"index": "author"})
        )

        # Merge
        df_merged = df.merge(author_info_df, on="author", how="left")
        logger.info(
            f"Merged dataset: {len(df_merged)} messages, "
            f"{df_merged['author'].nunique()} authors"
        )

        return df_merged


# ====================================================
# --- Analysis ---
# ====================================================
class BoardMemberEmojiPercentageAnalysis:
    """
    Compute and visualize the percentage of messages containing emojis
    per board member.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df.copy()
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def compute_percentage_per_author(self) -> pd.DataFrame:
        """
        Compute percentage of messages with emojis per author.

        Returns:
            pd.DataFrame: Aggregated results at author level.
        """
        df = self.df[self.df["Board_function"].str.lower() != "not_applicable"]

        df_by_author = (
            df.groupby(["author", "Board_function"], as_index=False)
            .agg(
                total_messages=("message", "count"),
                total_messages_with_emojis=("has_emoji", "sum")
            )
        )

        df_by_author["percentage_messages_with_emojis"] = (
            df_by_author["total_messages_with_emojis"] / df_by_author["total_messages"] * 100
        ).round(2)

        # Sort descending by percentage
        df_by_author.sort_values("percentage_messages_with_emojis", ascending=False, inplace=True)

        logger.info("Computed percentage of messages with emojis per author")
        return df_by_author

    def plot(self):
        """
        Generate and save the seaborn barplot (one bar per board member),
        highlighting the highest emoji usage in red.
        """
        df_plot = self.compute_percentage_per_author()
        if df_plot.empty:
            logger.warning("No data available for plotting")
            return

        # Identify highest author
        max_value = df_plot["percentage_messages_with_emojis"].max()
        max_author = df_plot.loc[
            df_plot["percentage_messages_with_emojis"] == max_value, "author"
        ].iloc[0]

        # Colors: red for highest, navy for others
        colors = ["red" if auth == max_author else "navy" for auth in df_plot["author"]]

        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df_plot,
            x="author",
            y="percentage_messages_with_emojis",
            palette=colors
        )

        # Add board function as label above bars
        for i, row in df_plot.iterrows():
            ax.text(
                i,
                row["percentage_messages_with_emojis"] + 0.5,
                row["Board_function"],
                ha="center",
                fontsize=9,
                rotation=90,
                color="black"
            )

        ax.set_title("Percentage berichten met emoji’s per bestuurslid", fontsize=14)
        ax.set_xlabel("Board Member")
        ax.set_ylabel("Messages with Emojis (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save figure
        output_path = self.img_dir / "wk2_percentage_messages_with_emojis_by_board_member.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved to {output_path}")


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze emoji usage per board member"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    config = ConfigLoader(config_path).load()

    # Logger setup
    log_filename = "wk2_percentage_messages_with_emojis_by_board_member.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized")

    # Load data
    df_merged = DataHandler(config).load_data()

    # Image directory
    img_dir = Path(config["Images"]["imgdir"]).resolve()

    # Run analysis
    analysis = BoardMemberEmojiPercentageAnalysis(df_merged, img_dir)
    analysis.plot()

    logger.info("Emoji percentage analysis by board member complete")


if __name__ == "__main__":
    main()
