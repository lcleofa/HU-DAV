"""
WhatsApp Chat Analysis by Board Function
=======================================

This script analyzes WhatsApp chat data to compute the
**percentage of messages containing emojis per board function**.

Steps:
1. Load preprocessed WhatsApp message data.
2. Load resident metadata.
3. Merge both datasets on author.
4. Clean message text.
5. Compute emoji usage per author.
6. Aggregate to board-function level.
7. Visualize results as a bar plot, highlighting the highest category.

Outputs:
- Bar chart image: img/wk2_percentage_messages_with_emojis_by_board_function.png
- Log file: wk2_percentage_messages_with_emojis_by_board_function.log

Usage:
    python emoji_percentage_by_board_function.py --config config.toml
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
    text = re.sub(r"http\\S+", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\\s+", " ", text).strip()
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
                r"[\\U0001F300-\\U0001FAFF]", regex=True
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
class BoardFunctionEmojiPercentageAnalysis:
    """
    Compute and visualize the percentage of messages containing emojis
    per board function.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df.copy()
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    # def compute_percentage(self) -> pd.DataFrame:
    #     """
    #     Compute percentage of messages with emojis per board function.

    #     Returns:
    #         pd.DataFrame: Aggregated results.
    #     """
    #     # Filter valid board functions
    #     df = self.df[self.df["Board_function"].str.lower() != "not_applicable"]

    #     # Per author aggregation
    #     df_by_author = (
    #         df.groupby("author", as_index=False)
    #         .agg(
    #             message_count=("author", "size"),
    #             Board_function=("Board_function", "first"),
    #             messages_with_emojies=("has_emoji", "sum"),
    #         )
    #     )

    #     # Aggregate to board-function level
    #     df_by_function = (
    #         df_by_author
    #         .groupby("Board_function", as_index=False)
    #         .agg(
    #             total_messages=("message_count", "sum"),
    #             total_messages_with_emojis=("messages_with_emojies", "sum"),
    #         )
    #     )

    #     df_by_function["percentage_messages_with_emojies"] = (
    #         df_by_function["total_messages_with_emojis"]
    #         / df_by_function["total_messages"]
    #         * 100
    #     ).round(2)

    #     df_by_function.sort_values(
    #         "percentage_messages_with_emojies",
    #         ascending=False,
    #         inplace=True
    #     )

    #     logger.info("Computed percentage of messages with emojis per board function")
    #     return df_by_function

    def compute_percentage(self) -> pd.DataFrame:
        """
        Compute percentage of messages with emojis per author.

        Returns:
            pd.DataFrame: Aggregated results at author level.
        """
        df = self.df[self.df["Board_function"].str.lower() != "not_applicable"]

        # Aggregate per author
        df_by_author = (
            df.groupby(["author", "Board_function"], as_index=False)
            .agg(
                total_messages=("message", "count"),
                messages_with_emojis=("has_emoji", "sum")
            )
        )

        df_by_author["percentage_messages_with_emojis"] = (
            df_by_author["messages_with_emojis"] / df_by_author["total_messages"] * 100
        ).round(2)

        df_by_author.sort_values(
            "percentage_messages_with_emojis",
            ascending=False,
            inplace=True
        )

        logger.info("Computed percentage of messages with emojis per author")
        return df_by_author


    # def plot(self):
    #     """
    #     Generate and save the seaborn barplot.
    #     """
    #     df_plot = self.compute_percentage()
    #     if df_plot.empty:
    #         logger.warning("No data available for plotting")
    #         return

    #     # Identify highest value
    #     max_value = df_plot["percentage_messages_with_emojies"].max()
    #     max_function = df_plot.iloc[0]["Board_function"]

    #     # Colors: red for highest, navy for others
    #     colors = [
    #         "red" if bf == max_function else "navy"
    #         for bf in df_plot["Board_function"]
    #     ]

    #     plt.figure(figsize=(10, 6))
    #     ax = sns.barplot(
    #         data=df_plot,
    #         x="Board_function",
    #         y="percentage_messages_with_emojies",
    #         palette=colors
    #     )

    #     ax.set_title(
    #         "Percentage berichten met emoji’s per bestuursfunctie",
    #         fontsize=14,
    #         fontweight="bold"
    #     )
    #     ax.set_xlabel("Bestuursfunctie")
    #     ax.set_ylabel("Berichten met emoji’s (%)")

    #     plt.tight_layout()

    #     output_path = self.img_dir / "wk2_percentage_messages_with_emojis_by_board_function.png"
    #     plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #     plt.close()

    #     logger.info(f"Plot saved to {output_path}")

    # def plot(self):
    #     """
    #     Generate and save the seaborn barplot (one bar per author, colored by board function).
    #     """
    #     df_plot = self.compute_percentage()
    #     if df_plot.empty:
    #         logger.warning("No data available for plotting")
    #         return

    #     plt.figure(figsize=(12, 6))
    #     ax = sns.barplot(
    #         data=df_plot,
    #         x="author",
    #         y="percentage_messages_with_emojis",
    #         hue="Board_function",
    #         dodge=False,
    #         palette="tab10"
    #     )

    #     ax.set_title(
    #         "Percentage berichten met emoji’s per author",
    #         fontsize=14,
    #         fontweight="bold"
    #     )
    #     ax.set_xlabel("Author")
    #     ax.set_ylabel("Berichten met emoji’s (%)")
    #     plt.xticks(rotation=45, ha="right")
    #     plt.legend(title="Board function")
    #     plt.tight_layout()

    #     output_path = self.img_dir / "wk2_Percentage_berichten_met_emoji_s_per_bestuursfunctie.png"
    #     plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #     plt.close()

    #     logger.info(f"Plot saved to {output_path}")

    # def plot(self):
    #     """
    #     Generate and save the seaborn barplot (one bar per author),
    #     highlighting the author with the highest emoji usage in red.
    #     """
    #     df_plot = self.compute_percentage()
    #     if df_plot.empty:
    #         logger.warning("No data available for plotting")
    #         return

    #     # Identify highest value
    #     max_value = df_plot["percentage_messages_with_emojis"].max()
    #     max_author = df_plot.iloc[0]["author"]  # already sorted descending

    #     # Colors: red for highest, navy for others
    #     colors = [
    #         "red" if author == max_author else "navy"
    #         for author in df_plot["author"]
    #     ]

    #     plt.figure(figsize=(12, 6))
    #     ax = sns.barplot(
    #         data=df_plot,
    #         x="author",
    #         y="percentage_messages_with_emojis",
    #         palette=colors
    #     )

    #     ax.set_title(
    #         "Percentage berichten met emoji’s per author",
    #         fontsize=14,
    #         fontweight="bold"
    #     )
    #     ax.set_xlabel("Author")
    #     ax.set_ylabel("Berichten met emoji’s (%)")
    #     plt.xticks(rotation=45, ha="right")
    #     ax.get_legend().remove() if ax.get_legend() else None  # remove legend if exists
    #     plt.tight_layout()

    #     output_path = self.img_dir / "wk2_Percentage_berichten_met_emoji_s_per_bestuursfunctie.png"
    #     plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #     plt.close()

    #     logger.info(f"Plot saved to {output_path}")

    def plot(self):
        """
        Generate and save the seaborn barplot (one bar per author),
        x-axis shows Board_function, highlighting the highest emoji usage in red.
        """
        df_plot = self.compute_percentage()
        if df_plot.empty:
            logger.warning("No data available for plotting")
            return

        # Identify highest value
        max_value = df_plot["percentage_messages_with_emojis"].max()
        max_author = df_plot.iloc[0]["author"]  # already sorted descending

        # Colors: red for highest, navy for others
        colors = [
            "red" if author == max_author else "navy"
            for author in df_plot["author"]
        ]

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df_plot,
            x="Board_function",  # x-axis shows Board_function
            y="percentage_messages_with_emojis",
            palette=colors
        )

        ax.set_title(
            "Percentage berichten met emoji’s per author",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Board function")
        ax.set_ylabel("Berichten met emoji’s (%)")
        plt.xticks(rotation=45, ha="right")
        ax.get_legend().remove() if ax.get_legend() else None  # remove legend
        plt.tight_layout()

        output_path = self.img_dir / "wk2_Percentage_berichten_met_emoji_s_per_bestuursfunctie.png"
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
    log_filename = "wk2_percentage_messages_with_emojis_by_board_function.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized")

    # Load data
    df_merged = DataHandler(config).load_data()

    # Image directory
    img_dir = Path(config["Images"]["imgdir"]).resolve()

    # Run analysis
    analysis = BoardFunctionEmojiPercentageAnalysis(df_merged, img_dir)
    analysis.plot()

    logger.info("Emoji percentage analysis by board function complete")


if __name__ == "__main__":
    main()
