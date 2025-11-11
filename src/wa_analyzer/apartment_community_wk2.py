# %%
"""
WhatsApp Chat Analysis by Board Function
========================================

This script analyzes WhatsApp chat data to identify board role communication patterns:
either by **emoji usage** or **average message length**.

It performs the following:
1. Loads pre-processed WhatsApp message data and resident metadata.
2. Merges both datasets based on the message author.
3. Calculates metrics (emoji usage or message length) per author and board role.
4. Identifies the top author for each board role.
5. Visualizes the results as a bar plot, highlighting the board role with the highest value.

Outputs:
    - A bar chart image: `img/wk2_<metric>_by_board_function.png`
    - A log file: `wk2_<metric>_by_board_function.log`

Usage:
    python emoji_usage_analysis.py --config config.toml --metric emoji
    python emoji_usage_analysis.py --config config.toml --metric length
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from logger_setup import LoggerSetup
from config_loader import ConfigLoader

sns.set_theme(style="whitegrid", palette="muted")


# ====================================================
# --- Board Function Analysis (Emoji / Message Length) ---
# ====================================================
class BoardFunctionAnalysis:
    """
    Analyze and visualize WhatsApp chat metrics per board function.

    Attributes:
        df (pd.DataFrame): Merged WhatsApp data and metadata.
        img_dir (Path): Directory where the plot image will be saved.
        metric (str): Metric to analyze ('emoji' or 'length').
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, metric: str):
        """
        Initialize the BoardFunctionAnalysis class.

        Args:
            df (pd.DataFrame): Input DataFrame containing message data and board function info.
            img_dir (Path): Directory to save generated plots.
            metric (str): Metric to analyze ('emoji' or 'length').
        """
        self.df = df.copy()
        self.img_dir = img_dir
        self.metric = metric.lower()
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def compute_metric(self):
        """
        Compute the relevant metric (emoji usage or average message length)
        grouped by author and Board_function.
        """
        # Ensure required columns exist
        required_cols = {"author", "Board_function", "message"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        # Filter out non-applicable board roles
        df_filtered = self.df[self.df["Board_function"].str.lower() != "not_applicable"]
        logger.info(f"Filtered dataset to {len(df_filtered)} messages with valid Board_function values.")

        # Metric-specific computation
        if self.metric == "emoji":
            if "has_emoji" not in df_filtered.columns:
                df_filtered["has_emoji"] = df_filtered["message"].str.contains(
                    r"[\U0001F300-\U0001FAFF]", regex=True
                ).astype(int)
            metric_df = (
                df_filtered.groupby(["Board_function", "author"])["has_emoji"]
                .sum()
                .reset_index()
                .rename(columns={"has_emoji": "metric_value"})
            )

        elif self.metric == "length":
            df_filtered["message_length"] = df_filtered["message"].str.len()
            metric_df = (
                df_filtered.groupby(["Board_function", "author"])["message_length"]
                .mean()
                .reset_index()
                .rename(columns={"message_length": "metric_value"})
            )
        else:
            logger.error(f"Invalid metric '{self.metric}'. Must be 'emoji' or 'length'.")
            return pd.DataFrame()

        return metric_df

    def plot_metric(self):
        """
        Generate and save a plot of the selected metric per board function.
        """
        metric_df = self.compute_metric()
        if metric_df.empty:
            logger.warning("No data available for plotting.")
            return

        # Identify top author per Board_function
        top_per_function = (
            metric_df.loc[metric_df.groupby("Board_function")["metric_value"].idxmax()]
            .reset_index(drop=True)
        )

        # Sort by metric value
        top_sorted = top_per_function.sort_values("metric_value", ascending=False)

        # Identify max
        max_idx = top_sorted["metric_value"].idxmax()
        max_function = top_sorted.loc[max_idx, "Board_function"]
        max_value = top_sorted.loc[max_idx, "metric_value"]

        # Color highlight
        colors = [
            "red" if bf == max_function else sns.color_palette("muted")[0]
            for bf in top_sorted["Board_function"]
        ]

        # Plot
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(
            data=top_sorted,
            x="Board_function",
            y="metric_value",
            palette=colors
        )

        # Title & labels
        if self.metric == "emoji":
            title = "Emojigebruik per bestuursfunctie in Flatgebouw-chat: Commissielid domineert!"
            ylabel = "Aantal Emojies"
            file_suffix = "emoji"
        else:  # length
            title = "Berichtlengte per bestuursfunctie: Algemeen bestuurslid tekst gemiddeld het meest in Flatgebouw-chat"
            ylabel = "Gemiddelde Berichtlengte"
            file_suffix = "berichtlengte"

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Bestuursfunctie", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        # Save
        filename = f"wk2_{file_suffix}_bestuursfunctie_comparing_categories.png"
        save_path = self.img_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


# ====================================================
# --- Data Handler ---
# ====================================================
class DataHandler:
    """Load and merge WhatsApp data with resident metadata."""

    def __init__(self, config: dict):
        self.config = config
        self.datafile = Path(config["processed"]) / config["current"]
        self.resident_meta_file = Path(config["meta"]) / config["resident_metadata"]

    def load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.datafile)
        logger.info(f"Loaded WhatsApp data with {len(df)} messages.")

        # has_emoji flag if missing
        if "has_emoji" not in df.columns:
            df["has_emoji"] = df["message"].str.contains(
                r"[\U0001F300-\U0001FAFF]", regex=True
            ).astype(int)
            logger.info("'has_emoji' column created based on message content.")

        # Load metadata
        with open(self.resident_meta_file, "r", encoding="utf-8") as f:
            resident_metadata = json.load(f)

        author_info_df = pd.DataFrame.from_dict(resident_metadata, orient="index").reset_index()
        author_info_df.rename(columns={"index": "author"}, inplace=True)

        df_merged = df.merge(author_info_df, on="author", how="left")
        logger.info(f"Merged dataset: {len(df_merged)} messages, {df_merged['author'].nunique()} authors.")
        return df_merged


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze WhatsApp data per Board_function")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--metric", type=str, choices=["emoji", "length"], default="emoji", help="Metric to analyze")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config).resolve()
    config = ConfigLoader(config_path).load()

    # Logger setup
    log_filename = f"wk2_{args.metric}_by_board_function.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # Load & merge data
    df_merged = DataHandler(config).load_data()

    # Image directory from config
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analysis = BoardFunctionAnalysis(df_merged, img_dir, args.metric)
    analysis.plot_metric()

    logger.info(f"{args.metric.capitalize()} analysis by Board_function complete.")


if __name__ == "__main__":
    main()
