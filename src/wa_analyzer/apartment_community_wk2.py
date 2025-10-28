import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import json

from logger_setup import LoggerSetup
from config_loader import ConfigLoader

# ====================================================
# --- Emoji Usage Analysis by Floor ---
# ====================================================
class EmojiByFloorAnalysis:
    """Analyze and visualize emoji usage per floor in WhatsApp data."""

    def __init__(self, df: pd.DataFrame, img_dir: Path):
        self.df = df
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def plot_emoji_usage(self):
        """Create a bar plot showing emoji usage per floor."""
        logger.info("Preparing emoji usage data per floor...")

        if "Floor_nr" not in self.df.columns or "has_emoji" not in self.df.columns:
            logger.error("Required columns 'Floor_nr' or 'has_emoji' are missing in the DataFrame.")
            return
        
        # --- Round Floor_nr to integer ---
        # self.df["Floor_nr"] = self.df["Floor_nr"].astype(int)

        self.df = self.df[self.df["Floor_nr"].notna()]
        self.df["Floor_nr"] = self.df["Floor_nr"].astype(int)

        # Group by floor and count messages with emojis
        emoji_by_floor = (
            self.df.groupby("Floor_nr")["has_emoji"]
            .sum()
            .reset_index()
            .rename(columns={"has_emoji": "emoji_count"})
            .sort_values("Floor_nr")
        )

        if emoji_by_floor.empty:
            logger.warning("No emoji data found in dataset.")
            return

        # Identify floor with highest emoji count
        max_floor = emoji_by_floor.loc[emoji_by_floor["emoji_count"].idxmax(), "Floor_nr"]
        max_value = emoji_by_floor["emoji_count"].max()

        # Set color palette (highlight max)
        colors = ["red" if floor == max_floor else "#86b9e0" for floor in emoji_by_floor["Floor_nr"]]

        # --- Plot ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=emoji_by_floor, x="Floor_nr", y="emoji_count", palette=colors)

        plt.title("Aantal berichten met emoji per verdieping", fontsize=16, fontweight="bold")
        plt.xlabel("Verdieping", fontsize=12)
        plt.ylabel("Aantal berichten met emoji", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Annotate highest bar
        plt.text(
            x=emoji_by_floor.index[emoji_by_floor["Floor_nr"] == max_floor][0],
            y=max_value + (max_value * 0.02),
            s=f"Hoogste verdieping {max_floor}",
            ha="center",
            color="red",
            fontweight="bold"
        )

        # --- Save plot ---
        save_path = self.img_dir / "wk2_emoji_verdieping_comparing_categories.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved emoji usage plot to {save_path}")


# ====================================================
# --- Data Handler ---
# ====================================================
class DataHandler:
    """Load WhatsApp data and resident metadata."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config["raw"])
        self.processed_path = Path(config["processed"])
        self.meta_path = Path(config["meta"])
        self.datafile = Path(config["processed"]) / config["current"]
        self.resident_meta_file = Path(config["meta"]) / config["resident_metadata"]

    def load_data(self):
        """Load WhatsApp parquet and resident metadata, return merged DataFrame."""
        # Load WhatsApp data
        df = pd.read_parquet(self.datafile)
        logger.info(f"Loaded WhatsApp data with {len(df)} messages.")

        # Ensure has_emoji column exists
        if "has_emoji" not in df.columns:
            df["has_emoji"] = df["content"].str.contains(r"[\U0001F300-\U0001FAFF]", regex=True).astype(int)
            logger.info("'has_emoji' column created based on message content.")

        # Load resident metadata
        with open(self.resident_meta_file, "r", encoding="utf-8") as f:
            resident_metadata = json.load(f)

        author_info_df = pd.DataFrame.from_dict(resident_metadata, orient="index")
        author_info_df.reset_index(inplace=True)
        author_info_df.rename(columns={"index": "author"}, inplace=True)

        # Merge WhatsApp data with metadata
        df_merged = df.merge(author_info_df, on="author", how="left")
        logger.info(f"Merged dataset has {len(df_merged)} messages from {df_merged['author'].nunique()} authors.")
        return df_merged


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze emoji usage by floor")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config).resolve()
    config = ConfigLoader(config_path).load()

    # Setup logger
    log_filename = "wk2_emoji_by_floor.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # Load data
    data_handler = DataHandler(config)
    df_merged = data_handler.load_data()

    # Run analysis
    img_dir = Path.cwd() / "img"
    analysis = EmojiByFloorAnalysis(df_merged, img_dir)
    analysis.plot_emoji_usage()

    logger.info("Emoji usage analysis complete.")


if __name__ == "__main__":
    main()
