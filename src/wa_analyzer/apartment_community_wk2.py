import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import tomllib
from loguru import logger

from logger_setup import LoggerSetup

from config_loader import ConfigLoader

from data_handler_meta_data import DataHandler


# ====================================================
# --- Message Analysis ---
# ====================================================
class MessageAnalysis:
    """Analyze and visualize message trends for a keyword."""

    def __init__(self, df: pd.DataFrame, keyword: str, img_dir: Path):
        self.df = df
        self.keyword = keyword
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def plot_trend(self):
        """Plot keyword trends over time."""
        keyword_df = self.df[self.df["message"].str.contains(self.keyword, case=False)]

        trend_df = keyword_df.set_index("timestamp").resample("M").size().reset_index(name="count")

        if trend_df.empty:
            logger.warning(f"No messages found for keyword '{self.keyword}'.")
            return

        max_idx = trend_df["count"].idxmax()
        max_row = trend_df.loc[max_idx]
        max_count = max_row["count"]

        plt.figure(figsize=(12, 6))

        bar_colors = ["gray"] * len(trend_df)
        bar_colors[max_idx] = "orange"

        plt.bar(trend_df["timestamp"], trend_df["count"], color=bar_colors, width=20, label="Aantal berichten")
        plt.scatter(max_row["timestamp"], max_count, color="red", s=100, zorder=5, label="Piek")

        avg_count = trend_df["count"].mean()
        plt.axhline(y=avg_count, color="blue", linestyle="--", linewidth=1.5, label=f"Gemiddeld ({avg_count:.1f})")

        plt.annotate(
            f"Vervanging {self.keyword} deurdrangers",
            xy=(max_row["timestamp"], max_count),
            xytext=(max_row["timestamp"] + pd.DateOffset(months=2), max_count),
            arrowprops=dict(arrowstyle="->", color="red"),
            va="center",
            color="red"
        )

        plt.xlabel("Tijd")
        plt.ylabel(f"Aantal '{self.keyword}' berichten")
        plt.title(f"'{self.keyword}' gesprekken over de jaren heen")

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.legend()

        save_path = self.img_dir / f"wk2_{self.keyword}_gesprekken_comparing_categories.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze messages by keyword")
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        choices=["lift", "schoon", "camera"],
        help="Keyword to analyze: 'lift', 'schoon' or 'camera'"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # Setup logger from utilities (with custom filename)
    log_filename = "wk2_comparing_categories.log"
    logger_obj = LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # Load data
    data_handler = DataHandler(config)
    logger.info(f"Loading data from {data_handler.datafile}")
    df, author_info_df = data_handler.load_data()

    # Run analysis
    keyword = args.keyword
    logger.info(f"=== Analyzing keyword: '{keyword}' ===")
    img_dir = Path.cwd() / "img"
    analysis = MessageAnalysis(df, keyword, img_dir)
    analysis.plot_trend()

    logger.info("Done!")


if __name__ == "__main__":
    main()
