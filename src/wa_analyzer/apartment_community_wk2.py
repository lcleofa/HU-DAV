import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import tomllib
from loguru import logger


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None

    def load(self):
        with self.config_path.open("rb") as f:
            self.config = tomllib.load(f)
        return self.config

class DataHandler:
    def __init__(self, config: dict):
        # Use current working directory (project root)
        project_root = Path.cwd() 
        self.datafile = project_root / config["processed"] / config["current"]
        
    def load_data(self):
        if not self.datafile.exists():
            raise FileNotFoundError(
                f"{self.datafile} does not exist. Run src/preprocess.py first and check the timestamp!"
            )
        df = pd.read_parquet(self.datafile)
        return df

class MessageAnalysis:
    def __init__(self, df: pd.DataFrame, keyword: str, img_dir: Path):
        self.df = df
        self.keyword = keyword
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)  # make sure folder exists

    def add_features(self):
        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["msg_length"] = self.df["message"].str.len()
        return self.df

    def plot_trend(self):
        # Filter messages containing the keyword
        keyword_df = self.df[self.df["message"].str.contains(self.keyword, case=False)]

        # Resample monthly
        trend_df = keyword_df.set_index("timestamp").resample("M").size().reset_index(name="count")

        if trend_df.empty:
            logger.warning(f"No messages found for keyword '{self.keyword}'.")
            return

        # Find the peak month
        max_idx = trend_df["count"].idxmax()
        max_row = trend_df.loc[max_idx]
        max_date = max_row["timestamp"].strftime("%Y-%m")
        max_count = max_row["count"]

        plt.figure(figsize=(12, 6))

        # --- Bar colors: make all gray except the peak month (orange) ---
        bar_colors = ["gray"] * len(trend_df)
        bar_colors[max_idx] = "orange"

        # --- Plot as bar chart ---
        plt.bar(trend_df["timestamp"], trend_df["count"], color=bar_colors, label="Aantal berichten", width=20)

        # --- Highlight the peak point with a red marker ---
        plt.scatter(max_row["timestamp"], max_count, color="red", s=100, zorder=5, label="Piek")

        # --- Average line ---
        avg_count = trend_df["count"].mean()
        plt.axhline(y=avg_count, color="blue", linestyle="--", linewidth=1.5, label=f"Gemiddeld ({avg_count:.1f})")

        # --- Annotate the peak ---
        plt.annotate(
            f"Vervanging {self.keyword} deurdrangers",
            xy=(max_row["timestamp"], max_count),
            xytext=(max_row["timestamp"] + pd.DateOffset(months=2), max_count),
            arrowprops=dict(arrowstyle="->", color="red"),
            va="center",
            color="red"
        )

        # --- Labels and title ---
        plt.xlabel("Maand")
        plt.ylabel(f"Aantal '{self.keyword}' berichten")
        plt.title(f"'{self.keyword}' gesprekken over de jaren heen")

        # --- Format x-axis every 3 months ---
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.legend()

        # --- Save plot ---
        save_path = self.img_dir / f"wk2_{self.keyword}_gesprekken_comparing_categories.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


def main():

    # Define log file path
    log_file_path = Path(__file__).parent / "logs" / "logfile.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # Add logger to write messages to file
    logger.add(log_file_path, rotation="10 MB", retention="10 days", level="INFO", encoding="utf-8")

    # Load keyword from command-line argument with restricted choices
    parser = argparse.ArgumentParser(description="Analyze messages by keyword")
    parser.add_argument(
        "--keyword",
        type=str,
        required=True,
        choices=["lift", "schoon", "camera"],  # only allow these keywords
        help="Keyword to analyze: 'lift', 'schoon' or 'camera'"
    )
    args = parser.parse_args()

    config_loader = ConfigLoader(Path.cwd() / "config.toml")
    config = config_loader.load()

    # Load data
    data_handler = DataHandler(config)
    logger.info(f"Loading data from {data_handler.datafile}")
    df = data_handler.load_data()

    keyword = args.keyword
    logger.info(f"=== Analyzing keyword: '{keyword}' ===")
    img_dir = Path.cwd() / "img"
    analysis = MessageAnalysis(df, keyword, img_dir)
    analysis.plot_trend()

    logger.info("Done!") 

if __name__ == "__main__":
    main()
