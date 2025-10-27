# %%
import argparse
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import tomllib
from loguru import logger
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

from logger_setup import LoggerSetup
from config_loader import ConfigLoader


# ====================================================
# --- Data Handler with Metadata merging ---
# ====================================================
class DataHandler_merge_meta:
    def __init__(self, config: dict):
        self.project_root = Path.cwd()
        self.datafile = self.project_root / config["processed"] / config["current"]
        self.metadata_file = self.project_root / config["meta"] / config["resident_metadata"]

    def load_data(self):
        if not self.datafile.exists():
            raise FileNotFoundError(
                f"{self.datafile} does not exist. Run src/preprocess.py first and check the timestamp!"
            )
        df = pd.read_parquet(self.datafile)

        # Load and merge metadata
        with open(self.metadata_file, "r") as f:
            nested_users = json.load(f)
        author_info_df = (
            pd.DataFrame(nested_users)
            .T.reset_index()
            .rename(columns={"index": "author"})
        )
        df_merged = df.merge(author_info_df, on="author", how="left")
        logger.info("Data successfully loaded and merged with metadata.")
        return df_merged, author_info_df


# ====================================================
# --- Analysis ---
# ====================================================
class ElevatorAnalysis:
    def __init__(self, df: pd.DataFrame, keyword: str, img_dir: Path):
        self.df = df
        self.keyword = keyword
        self.img_dir = img_dir
        self.img_dir.mkdir(exist_ok=True)
        self.floor_stats = None

    def preprocess(self):
        pattern = rf"{self.keyword}"
        self.df["mentions_keyword"] = self.df["message"].str.contains(
            pattern, case=False, na=False
        )
        self.df["hour"] = pd.to_datetime(self.df["timestamp"]).dt.hour
        logger.info(f"Preprocessing complete: messages containing '{self.keyword}' identified.")

    def aggregate_by_floor(self):
        self.floor_stats = (
            self.df[self.df["mentions_keyword"]]
            .groupby("Floor_nr")
            .agg(
                keyword_msgs=("message", "count"),
                avg_msg_length=("message_length", "mean"),
                n_authors=("author", "nunique"),
            )
            .reset_index()
        )
        logger.info("Aggregation complete. Stats per floor computed.")
        print(f"\nAggregated statistics per floor for keyword '{self.keyword}':")
        print(self.floor_stats)

    def correlation_analysis(self):
        corr_msgs = self.floor_stats["Floor_nr"].corr(self.floor_stats["keyword_msgs"])
        corr_length = self.floor_stats["Floor_nr"].corr(self.floor_stats["avg_msg_length"])
        print(f"\n=== Correlation Analysis for '{self.keyword}' ===")
        print(f"Correlation (Floor vs {self.keyword} Mentions): {corr_msgs:.2f}")
        print(f"Correlation (Floor vs Avg Message Length): {corr_length:.2f}")
        return corr_msgs, corr_length

    def regression_analysis(self):
        X = sm.add_constant(self.floor_stats["Floor_nr"])
        y = self.floor_stats["keyword_msgs"]
        model = sm.OLS(y, X).fit()
        print(f"\n=== Regression Analysis for '{self.keyword}' ===")
        print(model.summary())
        return model

    def plot_keyword_mentions(self):
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=self.floor_stats,
            x="Floor_nr",
            y="keyword_msgs",
            scatter_kws={"s": 80},
            line_kws={"color": "red"},
        )
        plt.title(f"Hogere verdiepingen praten vaker over de '{self.keyword}'", fontsize=14, fontweight="bold")
        plt.xlabel("Etage nummer")
        plt.ylabel(f"Gemiddelde aantal berichten met '{self.keyword}'")
        plt.ylim(0)
        plt.tight_layout()

        out_path = self.img_dir / f"elevator_mentions_{self.keyword}.png"
        plt.savefig(out_path, dpi=300)
        logger.info(f"Plot saved to {out_path}")
        # plt.show()

        save_path = self.img_dir / f"wk5_aantal_berichten_per_etage_relationship.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


# ====================================================
# --- Main ---
# ====================================================
def main():
    # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- Setup logger ---
    log_filename = "wk5_relationship.log"
    logger_obj = LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # --- Load data ---
    data_handler = DataHandler_merge_meta(config)
    df, author_info_df = data_handler.load_data()


    logger.info("Data successfully loaded and merged with metadata.")

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


    # --- Run analysis ---

    keyword = args.keyword
    logger.info(f"=== Analyzing keyword: '{keyword}' ===")

    img_dir = Path.cwd() / "img"

    analysis = ElevatorAnalysis(df, keyword, img_dir)
    analysis.preprocess()
    analysis.aggregate_by_floor()
    analysis.correlation_analysis()
    analysis.regression_analysis()
    analysis.plot_keyword_mentions()

if __name__ == "__main__":
    main()
