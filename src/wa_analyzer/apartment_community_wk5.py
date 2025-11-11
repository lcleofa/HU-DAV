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
    """
    Handles loading of preprocessed message data and merges it with author/resident metadata.

    Attributes:
        project_root (Path): Root directory of the project.
        datafile (Path): Path to the processed message data parquet file.
        metadata_file (Path): Path to the resident metadata JSON file.
    """

    def __init__(self, config: dict):
        """
        Initialize the DataHandler_merge_meta class.

        Args:
            config (dict): Configuration dictionary loaded from config.toml containing file paths.
        """
        self.project_root = Path.cwd()
        self.datafile = self.project_root / config["processed"] / config["current"]
        self.metadata_file = self.project_root / config["meta"] / config["resident_metadata"]

    def load_data(self):
        """
        Load message data and merge it with author/resident metadata.

        Returns:
            tuple:
                - df_merged (pd.DataFrame): DataFrame containing messages merged with metadata.
                - author_info_df (pd.DataFrame): DataFrame containing author metadata.
        
        Raises:
            FileNotFoundError: If the processed datafile does not exist.
        """
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
    """
    Analyze message activity per floor in relation to a specific keyword.

    Attributes:
        df (pd.DataFrame): DataFrame containing messages and metadata.
        keyword (str): Keyword to filter messages by (e.g., 'lift').
        img_dir (Path): Directory to save generated plots.
        floor_stats (pd.DataFrame): Aggregated statistics per floor after analysis.
    """

    def __init__(self, df: pd.DataFrame, keyword: str, img_dir: Path):
        """
        Initialize ElevatorAnalysis.

        Args:
            df (pd.DataFrame): DataFrame containing messages and author metadata.
            keyword (str): Keyword to filter messages.
            img_dir (Path): Directory for saving plots.
        """
        self.df = df
        self.keyword = keyword
        self.img_dir = img_dir
        self.img_dir.mkdir(exist_ok=True)
        self.floor_stats = None

    def preprocess(self):
        """
        Preprocess the dataset by:
            - Flagging messages containing the keyword (case-insensitive).
            - Extracting the hour from the timestamp.

        Adds columns:
            - 'mentions_keyword' (bool)
            - 'hour' (int)
        """
        pattern = rf"{self.keyword}"
        self.df["mentions_keyword"] = self.df["message"].str.contains(
            pattern, case=False, na=False
        )
        self.df["hour"] = pd.to_datetime(self.df["timestamp"]).dt.hour
        logger.info(f"Preprocessing complete: messages containing '{self.keyword}' identified.")

    def aggregate_by_floor(self):
        """
        Aggregate keyword-related statistics per floor, including:
            - Number of messages mentioning the keyword
            - Average message length
            - Number of unique authors

        Stores the result in self.floor_stats and prints the table.
        """
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
        """
        Compute correlations between floor number and:
            - Number of messages mentioning the keyword
            - Average message length

        Returns:
            tuple: (correlation_msgs, correlation_length)

        Prints correlation results to the console.
        """
        corr_msgs = self.floor_stats["Floor_nr"].corr(self.floor_stats["keyword_msgs"])
        corr_length = self.floor_stats["Floor_nr"].corr(self.floor_stats["avg_msg_length"])
        print(f"\n=== Correlation Analysis for '{self.keyword}' ===")
        print(f"Correlation (Floor vs {self.keyword} Mentions): {corr_msgs:.2f}")
        print(f"Correlation (Floor vs Avg Message Length): {corr_length:.2f}")
        return corr_msgs, corr_length

    def regression_analysis(self):
        """
        Perform OLS regression of keyword message count on floor number.

        Returns:
            model: Fitted statsmodels OLS regression model.

        Prints regression summary to the console.
        """
        X = sm.add_constant(self.floor_stats["Floor_nr"])
        y = self.floor_stats["keyword_msgs"]
        model = sm.OLS(y, X).fit()
        print(f"\n=== Regression Analysis for '{self.keyword}' ===")
        print(model.summary())
        return model

    def plot_keyword_mentions(self):
        """
        Generate and save a scatter plot with regression line:
            - x-axis: Floor number
            - y-axis: Number of messages mentioning the keyword
        Also adds correlation values for:
            - Floor vs keyword mentions
            - Floor vs average message length
        Saves the plot as PNG files in self.img_dir.
        """
        # Compute correlations
        corr_msgs, corr_length = self.correlation_analysis()

        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=self.floor_stats,
            x="Floor_nr",
            y="keyword_msgs",
            scatter_kws={"s": 80},
            line_kws={"color": "red"},
        )

        plt.title(f"Flatgebouw chat: Hogere verdiepingen praten vaker over de '{self.keyword}'", fontsize=14, fontweight="bold")
        plt.xlabel("Etage nummer")
        plt.ylabel(f"Aantal berichten met '{self.keyword}'")
        plt.ylim(0)

        # Annotate correlations on the plot
        plt.text(
            0.05, 0.95,
            f"Correlatie Etage vs Aantal berichten: {corr_msgs:.2f}\n"
            f"Correlatie Etage vs Gemiddelde berichtlengte: {corr_length:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

        plt.tight_layout()

        # Save plots
        out_path = self.img_dir / f"elevator_mentions_{self.keyword}.png"
        plt.savefig(out_path, dpi=300)
        logger.info(f"Plot saved to {out_path}")

        save_path = self.img_dir / f"wk5_aantal_berichten_per_etage_relationship_{self.keyword}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")



# ====================================================
# --- Main ---
# ====================================================
def main():
    """
    Main script entry point:
        - Loads configuration
        - Sets up logging
        - Loads data and merges metadata
        - Parses the keyword from command-line arguments
        - Performs preprocessing, aggregation, correlation, regression, and plotting
    """
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

    # --- Prepare image directory from config ---
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Image directory set to: {img_dir}")

    # --- Read allowed keywords from config ---
    allowed_keywords = config["Analysis"]["keywords"]

    # --- Load keyword from command-line argument ---
    parser = argparse.ArgumentParser(description="Analyze messages by keyword")

    parser.add_argument(
    "--keyword",
    type=str,
    required=True,
    choices=allowed_keywords,  # dynamic choices
    help=f"Keyword to analyze. Allowed: {', '.join(allowed_keywords)}"
    )

    args = parser.parse_args()

    # --- Run analysis ---
    keyword = args.keyword
    logger.info(f"=== Analyzing keyword: '{keyword}' ===")

    analysis = ElevatorAnalysis(df, keyword, img_dir)
    analysis.preprocess()
    analysis.aggregate_by_floor()
    analysis.correlation_analysis()
    analysis.regression_analysis()
    analysis.plot_keyword_mentions()


if __name__ == "__main__":
    main()
