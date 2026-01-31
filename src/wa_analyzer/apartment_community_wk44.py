# %%
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from logger_setup import LoggerSetup
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler


# ====================================================
# --- Question Message Length Analysis (All Authors) ---
# ====================================================
class QuestionLengthAnalysis:
    """Analyze and visualize question message length distributions
    for male vs. female authors (all authors, no top-N filtering).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        author_info_df: pd.DataFrame,
        img_dir: Path,
        max_length: int,
        bins: int = 35,
        bw_adjust: float = 1.6,
    ):
        self.df = df.copy()
        self.author_info_df = author_info_df.copy()
        self.max_length = max_length
        self.bins = bins
        self.bw_adjust = bw_adjust
        self.img_dir = img_dir
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        """Prepare and clean dataset for question-length analysis."""
        logger.info("Preparing data for question length analysis (all authors)...")

        # --- Map author metadata ---
        author_map = dict(zip(self.author_info_df["author"], self.author_info_df["Name"]))
        gender_map = dict(zip(self.author_info_df["author"], self.author_info_df["Gender"]))

        self.df["author_name"] = self.df["author"].map(author_map).fillna(self.df["author"])
        self.df["author_gender"] = self.df["author"].map(gender_map).fillna("Unknown")

        # --- Keep only question messages ---
        df_q = self.df[self.df["message"].str.endswith("?")].copy()
        logger.info(f"Found {len(df_q)} question messages.")

        # --- Filter message length ---
        df_q = df_q[
            (df_q["msg_length"] > 0) &
            (df_q["msg_length"] <= self.max_length)
        ]

        logger.info(f"{len(df_q)} messages remain after length filtering.")

        self.df_question = df_q

    def plot_histogram(self):
        """Plot histogram + KDE of question lengths by gender (log scale)."""
        if not hasattr(self, "df_question"):
            raise RuntimeError("prepare_data() must be run before plotting.")

        logger.info("Plotting histogram with KDE by gender...")

        df_male = self.df_question[self.df_question["author_gender"] == "Male"]
        df_female = self.df_question[self.df_question["author_gender"] == "Female"]

        plt.figure(figsize=(14, 8))

        # --- Male ---
        sns.histplot(
            df_male["msg_length"],
            bins=self.bins,
            stat="count",
            kde=True,
            kde_kws={"bw_adjust": self.bw_adjust},
            alpha=0.45,
            label="Mannelijke auteurs",
        )

        # --- Female ---
        sns.histplot(
            df_female["msg_length"],
            bins=self.bins,
            stat="count",
            kde=True,
            kde_kws={"bw_adjust": self.bw_adjust},
            alpha=0.45,
            label="Vrouwelijke auteurs",
        )

        # --- Log scale ---
        plt.xscale("log")

        plt.title(
            "Lengte van vraagberichten (log-schaal)\n"
            "Gesmoothde distributies – alle mannen vs. alle vrouwen",
            fontsize=16,
        )
        plt.xlabel("Berichtlengte (aantal tekens, log-schaal)")
        plt.ylabel("Aantal berichten")
        plt.legend(title="Geslacht")

        plt.tight_layout()

        save_path = self.img_dir / "vraaglengte_distributie_man_vrouw_log.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot to {save_path}")


# ====================================================
# --- Main entry point ---
# ====================================================
def main():
    """Main entry point for the question length analysis script."""

    # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- CLI arguments ---
    parser = argparse.ArgumentParser(
        description="Analyze question message length distributions by gender (all authors)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum question message length (overrides config)",
    )
    args = parser.parse_args()

    # --- Setup logger ---
    log_filename = "question_length_gender.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # --- Load data ---
    data_handler = DataHandler(config)
    logger.info(f"Loading data from {data_handler.datafile}")
    df, author_info_df = data_handler.load_data()

    # --- Determine max length ---
    max_length = config["Analysis"]["max_question_length"]
    if args.max_length is not None:
        max_length = args.max_length

    # --- Image directory ---
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    # --- Run analysis ---
    analysis = QuestionLengthAnalysis(
        df=df,
        author_info_df=author_info_df,
        img_dir=img_dir,
        max_length=max_length,
        bins=35,
        bw_adjust=1.6,
    )

    analysis.prepare_data()
    analysis.plot_histogram()

    logger.info("Question length analysis finished successfully.")


if __name__ == "__main__":
    main()
