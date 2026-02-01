# %%
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
from scipy.stats import mannwhitneyu 
from cliffs_delta import cliffs_delta


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

        logger.info("Plotting histogram with KDE and medians by gender...")

        df_male = self.df_question[self.df_question["author_gender"] == "Male"]
        df_female = self.df_question[self.df_question["author_gender"] == "Female"]

        # --- Compute medians ---
        median_male = df_male["msg_length"].median()
        median_female = df_female["msg_length"].median()

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

        # --- Median lines ---
        plt.axvline(
            median_male,
            linestyle="--",
            linewidth=2.5,
            label=f"Mediaan mannen (≈ {median_male:.0f})",
        )

        plt.axvline(
            median_female,
            linestyle="--",
            linewidth=2.5,
            label=f"Mediaan vrouwen (≈ {median_female:.0f})",
        )

        # --- Log scale ---
        plt.xscale("log")

        plt.title(
            "Lengte van vraagberichten (log-schaal)\n"
            "Distributies vraagberichten – mannen vs. vrouwen whatsapp Flatgemeenschap",
            fontsize=16,
        )
        plt.xlabel("Berichtlengte (aantal tekens, log-schaal)")
        plt.ylabel("Aantal berichten")
        plt.legend(title="Geslacht")

        plt.tight_layout()

        save_path = self.img_dir / "wk4_R_distributie_vraaglengtes_man_vrouw.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(
            f"Saved plot to {save_path} | "
            f"Median male: {median_male:.2f}, Median female: {median_female:.2f}"
        )

    def log_summary(self):
        """Log dataset overview and statistical summary of question lengths by gender."""
        if not hasattr(self, "df_question"):
            raise RuntimeError("prepare_data() must be run before printing summary.")

        df = self.df_question
        df_male = df[df["author_gender"] == "Male"]
        df_female = df[df["author_gender"] == "Female"]

        # --- Dataset overview ---
        total_msgs = len(self.df)
        total_authors = self.author_info_df["author"].nunique()
        n_male_authors = (self.author_info_df["Gender"] == "Male").sum()
        n_female_authors = (self.author_info_df["Gender"] == "Female").sum()
        n_questions = len(df)
        n_questions_male = len(df_male)
        n_questions_female = len(df_female)

        logger.info("Datasetoverzicht:")
        logger.info(f"  Totaal aantal berichten in chat : {total_msgs}")
        logger.info(f"  Totaal aantal deelnemers        : {total_authors}")
        logger.info(f"    ├─ Mannen                     : {n_male_authors}")
        logger.info(f"    └─ Vrouwen                    : {n_female_authors}")
        logger.info(f"  Aantal vragen                   : {n_questions}")
        logger.info(f"    ├─ Mannen                     : {n_questions_male}")
        logger.info(f"    └─ Vrouwen                    : {n_questions_female}")

        # --- Robust statistics ---
        median_male = df_male["msg_length"].median()
        iqr_male = df_male["msg_length"].quantile(0.75) - df_male["msg_length"].quantile(0.25)
        median_female = df_female["msg_length"].median()
        iqr_female = df_female["msg_length"].quantile(0.75) - df_female["msg_length"].quantile(0.25)

        logger.info("Robuuste statistieken:")
        logger.info(f"  Mediaan mannen   : {median_male:.1f} (IQR = {iqr_male:.1f})")
        logger.info(f"  Mediaan vrouwen  : {median_female:.1f} (IQR = {iqr_female:.1f})")

        # --- Mann–Whitney U-test ---
        u_stat, p_val = mannwhitneyu(df_female["msg_length"], df_male["msg_length"], alternative="two-sided")
        logger.info("Mann–Whitney U-test:")
        logger.info(f"  U-statistiek : {u_stat:.1f}")
        logger.info(f"  p-waarde     : {p_val:.6f}")

        # --- Effect size (Cliff's delta) ---
        delta, interpretation = cliffs_delta(df_female["msg_length"], df_male["msg_length"])
        logger.info("Effect size:")
        logger.info(f"  Cliff’s delta = {delta:.3f}")
        logger.info(f"  Interpretatie : {interpretation}")

        # --- Bootstrap 95% CI verschil medianen ---
        np.random.seed(42)
        n_boot = 10000
        med_diff = []
        for _ in range(n_boot):
            sample_male = np.random.choice(df_male["msg_length"], size=len(df_male), replace=True)
            sample_female = np.random.choice(df_female["msg_length"], size=len(df_female), replace=True)
            med_diff.append(np.median(sample_female) - np.median(sample_male))
        ci_low, ci_high = np.percentile(med_diff, [2.5, 97.5])
        logger.info("Bootstrap 95% betrouwbaarheidsinterval:")
        logger.info(f"  Verschil medianen (vrouw - man): [{ci_low:.1f}, {ci_high:.1f}]")



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
    log_filename = "wk4_vraagberichten_distributie.log"
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
    analysis.log_summary()
    analysis.plot_histogram()

    logger.info("Question length analysis finished successfully.")


if __name__ == "__main__":
    main()
