# %%
"""
Stylometric Analysis of WhatsApp Messages (Exclamation Marks Only)
------------------------------------------------------------------
Performs stylometric clustering using character trigram analysis.
Visualizes clusters of residents' writing styles based on messages
containing exclamation marks (!).
"""

import argparse
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA

from logger_setup import LoggerSetup
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")


# ====================================================
# --- Stylometric Analysis Class ---
# ====================================================
class ExclamationStylometry:
    """
    Performs stylometric analysis on WhatsApp messages that contain
    exclamation marks, using character trigram vectors and PCA for
    visualization.

    Attributes:
        df (pd.DataFrame): Input dataframe with WhatsApp messages.
        img_dir (Path): Directory to store generated figures.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, config):
        """
        Initialize the analysis class.

        Args:
            df (pd.DataFrame): DataFrame containing WhatsApp chat messages.
            img_dir (Path): Directory path for saving plots and results.
        """
        self.df = df.copy()
        self.img_dir = img_dir
        self.config = config
        self.img_dir.mkdir(exist_ok=True)

    # ------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------
    @staticmethod
    def clean_message(text: str) -> str:
        """
        Cleans a message by removing links, newlines, and excess whitespace.

        Args:
            text (str): Original message text.

        Returns:
            str: Cleaned, lowercased version of the message.
        """
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)  # remove links
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------
    # Filter Exclamation Messages
    # ------------------------------------------------
    def filter_exclamation_messages(self) -> pd.DataFrame:
        """
        Filters the dataset to include only messages containing exclamation marks.

        Returns:
            pd.DataFrame: Subset of the original DataFrame with cleaned text and
            a boolean column `is_exclamation`.
        """
        df = self.df.copy()
        df["clean_text"] = df["message"].apply(self.clean_message)
        df = df[df["clean_text"].str.len() > 0]
        df["is_exclamation"] = df["clean_text"].str.contains(r"!")
        df_ex = df[df["is_exclamation"]].copy()

        logger.info(f"{df_ex.shape[0]} messages contain exclamation marks.")
        return df_ex

    # ------------------------------------------------
    # Stylometric Vectorization and PCA
    # ------------------------------------------------
    def compute_stylometry(self, df_ex: pd.DataFrame) -> pd.DataFrame:
        """
        Performs stylometric vectorization and PCA for visualization.

        Aggregates messages per author, extracts character trigrams,
        computes pairwise Manhattan distances, and reduces the dimensionality
        using PCA for clustering visualization.

        Args:
            df_ex (pd.DataFrame): DataFrame of messages containing exclamation marks.

        Returns:
            pd.DataFrame: DataFrame with authors and their PCA coordinates (x, y).
        """
        author_texts = (
            df_ex.groupby("author")["clean_text"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        logger.info(f"Aggregated text per author: {author_texts.shape[0]} authors")

        vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
        X = vectorizer.fit_transform(author_texts["clean_text"])
        distance = manhattan_distances(X, X)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(distance)

        author_texts["x"] = coords[:, 0]
        author_texts["y"] = coords[:, 1]

        # Discard outliers
        # author_texts = author_texts[author_texts["x"] <= 80000].copy()
        # threshold_x = config["Stylometry"]["outlier_threshold_x"]
        threshold_x = self.config["Stylometry"]["outlier_threshold_x"]
        author_texts = author_texts[author_texts["x"] <= threshold_x].copy()

        logger.info(f"Filtered authors with PCA1 > {threshold_x}; remaining: {author_texts.shape[0]}")
        return author_texts

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    def plot_clusters(self, author_texts: pd.DataFrame):
        """
        Creates and saves a PCA scatter plot showing stylometric clustering
        of messages with exclamation marks.

        Args:
            author_texts (pd.DataFrame): DataFrame with PCA coordinates per author.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=author_texts,
            x="x",
            y="y",
            hue="author",
            palette="tab10",
            s=90,
            edgecolor="black",
        )

        plt.title("Stylometric Clustering of Exclamation (!) Messages per Author", fontsize=13)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend([], [], frameon=False)
        plt.tight_layout()

        out_path = self.img_dir / "wk6_2_1_pca_modelling_stylometry_exclamation.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PCA scatterplot to {out_path}")
        plt.close()

    # ------------------------------------------------
    # Cluster Inspection
    # ------------------------------------------------
    def inspect_cluster(self, author_texts, df_ex, xmin, xmax, ymin, ymax, sample_n=10):
        """
        Inspects authors in a specific cluster region based on PCA coordinates.

        This method:
        - Finds authors located in the given coordinate range
        - Displays their messages
        - Generates a zoomed-in plot highlighting the cluster

        Args:
            author_texts (pd.DataFrame): DataFrame containing authors and PCA coordinates.
            df_ex (pd.DataFrame): Original exclamation message dataset.
            xmin (float): Minimum x-coordinate boundary.
            xmax (float): Maximum x-coordinate boundary.
            ymin (float): Minimum y-coordinate boundary.
            ymax (float): Maximum y-coordinate boundary.
            sample_n (int, optional): Number of messages per author to display. Defaults to 10.

        Returns:
            pd.DataFrame | None: Authors within the specified cluster region,
            or None if no authors found.
        """

        cluster_authors_df = author_texts[
            (author_texts["x"] > xmin) & (author_texts["x"] < xmax)
            & (author_texts["y"] > ymin) & (author_texts["y"] < ymax)
        ].copy()

        if cluster_authors_df.empty:
            logger.warning("No authors found in this coordinate range.")
            return None

        logger.info("=== Authors in this cluster ===")
        for _, row in cluster_authors_df[["author", "x", "y"]].iterrows():
            logger.info(f"Author: {row['author']}, x={row['x']:.2f}, y={row['y']:.2f}")
        logger.info("================================")

        selected_authors = cluster_authors_df["author"].tolist()
        subset = df_ex[df_ex["author"].isin(selected_authors)]

        # Log messages per author
        for auth in selected_authors:
            logger.info(f"--- Messages from {auth} ---")
            msgs = subset[subset["author"] == auth]["clean_text"].sample(
                min(sample_n, len(subset[subset["author"] == auth])), random_state=42
            )
            for msg in msgs:
                logger.info(f"• {msg}")

        # Zoomed plot visualization
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=author_texts,
            x="x",
            y="y",
            hue="author",
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            data=cluster_authors_df,
            x="x",
            y="y",
            hue="author",
            s=120,
            edgecolor="black",
        )
        plt.xlim(xmin - 1000, xmax + 1000)
        plt.ylim(ymin - 500, ymax + 500)
        plt.title("Zoomed-in Cluster Region (Selected Authors Only)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Author", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out_zoom = self.img_dir / "wk6_2_2_pca_modelling_stylometry_zoomed_cluster.png"
        plt.savefig(out_zoom, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved zoomed cluster plot to {out_zoom}")

        return cluster_authors_df


# ====================================================
# --- Main ---
# ====================================================
def main():
    """
    Command-line entry point for running the exclamation mark stylometry analysis.

    Loads configuration, initializes logging, filters messages containing exclamation marks,
    computes stylometric features via PCA, and optionally inspects a user-defined cluster region.
    """

        # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    parser = argparse.ArgumentParser(description="Stylometric Analysis of Exclamation Messages")
    # parser.add_argument("--xmin", type=float, default=10000)
    # parser.add_argument("--xmax", type=float, default=40000)
    # parser.add_argument("--ymin", type=float, default=-6500)
    # parser.add_argument("--ymax", type=float, default=-4500)

    parser.add_argument("--xmin", type=float, default=config["Stylometry"]["xmin"])
    parser.add_argument("--xmax", type=float, default=config["Stylometry"]["xmax"])
    parser.add_argument("--ymin", type=float, default=config["Stylometry"]["ymin"])
    parser.add_argument("--ymax", type=float, default=config["Stylometry"]["ymax"])


    args = parser.parse_args()

    # --- Setup logger ---
    log_filename = "wk6_pca_modelling.log"
    logger_obj = LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # --- Load data ---
    data_handler = DataHandler(config)
    df, _ = data_handler.load_data()

        # --- Prepare image directory from config ---
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Image directory set to: {img_dir}")

    # --- Run analysis ---
    # img_dir = Path.cwd() / "img"
    analysis = ExclamationStylometry(df, img_dir, config)

    df_ex = analysis.filter_exclamation_messages()
    author_texts = analysis.compute_stylometry(df_ex)
    analysis.plot_clusters(author_texts)

    # Optional: Inspect a region
    # cluster_df = analysis.inspect_cluster(author_texts, df_ex, args.xmin, args.xmax, args.ymin, args.ymax)

    sample_n = config["Stylometry"]["sample_n"]
    cluster_df = analysis.inspect_cluster(
        author_texts, df_ex, args.xmin, args.xmax, args.ymin, args.ymax, sample_n=sample_n
)


    if cluster_df is not None and not cluster_df.empty:
        logger.info("=== Focus Cluster Summary ===")
        for _, row in cluster_df[["author", "x", "y"]].iterrows():
            logger.info(f"Author: {row['author']}, x={row['x']:.2f}, y={row['y']:.2f}")


if __name__ == "__main__":
    main()
