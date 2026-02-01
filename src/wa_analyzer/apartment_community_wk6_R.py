"""
Communicative Style Clustering of WhatsApp Residents
----------------------------------------------------
Identifies behavioral communication styles per resident
based on aggregated WhatsApp chat features.

Methods:
- Feature engineering per author
- PCA (interpretability)
- t-SNE (visual clustering)
- DBSCAN clustering
- Manhattan-distance validation
- Cluster summary visualization
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.stats import mannwhitneyu

from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler
from logger_setup import LoggerSetup

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")


# ====================================================
# --- Communicative Style Analysis Class ---
# ====================================================
class CommunicativeStyleAnalysis:
    """
    Performs clustering of WhatsApp residents based on
    communicative behavior (style, not content).
    """

    FEATURE_COLS = [
        "avg_message_length",
        "message_length_std",
        "emoji_ratio",
        "link_ratio",
        "topk_ratio",
        "msg_count_norm",
        "avg_hour",
        "day_of_week_std",
    ]

    CLUSTER_LABELS = {
        0: "Expressieve sociale chatters",
        1: "Stille functionele gebruikers",
        2: "Informatieve schrijvers",
        3: "Actieve snelle reageerders",
        -1: "Outliers / atypisch gedrag",
    }

    def __init__(self, df: pd.DataFrame, meta_df: pd.DataFrame, img_dir: Path, config: dict):
        self.df = df.copy()
        self.meta_df = meta_df.copy()
        self.img_dir = img_dir
        self.config = config

        self.img_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------
    def engineer_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df["hour_int"] = df["hour"].apply(lambda t: t.hour)

        author_features = (
            df.groupby("author")
            .agg(
                msg_count=("message", "count"),
                avg_message_length=("message_length", "mean"),
                message_length_std=("message_length", "std"),
                emoji_ratio=("has_emoji", "mean"),
                link_ratio=("has_link", "mean"),
                topk_ratio=("is_topk", "mean"),
                avg_hour=("hour_int", "mean"),
                day_of_week_std=("day_of_week", "std"),
            )
            .reset_index()
        )

        author_features["msg_count_norm"] = (
            author_features["msg_count"] / author_features["msg_count"].sum()
        )

        author_features[["message_length_std", "day_of_week_std"]] = (
            author_features[["message_length_std", "day_of_week_std"]].fillna(0)
        )

        author_features.drop(columns="msg_count", inplace=True)

        logger.info(f"Engineered features for {len(author_features)} authors")
        return author_features

    # ------------------------------------------------
    # Scaling
    # ------------------------------------------------
    def scale_features(self, author_features: pd.DataFrame) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(author_features[self.FEATURE_COLS])

    # ------------------------------------------------
    # PCA
    # ------------------------------------------------
    def run_pca(self, X_scaled: np.ndarray, author_features: pd.DataFrame):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)

        author_features["PC1"] = coords[:, 0]
        author_features["PC2"] = coords[:, 1]

        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
        return author_features

    def plot_pca(self, author_features: pd.DataFrame):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=author_features, x="PC1", y="PC2", s=90)
        plt.axhline(0, color="grey", lw=0.5)
        plt.axvline(0, color="grey", lw=0.5)
        plt.title("PCA – communicatiestijl per bewoner")
        plt.tight_layout()

        out = self.img_dir / "wk6_R_1_pca_communicatiestijl_bewoner.png"
        plt.savefig(out, dpi=300)
        plt.close()

    # ------------------------------------------------
    # t-SNE
    # ------------------------------------------------
    def run_tsne(self, X_scaled: np.ndarray, author_features: pd.DataFrame):
        tsne = TSNE(
            n_components=2,
            perplexity=self.config["Clustering"]["perplexity"],
            random_state=42,
        )
        coords = tsne.fit_transform(X_scaled)
        author_features["TSNE1"] = coords[:, 0]
        author_features["TSNE2"] = coords[:, 1]
        return author_features

    def plot_tsne(self, author_features: pd.DataFrame):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=author_features, x="TSNE1", y="TSNE2", s=90)
        plt.title("t-SNE – Communicative Styles")
        plt.tight_layout()

        out = self.img_dir / "wk6_R_2_tnse_communicatiestijl.png"
        plt.savefig(out, dpi=300)
        plt.close()

    # ------------------------------------------------
    # Clustering
    # ------------------------------------------------
    def cluster_tsne(self, author_features: pd.DataFrame) -> pd.DataFrame:
        dbscan = DBSCAN(
            eps=self.config["Clustering"]["eps"],
            min_samples=self.config["Clustering"]["min_samples"],
        )

        author_features["cluster"] = dbscan.fit_predict(
            author_features[["TSNE1", "TSNE2"]]
        )
        author_features["cluster_label"] = author_features["cluster"].map(self.CLUSTER_LABELS)

        logger.info("Cluster sizes:")
        logger.info(author_features["cluster"].value_counts())
        return author_features

    def plot_clusters(self, author_features: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=author_features,
            x="TSNE1",
            y="TSNE2",
            hue="cluster_label",
            s=100,
        )
        plt.title("t-SNE Clusters met communicatiestijlen in de WhatsApp flatgebouw")
        plt.legend(title="Cluster")
        plt.tight_layout()

        out = self.img_dir / "wk6_R_3_tnse_communicatiestijl_labelled.png"
        plt.savefig(out, dpi=300)
        plt.close()

    # ------------------------------------------------
    # Manhattan-distance validation
    # ------------------------------------------------
    def plot_manhattan_distances(self, author_features: pd.DataFrame):
        X = author_features[self.FEATURE_COLS].values
        clusters = author_features["cluster"].values

        dist = pairwise_distances(X, metric="manhattan")

        within, between = [], []
        n = len(X)

        for i in range(n):
            for j in range(i + 1, n):
                if clusters[i] == -1 or clusters[j] == -1:
                    continue
                if clusters[i] == clusters[j]:
                    within.append(dist[i, j])
                else:
                    between.append(dist[i, j])

        stat, p = mannwhitneyu(within, between, alternative="less")

        logger.info(
            f"Manhattan test: mean within={np.mean(within):.3f}, "
            f"mean between={np.mean(between):.3f}, p={p:.4f}"
        )

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=[within, between])
        plt.xticks([0, 1], ["Binnen cluster", "Tussen clusters"])
        plt.ylabel("Manhattan distance")
        plt.title("Manhattan Distance: binnen clusters vs. tussen clusters")
        plt.tight_layout()

        out = self.img_dir / "wk6_R_5_Manhattan_distance_binnen_vs_tussen_cluster.png"
        plt.savefig(out, dpi=300)
        plt.close()

    # ------------------------------------------------
    # Cluster summary table (printscreen)
    # ------------------------------------------------
    def plot_cluster_summary(self, author_features: pd.DataFrame):
        summary = (
            author_features
            .groupby("cluster_label")[self.FEATURE_COLS]
            .mean()
            .round(3)
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")

        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            rowLabels=summary.index,
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        plt.title("Samenvatting Cluster Labeled tabel – communicatiestijl", pad=20)
        plt.tight_layout()

        out = self.img_dir / "wk6_R_4_labeled_cluster_tabel.png"
        plt.savefig(out, dpi=300)
        plt.close()


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Communicative Style Clustering")
    parser.parse_args()

    config = ConfigLoader(Path("config.toml")).load()
    LoggerSetup(config, "communicative_style_clustering.log").setup()

    data_handler = DataHandler(config)
    df, meta_df = data_handler.load_data()

    img_dir = Path(config["Images"]["imgdir"]).resolve()

    analysis = CommunicativeStyleAnalysis(df, meta_df, img_dir, config)

    author_features = analysis.engineer_features()
    X_scaled = analysis.scale_features(author_features)

    author_features = analysis.run_pca(X_scaled, author_features)
    analysis.plot_pca(author_features)

    author_features = analysis.run_tsne(X_scaled, author_features)
    analysis.plot_tsne(author_features)

    author_features = analysis.cluster_tsne(author_features)
    analysis.plot_clusters(author_features)

    analysis.plot_manhattan_distances(author_features)
    analysis.plot_cluster_summary(author_features)

    logger.info("Communicative style analysis completed successfully")


if __name__ == "__main__":
    main()
