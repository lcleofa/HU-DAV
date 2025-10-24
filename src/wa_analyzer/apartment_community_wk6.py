# %%
import argparse
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tomllib
from loguru import logger
import json
import warnings
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")


# ====================================================
# --- Config Loader ---
# ====================================================
class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None

    def load(self):
        with self.config_path.open("rb") as f:
            self.config = tomllib.load(f)
        return self.config


# ====================================================
# --- Logger Setup ---
# ====================================================
class LoggerSetup:
    """Configure Loguru logging."""

    def __init__(self, config):
        self.log_dir = Path(config["logging"]["logdir"]).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logfile = self.log_dir / "clustering_log.log"
        logger.add(
            logfile,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            enqueue=True
        )
        return logger


# ====================================================
# --- Data Handler ---
# ====================================================
class DataHandler:
    def __init__(self, config: dict):
        self.project_root = Path.cwd()
        self.datafile = self.project_root / config["processed"] / config["current"]
        self.metadata_file = self.project_root / config["meta"] / config["resident_metadata"]

    def load_data(self):
        if not self.datafile.exists():
            raise FileNotFoundError(
                f"{self.datafile} does not exist. Run src/preprocess.py first and check the timestamp!"
            )

        # Load WhatsApp messages
        df = pd.read_parquet(self.datafile)

        # Load author metadata
        with open(self.metadata_file, "r") as f:
            nested_users = json.load(f)

        author_info_df = (
            pd.DataFrame(nested_users)
            .T.reset_index()
            .rename(columns={"index": "author"})
        )

        logger.info(f"Loaded {len(df)} messages and {len(author_info_df)} author metadata entries.")

        return df, author_info_df


# ====================================================
# --- Clustering / PCA Analysis ---
# ====================================================
class TextClusteringAnalysis:
    def __init__(self, df: pd.DataFrame, author_info_df: pd.DataFrame, label_field: str, img_dir: Path):
        self.df = df.copy()
        self.author_info_df = author_info_df
        self.label_field = label_field
        self.img_dir = img_dir
        self.img_dir.mkdir(exist_ok=True)

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------
    @staticmethod
    def remove_url(text):
        return re.sub(r"https?://\S+", "", text)

    def clean_messages(self, keywords):
        pattern = '|'.join(keywords)
        filtered_df = self.df[self.df["message"].str.contains(pattern, case=False, na=False)].copy()

        filtered_df["message"] = (
            filtered_df["message"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .apply(self.remove_url)
            .str.lower()
        )

        filtered_df["size"] = filtered_df["message"].apply(len)
        logger.info(f"Filtered {len(filtered_df)} messages containing keywords: {keywords}")

        return filtered_df

    # ------------------------------------------------
    # Chunking by author
    # ------------------------------------------------
    def build_corpus(self, df_filtered, n=140, min_parts=2):
        corpus = {}
        for author in df_filtered["author"].unique():
            subset = df_filtered[df_filtered["author"] == author].reset_index()
            longseq = " ".join(subset["message"])
            parts = [longseq[i:i+n] for i in range(0, len(longseq), n)]
            if len(parts) > min_parts:
                corpus[author] = parts

        logger.info(f"Built corpus for {len(corpus)} authors.")
        return corpus

    # ------------------------------------------------
    # Vectorization, Distance, PCA
    # ------------------------------------------------
    def compute_pca(self, corpus):
        all_parts, labels = [], []
        for author, chunks in corpus.items():
            all_parts.extend(chunks)
            labels.extend([author] * len(chunks))

        logger.info(f"Total text chunks: {len(all_parts)}")

        vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
        X = vectorizer.fit_transform(all_parts)
        distance = manhattan_distances(X, X)
        pca = PCA(n_components=2)
        model = pca.fit_transform(distance)

        # Map author → selected metadata field
        label_map = dict(zip(self.author_info_df["author"], self.author_info_df[self.label_field]))
        metadata_labels = [label_map.get(a, "Unknown") for a in labels]

        pca_df = pd.DataFrame(model, columns=["PC1", "PC2"])
        pca_df[self.label_field] = metadata_labels

        return pca_df

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    def plot_pca(self, pca_df):
        label_field = self.label_field

        if label_field == "Older_then_65":
            palette = {True: "red", False: "blue"}
            legend_title = "65+"
        elif label_field == "Gender":
            palette = {"Male": "blue", "Female": "pink"}
            legend_title = "Gender"
        elif label_field == "Entrance_nr":
            palette = {1: "green", 2: "orange"}
            legend_title = "Entrance"
        elif label_field == "Nr_rooms":
            palette = {3: "red", 5: "blue"}
            legend_title = "Rooms"
        elif label_field == "Floor_nr":
            palette = {
                1: "#1f77b4",
                2: "#ff7f0e",
                3: "#2ca02c",
                4: "#d62728",
                5: "#9467bd",
                6: "#8c564b",
                7: "#e377c2",
                8: "#7f7f7f",
            }
            legend_title = "Floor"
        else:
            palette = "deep"
            legend_title = label_field

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=label_field,
            palette=palette,
            s=80,
            alpha=0.8
        )
        plt.title(f"PCA model op basis van {label_field}: cluster man/vrouw met opvallend berichtenstijl", fontsize=14)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        out_path = self.img_dir / f"pca_{label_field}.png"
        plt.savefig(out_path, dpi=300)
        logger.info(f"PCA plot saved to {out_path}")
        # plt.show()

        save_path = self.img_dir / f"wk6_pca_modelling.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {save_path}")


# ====================================================
# --- Main ---
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="WhatsApp Text Clustering PCA Analysis")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=["Older_then_65", "Gender", "Entrance_nr", "Nr_rooms", "Floor_nr"],
        help="Metadata label to color PCA by."
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs='+',
        required=True,
        help="Keywords to filter messages (e.g. lift camera trap ventilatie)"
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- Setup logger ---
    logger_obj = LoggerSetup(config).setup()

    # --- Load data and metadata ---
    data_handler = DataHandler(config)
    df, author_info_df = data_handler.load_data()

    # --- Run analysis ---
    img_dir = Path.cwd() / "img"
    analysis = TextClusteringAnalysis(df, author_info_df, args.label, img_dir)

    df_filtered = analysis.clean_messages(args.keywords)
    corpus = analysis.build_corpus(df_filtered)
    pca_df = analysis.compute_pca(corpus)
    analysis.plot_pca(pca_df)


if __name__ == "__main__":
    main()
