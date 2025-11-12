# camera_analysis.py
import sys
import re
from pathlib import Path
from duckdb import df
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import tomllib

from logger_setup import LoggerSetup
from config_loader import ConfigLoader
from data_handler_meta_data import DataHandler
       
class CameraAnalysis:
    """
    Analyze messages containing specific keywords (default: 'camera') and visualize
    the total length of these messages per month.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing messages with 'timestamp' and 'message'.
        keywords (list[str]): List of keywords to search in messages.
        df_camera (pd.DataFrame): Filtered DataFrame containing only messages with keywords.
        camera_length_per_month (pd.Series): Aggregated message length per month.
        peaks (dict): Dictionary of key peaks for annotation.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        
        analysis_cfg = config.get("Analysis", {})
        # Prefer keywords_wk3, fallback to keywords, raise error if none defined
        if "keywords_wk3" in analysis_cfg:
            self.keywords = analysis_cfg["keywords_wk3"]
        elif "keywords" in analysis_cfg:
            self.keywords = analysis_cfg["keywords"]
        else:
            raise KeyError("No keywords defined in config under [Analysis]")

        # Strictly read thresholds from config, raise error if missing
        try:
            peaks_cfg = analysis_cfg["Peaks"]
            self.peak_thresholds = {
                "peak1_threshold": peaks_cfg["peak1_threshold"],
                "peak2_threshold": peaks_cfg["peak2_threshold"]
            }
        except KeyError as e:
            raise KeyError(f"Missing configuration in config.toml: {e}")

        self.plot_config = config.get("Plot", {})  # optional
        self.df_camera = None
        self.camera_length_per_month = None
        self.peaks = {}


    def filter_messages(self):
        """
        Filter messages that contain the specified keywords.
        Adds a new boolean column 'has_camera' and stores filtered DataFrame in self.df_camera.
        """
        pattern = r"\b" + "|".join(self.keywords) + r"\b"
        self.df["has_camera"] = self.df["message"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        self.df_camera = self.df[self.df["has_camera"]].copy()


    def aggregate_by_month(self):
        """
        Aggregate the total length of messages containing the keyword(s) per month.
        Converts the timestamp to naive datetime to avoid pandas timezone warnings.
        Stores results in self.camera_length_per_month.
        """
        self.df_camera["timestamp"] = self.df_camera["timestamp"].dt.tz_localize(None)

        camera_length = self.df_camera.groupby(
            self.df_camera["timestamp"].dt.to_period("M")
        )["message"].apply(lambda x: x.str.len().sum())

        self.camera_length_per_month = camera_length.copy()
        self.camera_length_per_month.index = camera_length.index.to_timestamp()


    def find_peaks(self):
        """
        Identify key peaks in message lengths for annotation in plots.
        Currently detects:
            - peak1: first value above 6000 (default)
            - peak2: first value above 12000 (default)
            - last: last value in the series
        Stores results in self.peaks as a dictionary with keys 'peak1', 'peak2', and 'last'.
        """
        series = self.camera_length_per_month

        peak1_val = self.peak_thresholds["peak1_threshold"]
        peak2_val = self.peak_thresholds["peak2_threshold"]

        peak1 = series[series > peak1_val].idxmin()
        self.peaks["peak1"] = (peak1, series.loc[peak1])

        peak2 = series[series > peak2_val].idxmin()
        self.peaks["peak2"] = (peak2, series.loc[peak2])

        last_idx = series.index[-1]
        last_val = series.iloc[-1]
        self.peaks["last"] = (last_idx, last_val)

    def plot(self, img_dir: Path | None = None):
        """
        Plot the aggregated message length per month, annotate peaks, and optionally save the figure.
        
        Args:
            img_dir (Path | None, optional): Directory to save the plot. If None, plot is not saved.
        """
        series = self.camera_length_per_month
        fig, ax = plt.subplots(figsize=(12, 6))

        # Line plot
        ax.plot(series.index, series.values, marker="o", linestyle="-", color="orange")
        ax.set_title(
            "Na lange discussies in flatgebouw geven leden alsnog groen licht voor camerabeveiliging",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Lengte berichten over 'camera'")

        # Mean line
        mean_val = series.mean()
        ax.axhline(y=mean_val, color="blue", linestyle="--", linewidth=1.5,
                   label=f"Gemiddelde ({mean_val:.0f})")
        ax.legend()

        # Style
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # X-axis ticks
        ax.set_xticks(series.index[::3])
        ax.set_xticklabels([d.strftime("%Y-%m") for d in series.index[::3]], rotation=45)

        # Annotations
        ax.annotate("Incidenten \nVeiligheidsoverleg",
                    xy=self.peaks["peak1"],
                    xytext=(self.peaks["peak1"][0] + pd.DateOffset(days=20), self.peaks["peak1"][1]),
                    ha="left", va="center", color="red", fontweight="bold")

        ax.annotate("Camerabeleid \nOffertes",
                    xy=self.peaks["peak2"],
                    xytext=(self.peaks["peak2"][0] + pd.DateOffset(days=20), self.peaks["peak2"][1]),
                    ha="left", va="center", color="red", fontweight="bold")

        ax.annotate("Besluitvorming\nInstallatie",
                    xy=self.peaks["last"],
                    xytext=(self.peaks["last"][0] + pd.DateOffset(days=20), self.peaks["last"][1]),
                    ha="left", va="center", color="green", fontweight="bold")

        plt.tight_layout()

        if img_dir:
            img_dir.mkdir(parents=True, exist_ok=True)
            file_path = img_dir / f"wk3_beveiliging_{'_'.join(self.keywords)}_time.png"
            plt.savefig(file_path, dpi=300)
            logger.info(f"Plot saved to {file_path}")

        plt.close()


def main():
    """
    Main function to run the camera message analysis workflow.
    """
    # --- Load configuration ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- Setup logger ---
    log_filename = "wk3_time_series.log"
    LoggerSetup(config, log_filename).setup()
    logger.info("Logger initialized successfully.")

    # --- Load data ---
    data_handler = DataHandler(config)
    logger.info(f"Loading data from {data_handler.datafile}")
    df, author_info_df = data_handler.load_data()  # <-- Unpack tuple properly

    # --- Prepare image directory ---
    img_dir = Path(config["Images"]["imgdir"]).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Image directory set to: {img_dir}")

    # keywords = config["Analysis"]["keywords_wk3"]
    analysis = CameraAnalysis(df, config=config)

    # --- Run analysis ---
    # analysis = CameraAnalysis(df)
    analysis.filter_messages()
    analysis.aggregate_by_month()
    analysis.find_peaks()
    analysis.plot(img_dir=img_dir)

    logger.info("Camera analysis completed successfully.")



if __name__ == "__main__":
    main()
