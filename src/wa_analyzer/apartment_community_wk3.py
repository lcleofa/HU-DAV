# camera_analysis.py
import sys
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import tomllib


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None

    def load(self):
        with self.config_path.open("rb") as f:
            self.config = tomllib.load(f)
        return self.config
    
class LoggerSetup:
    """Configure Loguru logging."""
    def __init__(self, config):
        self.log_dir = Path(config["logging"]["logdir"]).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logfile = self.log_dir / "logfile.log"
        logger.add(
            logfile,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            enqueue=True
        )
        return logger       
class CameraAnalysis:
    """Analyze messages containing 'camera' and plot results."""
    def __init__(self, df: pd.DataFrame, keywords: list[str] = ["camera"]):
        self.df = df
        self.keywords = keywords
        self.df_camera = None
        self.camera_length_per_month = None
        self.peaks = {}

    def filter_messages(self):
        # Create regex pattern for keywords
        pattern = r"\b" + "|".join(self.keywords) + r"\b"
        self.df["has_camera"] = self.df["message"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        self.df_camera = self.df[self.df["has_camera"]].copy()


    def aggregate_by_month(self):
        # Drop timezone info to avoid pandas warning
        self.df_camera["timestamp"] = self.df_camera["timestamp"].dt.tz_localize(None)

        # Group by month
        camera_length = self.df_camera.groupby(
            self.df_camera["timestamp"].dt.to_period("M")
        )["message"].apply(lambda x: x.str.len().sum())

        # Convert PeriodIndex to datetime
        self.camera_length_per_month = camera_length.copy()
        self.camera_length_per_month.index = camera_length.index.to_timestamp()


    def find_peaks(self):
        series = self.camera_length_per_month

        # Peak 1
        peak1 = series[series > 6000].idxmin()
        self.peaks["peak1"] = (peak1, series.loc[peak1])

        # Peak 2
        peak2 = series[series > 12000].idxmin()
        self.peaks["peak2"] = (peak2, series.loc[peak2])

        # Last point
        last_idx = series.index[-1]
        last_val = series.iloc[-1]
        self.peaks["last"] = (last_idx, last_val)

    def plot(self, save_dir: Path | None = None):
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

        # Save plot if save_dir is provided
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"wk3_beveiliging_{'_'.join(self.keywords)}_time.png"

            plt.savefig(file_path, dpi=300)
            logger.info(f"Plot saved to {file_path}")

        plt.close()


def main():
    # --- Load config ---
    config_path = Path("config.toml").resolve()
    config = ConfigLoader(config_path).load()

    # --- Setup logger ---
    logger_obj = LoggerSetup(config).setup()

    # --- Load data ---
    datafile = (Path(config["processed"]) / config["current"]).resolve()
    if not datafile.exists():
        logger_obj.warning("Datafile does not exist. Run preprocess.py first.")
        return

    df = pd.read_parquet(datafile)

    img_dir = Path.cwd() / "img"

    # --- Analysis ---
    analysis = CameraAnalysis(df)
    analysis.filter_messages()
    analysis.aggregate_by_month()
    analysis.find_peaks()
    analysis.plot(save_dir=img_dir)


if __name__ == "__main__":
    main()
