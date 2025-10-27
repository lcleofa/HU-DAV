from pathlib import Path
import pandas as pd

class DataHandler:
    """Load processed message data."""

    def __init__(self, config: dict):
        project_root = Path.cwd()
        self.datafile = project_root / config["processed"] / config["current"]

    def load_data(self):
        if not self.datafile.exists():
            raise FileNotFoundError(
                f"{self.datafile} does not exist. Run src/preprocess.py first and check the timestamp!"
            )
        df = pd.read_parquet(self.datafile)
        return df
