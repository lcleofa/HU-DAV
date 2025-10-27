# data_handler.py
from pathlib import Path
import pandas as pd
import json
from loguru import logger

class DataHandler:
    """Load processed message data along with author metadata."""

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
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"{self.metadata_file} does not exist!")

        with open(self.metadata_file, "r") as f:
            nested_users = json.load(f)

        author_info_df = (
            pd.DataFrame(nested_users)
            .T
            .reset_index()
            .rename(columns={"index": "author"})
            [['author', 'Name', 'Gender']]
        )

        logger.info(f"Loaded {len(df)} messages and {len(author_info_df)} author metadata entries.")

        return df, author_info_df
