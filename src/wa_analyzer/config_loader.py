from pathlib import Path
import tomllib

class ConfigLoader:
    """Load TOML configuration."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None

    def load(self):
        with self.config_path.open("rb") as f:
            self.config = tomllib.load(f)
        return self.config
