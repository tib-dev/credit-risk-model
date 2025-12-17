from pathlib import Path
from typing import Dict, Optional, Iterable
import logging
import yaml

from credit_risk_model.utils.project_root import get_project_root

logger = logging.getLogger(__name__)

# -------------------------
# Default path structure
# -------------------------
DEFAULT_STRUCTURE: Dict[str, Dict[str, str]] = {
    "data": {
        "data_dir": "data",
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
    },
    "reports": {
        "reports_dir": "reports",
        "plots_dir": "reports/plots",
    },
    "artifacts": {
        "artifacts_dir": "artifacts",
    },
}

# -------------------------
# Config loading
# -------------------------
DEFAULT_CONFIG_FILES = (
    "data.yaml",
    "features.yaml",
    "model.yaml",
    "api.yaml",
)


def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return data
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", path.name, exc)
        return {}


def load_config(
    config_dir: Optional[Path] = None,
    files: Iterable[str] = DEFAULT_CONFIG_FILES,
) -> Dict:
    root = get_project_root()
    config_dir = config_dir or (root / "configs")

    merged: Dict = {}
    for name in files:
        path = (config_dir / name).resolve()
        if not path.exists():
            logger.warning("Config file missing: %s", name)
            continue
        merged.update(_load_yaml(path))

    return merged


# -------------------------
# Path registry
# -------------------------
class PathRegistry:
    """
    Resolves and optionally creates project paths using defaults + YAML overrides.
    """

    def __init__(
        self,
        root: Path,
        config: Optional[Dict] = None,
        create_dirs: bool = True,
    ):
        self.root = root.resolve()
        self._create_dirs = create_dirs

        config = config or {}
        self.config: Dict[str, Dict[str, str]] = {}

        for section, defaults in DEFAULT_STRUCTURE.items():
            merged = dict(defaults)
            merged.update(config.get(section, {}))
            self.config[section] = merged

        self._init_sections()

    def _init_sections(self) -> None:
        for section, mapping in self.config.items():
            resolved: Dict[str, Path] = {}
            for key, rel_path in mapping.items():
                path = (self.root / rel_path).resolve()
                if self._create_dirs:
                    path.mkdir(parents=True, exist_ok=True)
                resolved[key] = path
            setattr(self, section, resolved)

    def __getitem__(self, section: str) -> Dict[str, Path]:
        return getattr(self, section)


# -------------------------
# Settings
# -------------------------
class Settings:
    """
    Central runtime settings object.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        config: Optional[Dict] = None,
        create_dirs: bool = True,
    ):
        self.root = root.resolve() if root else get_project_root()
        self.config = config or load_config(config_dir=self.root / "configs")
        self.paths = PathRegistry(self.root, self.config, create_dirs)

    @property
    def DATASET(self) -> Dict:
        return self.config.get("dataset", {})

    @property
    def DATA(self) -> Dict:
        return self.config.get("data", {})

    @property
    def FEATURES(self) -> Dict:
        return self.config.get("features", {})

    @property
    def MODEL(self) -> Dict:
        return self.config.get("model", {})

    @property
    def TUNING(self) -> Dict:
        return self.config.get("tuning", {})

    @property
    def API(self) -> Dict:
        return self.config.get("app", self.config.get("api", {}))


settings = Settings()
