"""
Load config from YAML. Access as config.lr, config.epochs, etc.
config.yaml is the single source of truth.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config(path: Path | None = None) -> SimpleNamespace:
    path = path or _CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sw = data["stopwords"]
    data["stopwords"] = frozenset(sw) if isinstance(sw, list) else frozenset(str(sw).split())
    return SimpleNamespace(**data)


config = _load_config()
