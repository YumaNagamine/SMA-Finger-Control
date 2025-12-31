from __future__ import annotations

import json
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load JSON config with a small amount of validation."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("video", "markers", "output"):
        if key not in data:
            raise ValueError(f"Config missing section '{key}' in {config_path}")

    return data
