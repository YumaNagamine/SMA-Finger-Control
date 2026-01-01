from __future__ import annotations

import json
from pathlib import Path


def load_config(config_path: str | Path, required_keys: tuple[str, ...] | None = None) -> dict:
    """Load JSON config with optional validation."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if required_keys:
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Config missing section '{key}' in {config_path}")

    return data
