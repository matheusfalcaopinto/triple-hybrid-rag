"""
Lightweight schema helper for PuppyGraph configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class GraphSchema:
    """Represents a PuppyGraph schema JSON file."""

    path: Path

    def load(self) -> Dict[str, Any]:
        """Load the schema JSON from disk."""
        return json.loads(self.path.read_text(encoding="utf-8"))

    @classmethod
    def from_path(cls, path: str | Path) -> "GraphSchema":
        return cls(Path(path))
