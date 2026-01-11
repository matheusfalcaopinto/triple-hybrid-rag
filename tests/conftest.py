from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT_DIR = Path(__file__).resolve().parents[1]


if importlib.util.find_spec("pydantic_settings") is None:  # pragma: no cover - import plumbing
    stub = ModuleType("pydantic_settings")
    # Minimal stubs to satisfy configuration imports during tests.
    stub.BaseSettings = importlib.import_module("pydantic").BaseModel  # type: ignore[attr-defined]
    stub.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = stub


# Ensure the package is importable as ``voice_agent_v4`` when running tests directly
# from the repository checkout.
# if "voice_agent_v4" not in sys.modules:  # pragma: no cover - import plumbing
#     spec = importlib.util.spec_from_file_location(
#         "voice_agent_v4",
#         ROOT_DIR / "__init__.py",
#         submodule_search_locations=[str(ROOT_DIR)],
#     )
#     if spec and spec.loader:
#         module = importlib.util.module_from_spec(spec)
#         sys.modules["voice_agent_v4"] = module
#         spec.loader.exec_module(module)
