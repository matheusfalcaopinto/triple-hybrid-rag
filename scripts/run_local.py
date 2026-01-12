"""Run Pipecat server locally for testing.

Usage:
    source pipecat_venv/bin/activate
    python scripts/run_pipecat_local.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    
    # Load environment
    if load_dotenv:
        env_path = repo_root / ".env"
        load_dotenv(dotenv_path=str(env_path), override=False)
    
    # Add repo to path (prefer local src over installed package)
    src_path = repo_root / "src"
    for path in (src_path, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    # Parse args
    parser = argparse.ArgumentParser(description="Run Pipecat voice agent locally")
    parser.add_argument("--port", type=int, default=5050, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Run with uvicorn
    import uvicorn
    
    uvicorn.run(
        "voice_agent.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
