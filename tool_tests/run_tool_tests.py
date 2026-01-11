from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_PARENT = PROJECT_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_tests.harness.runner import main as harness_main


def main() -> None:
    exit_code = asyncio.run(harness_main())
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
