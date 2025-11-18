"""CLI entry point for reservoir-cli command.

This thin wrapper ensures that the src-based ``core_lib`` package is
importable when running directly from a source checkout, and then
delegates to ``core_lib.cli.main`` for the actual implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure src/ is on sys.path so that ``core_lib`` is importable when
# running from a local checkout without installing the package.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core_lib.cli import main  # type: ignore[import]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

