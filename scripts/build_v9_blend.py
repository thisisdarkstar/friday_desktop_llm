"""Build the v9 training blend from the FIXED migrated corpus.

v9 changes vs v8:
  - press_shortcut key-combos now map to semantic action names (migration fix),
    recovering the 4 shortcut gate tests v8 lost.
  - hardcase_replay grows from 34 -> 54 targeted paraphrases for the remaining
    genuine v8 failures (version check, philosophical, STT screenshot, obs
    record, calendar STT, clean email, shortcut semantics).
  - v5_precision exposure nudged up 28% -> 30% to close the 7-test core gap;
    v7_productivity held at 2x so the productivity win is preserved.

Everything else (dedupe, conflict overrides, val-leak removal) is identical to
v8. Output is a new directory; v8 assets are untouched and reproducible.

Usage:
    python scripts/build_v9_blend.py
    python scripts/build_v9_blend.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Load the v8 builder module, then override its paths + weights for v9.
spec = importlib.util.spec_from_file_location(
    "friday_v8_blend", ROOT / "scripts" / "build_v8_blend.py")
blend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blend)

blend.MIGRATED = ROOT / "dataset" / "v9_migrated"
blend.HARDCASE = ROOT / "dataset" / "v9_hardcase"
blend.OUT_DIR = ROOT / "dataset" / "v9_blend"
blend.SUBSET_WEIGHTS = {
    "synthetic_full": 1,
    "calibration": 2,
    "v4_perfect": 2,
    "v5_precision": 6,   # v8: 5 -> 30% core exposure (was 28%)
    "v6_daily_essentials": 2,
    "v7_productivity": 2,
    "hardcase_replay": 8,
}


def main() -> None:
    # Reuse the v8 builder's main() with overridden module globals.
    blend.main()


if __name__ == "__main__":
    main()
