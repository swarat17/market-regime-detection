"""
Unit tests for Phase 6 — Gradio frontend utility functions.

These tests only cover pure functions (regime_badge, generate_interpretation)
and do NOT load any models or start the Gradio server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_regime_badge_bull():
    from frontend.app import regime_badge

    result = regime_badge("bull")
    assert "🟢" in result, f"Expected 🟢 in badge for bull, got: {result!r}"


def test_regime_badge_bear():
    from frontend.app import regime_badge

    result = regime_badge("bear")
    assert "🔴" in result, f"Expected 🔴 in badge for bear, got: {result!r}"


def test_interpretation_text_not_empty():
    from frontend.app import generate_interpretation

    result = generate_interpretation("bear", 0.91)
    assert (
        isinstance(result, str) and len(result) > 0
    ), f"Expected non-empty string from generate_interpretation, got: {result!r}"
