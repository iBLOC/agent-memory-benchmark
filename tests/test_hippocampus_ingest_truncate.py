"""Regression tests for the hippocampus AMB provider's ingest truncation.

Run:  uv run --no-sync python tests/test_hippocampus_ingest_truncate.py
(plain asserts, no pytest required)

Locks the 2026-10 fix: LifeBench-style 100-400KB documents pushed whole
through the server's per-insert path OOM-aborted the server (multi-MB
Debug-log formatting + FTS/index buffers). The provider now caps STORED
content at HIPPOCAMPUS_INGEST_CHAR_LIMIT with head+tail kept.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_bench.memory.hippocampus import HippocampusMemoryProvider  # noqa: E402


def test_no_cap_passthrough():
    os.environ["HIPPOCAMPUS_INGEST_CHAR_LIMIT"] = "0"
    big = "x" * 500_000
    assert HippocampusMemoryProvider._ingest_truncate(big) == big


def test_under_cap_passthrough():
    os.environ["HIPPOCAMPUS_INGEST_CHAR_LIMIT"] = "24000"
    small = "y" * 100
    assert HippocampusMemoryProvider._ingest_truncate(small) == small


def test_cap_head_and_tail_kept():
    os.environ["HIPPOCAMPUS_INGEST_CHAR_LIMIT"] = "24000"
    head_marker = "HEADMARKER"
    tail_marker = "TAILMARKER"
    text = head_marker + ("m" * 100_000) + tail_marker
    out = HippocampusMemoryProvider._ingest_truncate(text)
    assert len(out) <= 24_000 + 64, f"capped output too long: {len(out)}"
    assert out.startswith(head_marker), "head must be kept"
    assert out.endswith(tail_marker), "tail must be kept"
    assert "[middle elided at ingest]" in out, "elision marker must be present"


def test_cap_bound_is_respected():
    os.environ["HIPPOCAMPUS_INGEST_CHAR_LIMIT"] = "24000"
    out = HippocampusMemoryProvider._ingest_truncate("z" * 1_000_000)
    # 24k cap + elision marker + rounding slack
    assert len(out) <= 24_000 + 64


if __name__ == "__main__":
    test_no_cap_passthrough()
    test_under_cap_passthrough()
    test_cap_head_and_tail_kept()
    test_cap_bound_is_respected()
    print("4/4 ingest-truncate tests passed")
