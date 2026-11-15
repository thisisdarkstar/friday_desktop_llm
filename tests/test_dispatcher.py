"""Unit test suite for DesktopIntent tools and dispatcher."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dispatcher import dispatch, parse_llm_output


def test_single_intent():
    raw = '[{"tool": "get_system_stats", "args": {}}]'
    calls = parse_llm_output(raw)
    assert len(calls) == 1
    assert calls[0].tool == "get_system_stats"
    res = dispatch(raw)
    print("Single intent result:", res)
    assert res[0]["result"]["status"] == "success"


def test_multi_intent():
    raw = """
    [
      {"tool": "calculate", "args": {"expression": "15 * 80"}},
      {"tool": "create_timer", "args": {"duration_seconds": 120, "label": "Tea"}},
      {"tool": "chat_response", "args": {"message": "All set!"}}
    ]
    """
    calls = parse_llm_output(raw)
    assert len(calls) == 3
    assert calls[0].tool == "calculate"
    assert calls[1].tool == "create_timer"
    assert calls[2].tool == "chat_response"

    res = dispatch(raw)
    print("Multi intent results:")
    for r in res:
        print(" ->", r)
    assert res[0]["result"]["result"] == 1200


if __name__ == "__main__":
    test_single_intent()
    print("---")
    test_multi_intent()
    print("\nALL UNIT TESTS PASSED!")
