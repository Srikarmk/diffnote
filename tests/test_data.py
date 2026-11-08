import json
from pathlib import Path

from diffnote.data import DIFF_CHARS, TEST, VALID, clean_message, render

ROWS = [json.loads(l) for l in Path(__file__).parents[1].joinpath("data/commits.jsonl").open()]


def test_trailers_are_stripped():
    msg = "Fix the parser\n\nBody line.\n\nCo-Authored-By: Someone <x@y.z>\nSigned-off-by: A <a@b.c>"
    assert clean_message(msg) == "Fix the parser\n\nBody line."
    assert not any(l.lower().startswith("co-authored-by:") for r in ROWS for l in r["message"].splitlines())


def test_render_truncates_on_a_line_boundary():
    patch = "\n".join(f"+line {i}" for i in range(3000))
    out = render(" a.py | 3000 +", patch)
    assert out.startswith("a.py | 3000 +") and out.endswith("[... diff truncated ...]")
    assert len(out) < DIFF_CHARS + 100
    assert all(l.startswith("+line ") for l in out.splitlines()[2:-1])


def test_splits_are_by_repo_and_disjoint():
    by_split = {}
    for r in ROWS:
        by_split.setdefault(r["split"], set()).add(r["repo"])
    assert not (by_split["train"] & by_split["test"]) and not (by_split["train"] & by_split["valid"])
    assert not (by_split["valid"] & by_split["test"])
    assert len({r["id"] for r in ROWS}) == len(ROWS)
    assert len(TEST) == 6 and len(VALID) == 4
