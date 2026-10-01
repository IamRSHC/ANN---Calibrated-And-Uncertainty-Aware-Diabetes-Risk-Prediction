"""Static guard: only features.py (writer) and evaluate.py (reader) touch the test file."""
from __future__ import annotations

import re

from src import config

ALLOWED = {"src/config.py", "src/features.py", "src/evaluate.py"}
PATTERNS = [
    re.compile(r"TEST_FILE"),
    re.compile(re.escape(config.TEST_FILE)),
    re.compile(r"_read_test_split"),
]
SEARCH_DIRS = ["src", "notebooks", "app"]


def test_only_allowed_modules_reference_test_split():
    offenders = []
    for folder in SEARCH_DIRS:
        root = config.ROOT_DIR / folder
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".ipynb"} or ".ipynb_checkpoints" in path.parts:
                continue
            rel = path.relative_to(config.ROOT_DIR).as_posix()
            if rel in ALLOWED:
                continue
            text = path.read_text(encoding="utf-8")
            if any(p.search(text) for p in PATTERNS):
                offenders.append(rel)
    assert not offenders, f"test split referenced outside evaluate.py: {offenders}"
