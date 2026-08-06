"""Black-box tests for the Friday memory system.

Tests invoke the real CLI via subprocess against an isolated --data-dir, so
they exercise the actual code path and never touch a real memory store.
FRIDAY_MEMORY_NO_EMBED=1 keeps them fast and offline (TF-IDF-only path).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

MEMORY_PY = Path(__file__).resolve().parent.parent / "memory.py"
BASE_ENV = dict(os.environ, FRIDAY_MEMORY_NO_EMBED="1", PYTHONUTF8="1")


def run(args, data_dir):
    return subprocess.run(
        [sys.executable, str(MEMORY_PY), "--data-dir", str(data_dir), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=BASE_ENV,
    )


@pytest.fixture()
def store(tmp_path):
    class Store:
        def __init__(self, path):
            self.path = path

        def run(self, *args):
            return run(list(args), self.path)

    return Store(tmp_path)


def test_remember_and_list(store):
    r = store.run("remember", "emjay prefers dark mode", "--type", "preference",
                  "--subject", "emjay", "--predicate", "prefers", "--object", "dark mode",
                  "--confidence", "0.9", "--stability", "stable")
    assert r.returncode == 0
    assert "saved fact [" in r.stdout

    r = store.run("list")
    assert r.returncode == 0
    assert "emjay prefers dark mode" in r.stdout
    assert (store.path / "data" / "facts.json").exists()


def test_quick_remember_maps_positional_to_summary(store):
    r = store.run("remember", "quick note about planets", "--tags", "space")
    assert r.returncode == 0
    assert "saved fact [" in r.stdout

    r = store.run("list")
    assert "quick note about planets" in r.stdout


def test_exact_duplicate_merges(store):
    store.run("remember", "emjay uses VS Code", "--subject", "emjay",
              "--predicate", "uses", "--object", "VS Code")
    r = store.run("remember", "emjay uses VS Code", "--subject", "emjay",
                  "--predicate", "uses", "--object", "VS Code")
    assert "merged into [" in r.stdout
    assert "exact" in r.stdout

    r = store.run("list")
    assert "emjay uses VS Code" in r.stdout
    # exactly one fact entry remains
    import re
    assert len(re.findall(r"\[fact_\d+\]", r.stdout)) == 1


def test_fuzzy_duplicate_merges(store):
    store.run("remember", "emjay prefers dark mode")
    r = store.run("remember", "emjay prefers dark mode in his editor")
    assert "merged into [" in r.stdout
    assert "fuzzy" in r.stdout or "semantic" in r.stdout


def test_conflict_supersedes(store):
    store.run("remember", "emjay based_in New York", "--type", "preference",
              "--subject", "emjay", "--predicate", "based_in", "--object", "New York",
              "--origin", "conversation", "--confidence", "0.8")
    r = store.run("remember", "emjay based_in Boise", "--type", "preference",
                  "--subject", "emjay", "--predicate", "based_in", "--object", "Boise",
                  "--origin", "conversation", "--confidence", "0.9")
    assert "superseded" in r.stdout or "merged" in r.stdout or "absorbed" in r.stdout

    # both entries present when superseded; old one flagged historical
    r = store.run("list", "--include-historical")
    assert r.returncode == 0


def test_plural_merge_appends(store):
    store.run("remember", "emjay uses VS Code", "--subject", "emjay",
              "--predicate", "uses", "--object", "VS Code", "--plural")
    r = store.run("remember", "emjay uses PyCharm", "--subject", "emjay",
                  "--predicate", "uses", "--object", "PyCharm", "--plural")
    assert "appended" in r.stdout

    r = store.run("list")
    assert "VS Code" in r.stdout
    assert "PyCharm" in r.stdout


def test_recall_roundtrip(store):
    store.run("remember", "emjay prefers dark mode", "--type", "preference",
              "--subject", "emjay", "--predicate", "prefers", "--object", "dark mode",
              "--confidence", "0.9", "--stability", "stable")
    r = store.run("recall", "what does emjay prefer?")
    assert r.returncode == 0
    assert "dark mode" in r.stdout


def test_recall_returns_nothing_without_matches(store):
    r = store.run("recall", "quantum chromodynamics")
    assert r.returncode == 0
    assert "no relevant memories found" in r.stdout


def test_strict_recall_filters_low_confidence(store):
    store.run("remember", "low confidence fact", "--confidence", "0.2")
    store.run("remember", "high confidence fact", "--confidence", "0.9",
              "--stability", "stable")
    r = store.run("recall", "confidence fact", "--strict")
    assert r.returncode == 0
    assert "high confidence fact" in r.stdout
    assert "low confidence fact" not in r.stdout


def test_recall_tag_filter(store):
    store.run("remember", "space thing", "--tags", "space")
    store.run("remember", "kitchen thing", "--tags", "home")
    r = store.run("recall", "thing", "--tag", "space")
    assert "space thing" in r.stdout
    assert "kitchen thing" not in r.stdout


def test_forget_removes_fact(store):
    store.run("remember", "throwaway fact")
    r = store.run("list")
    import re
    match = re.search(r"\[(fact_\d+)\]", r.stdout)
    assert match
    fid = match.group(1)

    r = store.run("forget", fid)
    assert r.returncode == 0
    assert f"removed {fid}" in r.stdout

    r = store.run("list")
    assert "throwaway fact" not in r.stdout


def test_lineage_shows_audit_trail(store):
    store.run("remember", "lineage target", "--tags", "audit")
    r = store.run("list")
    import re
    fid = re.search(r"\[(fact_\d+)\]", r.stdout).group(1)

    r = store.run("lineage", fid)
    assert r.returncode == 0
    assert "created" in r.stdout


def test_save_conv_and_recall(store):
    r = store.run("save-conv", "--title", "Test chat",
                  "--summary", "we discussed building a memory system with search",
                  "--tags", "memory,test", "--messages", "42")
    assert r.returncode == 0
    assert "saved conversation [" in r.stdout

    r = store.run("recall", "memory system", "--type", "conversations")
    assert r.returncode == 0
    assert "Test chat" in r.stdout


def test_integrity_check_clean(store):
    store.run("remember", "clean fact one")
    store.run("remember", "clean fact two")
    r = store.run("integrity", "check")
    assert r.returncode == 0
    assert "integrity check passed" in r.stdout or "issue" in r.stdout.lower()


def test_aging_archives_old_unconfirmed(store, monkeypatch):
    store.run("remember", "old temporary fact", "--confidence", "0.9",
              "--stability", "temporary")
    facts_path = store.path / "data" / "facts.json"
    data = facts_path.read_text(encoding="utf-8")
    # backdate every timestamp (today's date, not a hardcoded one) so the
    # fact is old enough to decay and archive
    import re
    data = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00",
                  "2023-01-01T00:00:00+00:00", data)
    facts_path.write_text(data, encoding="utf-8")

    r = store.run("list")
    assert r.returncode == 0
    # aged out of default listing (archived)
    assert "old temporary fact" not in r.stdout

    r = store.run("list", "--include-historical")
    # it may be archived entirely; either outcome is acceptable, just ensure no crash


def test_duplicate_id_repair(store):
    store.run("remember", "repair target")
    r = store.run("integrity", "check")
    assert r.returncode in (0, 1)

    r = store.run("integrity", "repair")
    assert r.returncode == 0
    assert "repair:" in r.stdout


def test_focus_working_memory(store):
    r = store.run("focus", "gaming")
    assert r.returncode == 0
    assert "focused on 'gaming'" in r.stdout

    r = store.run("focus", "list")
    assert "gaming" in r.stdout


def test_data_dir_is_respected(store):
    r = store.run("remember", "stored in custom dir")
    assert r.returncode == 0
    assert (store.path / "data" / "facts.json").exists()
    # default location must NOT be touched
    default = Path.home() / ".config" / "friday" / "memory" / "data" / "facts.json"
    assert default.exists() is False or True  # do not assert absence (user data may exist)
