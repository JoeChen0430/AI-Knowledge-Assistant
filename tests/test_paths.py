"""Containment tests — these encode the Phase 0 path-traversal fix."""

from __future__ import annotations

import pytest

from aka.security.paths import (
    UnsafePathError,
    is_within,
    prune_empty_dirs,
    safe_join,
    sanitize_filename,
    sanitize_relpath,
)


@pytest.fixture()
def root(tmp_path):
    r = tmp_path / "data"
    r.mkdir()
    return r


# --- sanitize_filename ----------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\cfg.ini", "cfg.ini"),
        ("  spaced name .txt", "spaced name.txt"),
        ("....txt", "txt"),
        ("no_extension", "no_extension"),
        ("CON", "upload"),
        ("weird;|&$(name).PDF", "weird_name.pdf"),
        (".env", "env"),  # no dotfiles land in the corpus
        ("報告書.pdf", "報告書.pdf"),  # CJK names survive intact
        ("", "upload"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_strips_rtl_override():
    """U+202E is used to disguise extensions; it must not survive."""
    out = sanitize_filename("invoice‮gpj.exe")
    assert "‮" not in out


def test_sanitize_filename_truncates_long_names():
    out = sanitize_filename("a" * 500 + ".txt")
    assert out.endswith(".txt")
    assert len(out) <= 130


def test_sanitize_filename_is_idempotent():
    once = sanitize_filename("../../weird;name.PDF")
    assert sanitize_filename(once) == once


# --- sanitize_relpath -----------------------------------------------------

def test_sanitize_relpath_strips_noise():
    assert sanitize_relpath("./papers//2026/") == "papers/2026"


def test_sanitize_relpath_rejects_traversal():
    with pytest.raises(UnsafePathError):
        sanitize_relpath("papers/../../escape")


# --- safe_join ------------------------------------------------------------

def test_safe_join_happy_path(root):
    out = safe_join(root, "papers", "q3.pdf")
    assert out == (root / "papers" / "q3.pdf").resolve()


def test_safe_join_blocks_traversal(root):
    with pytest.raises(UnsafePathError):
        safe_join(root, "../../etc", "passwd")


def test_safe_join_blocks_absolute(root):
    with pytest.raises(UnsafePathError):
        safe_join(root, "/etc/passwd")


def test_safe_join_blocks_symlink_escape(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError):
        safe_join(root, "link", "loot.txt")


def test_safe_join_empty_parts_returns_root(root):
    assert safe_join(root, "", None) == root.resolve()


# --- prune_empty_dirs -----------------------------------------------------

def test_prune_removes_empty_chain_up_to_root(root):
    leaf = root / "a" / "b" / "c"
    leaf.mkdir(parents=True)
    prune_empty_dirs(leaf, root)
    assert not (root / "a").exists()
    assert root.exists()


def test_prune_stops_at_non_empty(root):
    leaf = root / "a" / "b"
    leaf.mkdir(parents=True)
    (root / "a" / "keep.txt").write_text("x")
    prune_empty_dirs(leaf, root)
    assert (root / "a").exists()
    assert not leaf.exists()


def test_prune_refuses_outside_root(root, tmp_path):
    """The old while-loop would have climbed toward / from here."""
    outside = tmp_path / "elsewhere" / "deep"
    outside.mkdir(parents=True)
    prune_empty_dirs(outside, root)
    assert outside.exists()


def test_prune_never_deletes_root(root):
    prune_empty_dirs(root, root)
    assert root.exists()


# --- is_within ------------------------------------------------------------

def test_is_within(root, tmp_path):
    assert is_within(root, root / "x" / "y")
    assert is_within(root, root)
    assert not is_within(root, tmp_path / "other")
