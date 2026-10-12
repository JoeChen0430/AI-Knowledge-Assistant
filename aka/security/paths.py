"""Filesystem containment helpers.

The old UI joined user-supplied text straight onto DATA_DIR, so a subfolder of
``../../`` wrote outside the repo. Everything that turns user input into a path
now goes through :func:`safe_join`.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

# Windows reserved device names; harmless on POSIX but cheap to exclude.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
# Allowlist of unicode word chars (keeps CJK names intact), dot, dash, space.
# Containment is guaranteed by safe_join, not by this charset; the charset only
# keeps names printable and free of separators/format tricks like U+202E.
_UNSAFE_CHARS = re.compile(r"[^\w.\- ]+", re.UNICODE)
_DOT_RUNS = re.compile(r"\.{2,}")
MAX_NAME_LEN = 120


class UnsafePathError(ValueError):
    """Raised when user input would escape the containment root."""


def sanitize_filename(name: str, *, fallback: str = "upload") -> str:
    """Reduce an arbitrary upload name to a single safe path segment."""
    name = unicodedata.normalize("NFKC", name or "")
    # Strip any directory component the client may have sent (both separators).
    name = name.replace("\\", "/").split("/")[-1]
    name = name.strip().strip(".")

    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""

    stem = _UNSAFE_CHARS.sub("_", stem)
    stem = _DOT_RUNS.sub(".", stem).strip("._ ")
    ext = _UNSAFE_CHARS.sub("", ext).lower()

    # A stem of only separators carries no information — fall back instead.
    if not stem.strip("._- ") or stem.lower() in _RESERVED:
        stem = fallback
    stem = stem[:MAX_NAME_LEN]

    return f"{stem}.{ext}" if ext else stem


def sanitize_relpath(relpath: str) -> str:
    """Sanitize a multi-segment relative path, dropping traversal segments."""
    raw = (relpath or "").replace("\\", "/")
    segments = []
    for seg in raw.split("/"):
        seg = seg.strip()
        if not seg or seg == ".":
            continue
        if seg == "..":
            raise UnsafePathError("parent-directory segments are not allowed")
        segments.append(sanitize_filename(seg, fallback="folder"))
    return "/".join(segments)


def safe_join(root: Path | str, *parts: str) -> Path:
    """Join user-controlled ``parts`` under ``root``, guaranteeing containment.

    Resolves symlinks on both sides before comparing, so a symlink planted
    inside the root cannot be used to step outside it.
    """
    root_path = Path(root).resolve()
    cleaned: list[str] = []
    for part in parts:
        if part in (None, ""):
            continue
        if Path(part).is_absolute():
            raise UnsafePathError(f"absolute paths are not allowed: {part!r}")
        rel = sanitize_relpath(str(part))
        if rel:
            cleaned.append(rel)

    candidate = root_path.joinpath(*cleaned) if cleaned else root_path

    # strict=False so we can validate paths that do not exist yet.
    resolved = candidate.resolve(strict=False)
    if resolved != root_path and root_path not in resolved.parents:
        raise UnsafePathError(f"path escapes containment root: {candidate}")
    return resolved


def is_within(root: Path | str, target: Path | str) -> bool:
    """True when ``target`` resolves to ``root`` or something inside it."""
    root_path = Path(root).resolve()
    try:
        target_path = Path(target).resolve(strict=False)
    except OSError:
        return False
    return target_path == root_path or root_path in target_path.parents


def prune_empty_dirs(leaf: Path, root: Path) -> None:
    """Remove empty directories from ``leaf`` upward, stopping at ``root``.

    The previous implementation looped ``while parent != data_root`` with no
    containment check, so a path outside the root walked to ``/``. This version
    refuses to start unless ``leaf`` is genuinely inside ``root`` and bounds the
    climb by the actual segment count.
    """
    root = Path(root).resolve()
    current = Path(leaf).resolve(strict=False)
    if not is_within(root, current) or current == root:
        return

    for _ in range(len(current.parts)):
        if current == root or not is_within(root, current):
            return
        try:
            if any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


def relative_label(path: Path | str, root: Path | str) -> str:
    """Display path relative to root, for logs and UI (never leaks abs paths)."""
    try:
        return os.path.relpath(Path(path).resolve(), Path(root).resolve())
    except ValueError:
        return Path(path).name
