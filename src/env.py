"""Utility functions for handling project paths.

This module provides small helper routines for resolving and duplicating
paths relative to the project root. It is a simplified version used in
TabDDPM but kept for backwards compatibility.
"""

from __future__ import annotations

import datetime
import os
import shutil
import typing as ty
from pathlib import Path

# Define common project directories
PROJ = Path('tab-ddpm/').absolute().resolve()
EXP = PROJ / 'exp'
DATA = PROJ / 'data'


def get_path(path: ty.Union[str, Path]) -> Path:
    """Resolve a path relative to the project root.

    Args:
        path: A path or string that may be relative or absolute.

    Returns:
        Path: The absolute path within the project structure.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        # Prepend project root for relative paths
        path = PROJ / path
    return path.resolve()


def get_relative_path(path: ty.Union[str, Path]) -> Path:
    """Return the path relative to the project directory.

    Args:
        path: Path or string to convert.

    Returns:
        Path: A path relative to :data:`PROJ`.
    """
    return get_path(path).relative_to(PROJ)


def duplicate_path(src: ty.Union[str, Path], alternative_project_dir: ty.Union[str, Path]) -> None:
    """Duplicate a file or directory to another project tree.

    If the destination exists, a timestamp suffix is appended to avoid
    overwriting the existing file or directory.

    Args:
        src: Source path relative to the current project.
        alternative_project_dir: Destination project directory.
    """
    src = get_path(src)
    alternative_project_dir = get_path(alternative_project_dir)
    dst = alternative_project_dir / src.relative_to(PROJ)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(dst.name + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)
