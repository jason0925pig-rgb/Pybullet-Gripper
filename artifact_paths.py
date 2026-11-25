"""Identify experiment artifacts relative to the repository when possible."""

import hashlib
from pathlib import Path
from typing import Union


ROOT = Path(__file__).resolve().parent


def artifact_key(path: Union[str, Path]) -> str:
    """Keep external artifact paths absolute and repository paths portable."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return str(resolved)


def sha256(path: Union[str, Path]) -> str:
    """Hash an absolute path or a repository-relative artifact key."""
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
