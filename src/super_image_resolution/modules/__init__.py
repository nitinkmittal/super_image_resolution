"""Init module."""
from pathlib import Path, PosixPath


def get_root_path() -> PosixPath:
    """Return root directory for the project."""
    return Path(__file__).parent.parent.parent
