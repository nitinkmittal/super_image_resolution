"""Contains valiation checks for arguments."""
from typing import Any


def validate_arg_type(arg: Any, type_: Any):
    """
    Validate argument against required type.

    Raises ValueError otherwise.
    """
    if not isinstance(arg, type_):
        raise ValueError(f"Expected {type_}, got {type(arg)}.")


def validate_arg_dim(arg: Any, ndim: int):
    """
    Validate argument against required dimension.

    Raises ValueError otherwise.
    """
    if arg.ndim != ndim:
        raise ValueError(f"Expected {ndim}-D {type(arg)}, " f"got {ndim}-D {type(arg)}")
