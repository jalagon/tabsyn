"""Entry point for running TabSyn and baseline models."""

from __future__ import annotations

import argparse
import torch

from utils import execute_function, get_args


def configure_device(args: argparse.Namespace) -> str:
    """Determine and set the device for computation.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The name of the device that will be used, e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    args.device = device
    return device


def resolve_save_path(args: argparse.Namespace) -> str:
    """Populate ``args.save_path`` with a default path if it is missing.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The path where synthetic samples will be stored.
    """
    if not args.save_path:
        args.save_path = f"synthetic/{args.dataname}/{args.method}.csv"
    return args.save_path


def main() -> None:
    """Parse CLI arguments and run the selected method."""
    args: argparse.Namespace = get_args()
    configure_device(args)
    resolve_save_path(args)
    main_fn = execute_function(args.method, args.mode)
    main_fn(args)


if __name__ == "__main__":
    main()

