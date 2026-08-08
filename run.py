#!/usr/bin/env python3
"""DistillRouter — single entry point for the project.

Usage:
    python run.py list-datasets
    python run.py prepare-data --dataset gsm8k
    python run.py prepare-data --dataset gsm8k math
    python run.py prepare-data --all

As the project grows (teacher labeling, student distillation, deployment
evaluation — see teacher/, student/, router/), new subcommands are added
to `build_parser()` below rather than new top-level scripts. This file is
meant to stay the one place you run.
"""
from __future__ import annotations

import argparse

from dataset import available_datasets, get_loader_class


def cmd_list_datasets(_args: argparse.Namespace) -> None:
    names = available_datasets()
    if not names:
        print("No datasets registered.")
        return
    print("Registered datasets:")
    for name in names:
        print(f"  - {name}")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    targets = available_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    for name in targets:
        loader_cls = get_loader_class(name)
        loader = loader_cls()
        print(f"\n=== Preparing '{name}' ===")
        counts = loader.prepare()
        for split, n in counts.items():
            print(f"  {split}: {n} examples -> data/processed/{name}/{split}.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list-datasets", help="List all registered dataset loaders.")
    p_list.set_defaults(func=cmd_list_datasets)

    p_prepare = subparsers.add_parser(
        "prepare-data", help="Download and normalize one or more datasets into data/processed/."
    )
    p_prepare.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to prepare, e.g. --dataset gsm8k math",
    )
    p_prepare.add_argument("--all", action="store_true", help="Prepare every registered dataset.")
    p_prepare.set_defaults(func=cmd_prepare_data)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
