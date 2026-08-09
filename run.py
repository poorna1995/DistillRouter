#!/usr/bin/env python3
"""DistillRouter — single entry point for the project.

Usage:
    python run.py list-datasets
    python run.py prepare-data --dataset gsm8k
    python run.py prepare-data --dataset gsm8k math
    python run.py prepare-data --all

    python run.py list-teachers
    python run.py label-data --dataset gsm8k --teacher heuristic
    python run.py label-data --all --split train validation

    python run.py oracle-label --dataset gsm8k --split calibration
    python run.py oracle-label --all --split calibration

As the project grows (student distillation, deployment evaluation — see
student/, router/), new subcommands are added to `build_parser()` below
rather than new top-level scripts. This file is meant to stay the one
place you run.
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


def cmd_list_teachers(_args: argparse.Namespace) -> None:
    from teacher import available_teachers  # lazy: teacher/ pulls in torch/transformers at import time

    names = available_teachers()
    if not names:
        print("No teachers registered.")
        return
    print("Registered teachers:")
    for name in names:
        print(f"  - {name}")


def cmd_label_data(args: argparse.Namespace) -> None:
    from teacher.labeler import label_dataset

    targets = available_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    for name in targets:
        for split in args.split:
            print(f"\n=== Labeling '{name}' ({split}) with teacher '{args.teacher}' ===")
            stats = label_dataset(name, args.teacher, split=split, limit=args.limit)
            print(
                f"  cache hits: {stats['cache_hits']}, deduped: {stats['deduped']}, "
                f"newly labeled: {stats['newly_labeled']} -> data/teacher/{name}/{split}.jsonl "
                f"({stats['total_cached']} total cached)"
            )


def cmd_oracle_label(args: argparse.Namespace) -> None:
    from oracle.labeler import label_dataset

    targets = available_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    for name in targets:
        for split in args.split:
            print(f"\n=== Oracle-labeling '{name}' ({split}) ===")
            stats = label_dataset(name, split=split)
            print(
                f"  cache hits: {stats['cache_hits']}, new attempts: {stats['new_attempts']}, "
                f"labels by tier: {stats['labels_by_tier']}, no tier succeeded: {stats['no_tier_succeeded']} "
                f"-> data/oracle/{name}/{split}.labels.jsonl"
            )


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

    p_list_teachers = subparsers.add_parser("list-teachers", help="List all registered teacher backends.")
    p_list_teachers.set_defaults(func=cmd_list_teachers)

    p_label = subparsers.add_parser(
        "label-data", help="Run a teacher over processed data, caching labels incrementally."
    )
    p_label.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to label, e.g. --dataset gsm8k math",
    )
    p_label.add_argument("--all", action="store_true", help="Label every registered dataset.")
    p_label.add_argument(
        "--teacher", default="heuristic", metavar="NAME",
        help="Registered teacher backend to use (default: heuristic). See `list-teachers`.",
    )
    p_label.add_argument(
        "--split", nargs="+", default=["train"], metavar="SPLIT",
        help="Canonical split(s) to label (default: train).",
    )
    p_label.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only label the first N examples of each split (for trying a real teacher before a full run).",
    )
    p_label.set_defaults(func=cmd_label_data)

    p_oracle = subparsers.add_parser(
        "oracle-label", help="Execute candidate tiers in cost order to derive ground-truth routing labels."
    )
    p_oracle.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to oracle-label, e.g. --dataset gsm8k math",
    )
    p_oracle.add_argument("--all", action="store_true", help="Oracle-label every registered dataset.")
    p_oracle.add_argument(
        "--split", nargs="+", default=["calibration"], metavar="SPLIT",
        help="Canonical split(s) to oracle-label (default: calibration).",
    )
    p_oracle.set_defaults(func=cmd_oracle_label)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
