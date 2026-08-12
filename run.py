#!/usr/bin/env python3
"""DistillRouter — single entry point for the project.

Usage:
    python run.py list-datasets
    python run.py prepare-data --dataset gsm8k
    python run.py prepare-data --all

    python run.py check-splits --all

    python run.py list-teachers
    python run.py label-data --dataset gsm8k --teacher qwen2.5-3b-v2
    python run.py label-data --all --split train validation

    python run.py oracle-label --dataset gsm8k --split calibration
    python run.py oracle-label --all --split calibration

    python run.py build-sft-data --all --teacher qwen2.5-3b-v2

Not yet implemented: student SFT training itself, and a CLI subcommand
for evaluation/oracle_check.py's teacher-vs-oracle check. New subcommands
go in build_parser() below rather than new top-level scripts.
"""
from __future__ import annotations

import argparse
import json

from dataset import datasets, get_loader_class


def cmd_list_datasets(_args: argparse.Namespace) -> None:
    names = datasets()
    if not names:
        print("No datasets registered.")
        return
    print("Registered datasets:")
    for name in names:
        print(f"  - {name}")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    targets = datasets() if args.all else args.dataset
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


def cmd_check_splits(args: argparse.Namespace) -> None:
    from common.split_integrity import SplitLeakageError, check_dataset_splits

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    failures = []
    for name in targets:
        try:
            check_dataset_splits(name)
        except SplitLeakageError as e:
            failures.append(name)
            print(f"FAIL '{name}':\n  {e}")
        else:
            print(f"OK   '{name}': train/validation/test/calibration are disjoint.")
    if failures:
        raise SystemExit(f"\ncheck-splits failed for: {', '.join(failures)}")


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

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    for name in targets:
        for split in args.split:
            print(f"\n=== Labeling '{name}' ({split}) with teacher '{args.teacher}' ===")
            stats = label_dataset(
                name,
                args.teacher,
                split=split,
                limit=args.limit,
                allow_self_referential_calibration=args.allow_self_referential_calibration,
            )
            print(
                f"  cache hits: {stats['cache_hits']}, deduped: {stats['deduped']}, "
                f"newly labeled: {stats['newly_labeled']} -> {stats['output_path']} "
                f"({stats['total_cached']} total cached)"
            )


def cmd_build_sft_data(args: argparse.Namespace) -> None:
    from student.data import build_sft_records
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    for name in targets:
        for split in args.split:
            build_sft_records(
                name, teacher_cls.prompt_version, output_version=teacher_cls.output_version, split=split,
                include_reasoning=not args.route_only,
            )


def cmd_train_student(args: argparse.Namespace) -> None:
    from pathlib import Path

    from common.config import PROJECT_ROOT
    from student.train import DEFAULT_STUDENT_MODEL_ID, train_student
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    variant = "sft-route-only" if args.route_only else "sft"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        PROJECT_ROOT / "checkpoints" / f"student-{variant}"
    )
    train_student(
        targets,
        teacher_cls.output_version,
        checkpoint_dir,
        model_id=args.model_id or DEFAULT_STUDENT_MODEL_ID,
        variant=variant,
        split=args.split,
        num_train_epochs=args.epochs,
    )


def cmd_evaluate_student(args: argparse.Namespace) -> None:
    from pathlib import Path

    from evaluation.student_eval import (
        generate_student_predictions,
        latency_comparison,
        reasoning_diversity,
        reasoning_embedding_similarity,
        route_accuracy_vs_teacher,
        save_predictions,
        teacher_agreement_report,
    )
    from common.schema import TeacherLabel, read_jsonl
    from common.config import TEACHER_DIR
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    for name in targets:
        print(f"\n=== Evaluating student on '{name}/{args.split}' (vs teacher '{args.teacher}') ===")
        predictions = generate_student_predictions(args.checkpoint_dir, name, args.split)

        accuracy = route_accuracy_vs_teacher(predictions, name, teacher_cls.output_version, args.split)
        print("Routing accuracy vs teacher:", accuracy)
        agreement = teacher_agreement_report(predictions, name, teacher_cls.output_version, args.split)
        print("Teacher Agreement / Distillation Fidelity (macro F1, per-tier P/R/F1, confusion matrix):", agreement)

        report = {"teacher_agreement_accuracy": accuracy, "teacher_agreement_distillation_fidelity": agreement}

        teacher_path = TEACHER_DIR / name / teacher_cls.output_version / f"{args.split}.jsonl"
        if teacher_path.exists():
            teacher_labels = read_jsonl(teacher_path, TeacherLabel)
            latency = latency_comparison(predictions, [label.latency_seconds for label in teacher_labels])
            print("Latency:", latency)
            report["latency"] = latency

        diversity = reasoning_diversity([p["reasoning"] for p in predictions])
        print("Reasoning diversity (exact-match):", diversity)
        similarity = reasoning_embedding_similarity([p["reasoning"] for p in predictions])
        print("Reasoning diversity (embedding similarity, lower = more diverse):", similarity)
        report["reasoning_diversity"] = {"exact_match": diversity, "embedding_similarity": similarity}

        eval_dir = Path(args.checkpoint_dir) / f"eval_{name}_{args.split}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
        n_saved = save_predictions(
            predictions, name, teacher_cls.output_version, args.split, str(eval_dir / "predictions.jsonl")
        )
        print(f"Saved classification_report.json + predictions.jsonl ({n_saved} rows) -> {eval_dir}")


def cmd_select_checkpoint(args: argparse.Namespace) -> None:
    from evaluation.student_eval import freeze_checkpoint, select_best_checkpoint
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    result = select_best_checkpoint(args.checkpoint_root, targets, teacher_cls.output_version, split=args.split)
    if args.freeze_to:
        freeze_checkpoint(result["best"]["checkpoint"], args.freeze_to)


def oracle_label(args: argparse.Namespace) -> None:
    from oracle.labeler import label_dataset

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    for name in targets:
        for split in args.split:
            print(f"\n=== Oracle-labeling '{name}' ({split}) — every tier, every query ===")
            stats = label_dataset(name, split=split, limit=args.limit)
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

    p_check_splits = subparsers.add_parser(
        "check-splits",
        help="Assert train/validation/test/calibration share no example id or exact query text.",
    )
    p_check_splits.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to check, e.g. --dataset gsm8k math",
    )
    p_check_splits.add_argument("--all", action="store_true", help="Check every registered dataset.")
    p_check_splits.set_defaults(func=cmd_check_splits)

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
        "--teacher", required=True, metavar="NAME",
        help="Registered teacher backend to use. See `list-teachers`. No default — every "
        "teacher makes real model calls, so this must be chosen explicitly.",
    )
    p_label.add_argument(
        "--split", nargs="+", default=["train"], metavar="SPLIT",
        help="Canonical split(s) to label (default: train).",
    )
    p_label.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only label the first N examples of each split (for trying a real teacher before a full run).",
    )
    p_label.add_argument(
        "--allow-self-referential-calibration", action="store_true",
        help="Debugging-only escape hatch: by default, labeling a teacher's own few-shot source "
        "split is refused (self-referential predictions). Pass this flag to override.",
    )
    p_label.set_defaults(func=cmd_label_data)

    p_build_sft = subparsers.add_parser(
        "build-sft-data", help="Join teacher output with query text into the student's SFT training records."
    )
    p_build_sft.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to build, e.g. --dataset gsm8k math",
    )
    p_build_sft.add_argument("--all", action="store_true", help="Build for every registered dataset.")
    p_build_sft.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend whose output to use. See `list-teachers`.",
    )
    p_build_sft.add_argument(
        "--split", nargs="+", default=["train"], metavar="SPLIT",
        help="Canonical split(s) to build (default: train).",
    )
    p_build_sft.add_argument(
        "--route-only", action="store_true",
        help="Ablation variant: drop <REASON> from the target, route-only. "
        "Writes alongside the reasoning+route baseline, doesn't replace it.",
    )
    p_build_sft.set_defaults(func=cmd_build_sft_data)

    p_train = subparsers.add_parser(
        "train-student", help="SFT a small causal LM on the student's training records (TRL SFTTrainer)."
    )
    p_train.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to pool into one training set, e.g. --dataset gsm8k math",
    )
    p_train.add_argument("--all", action="store_true", help="Pool every registered dataset.")
    p_train.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend whose SFT records to train on (must have run build-sft-data first).",
    )
    p_train.add_argument(
        "--route-only", action="store_true",
        help="Train on the route-only ablation variant instead of the reasoning+route baseline.",
    )
    p_train.add_argument("--split", default="train", metavar="SPLIT", help="Split to train on (default: train).")
    p_train.add_argument(
        "--model-id", default=None, metavar="HF_ID",
        help="Base model to fine-tune (default: google/gemma-3-270m-it, see student/train.py).",
    )
    p_train.add_argument("--epochs", type=float, default=3.0, help="Training epochs (default: 3).")
    p_train.add_argument(
        "--checkpoint-dir", default=None, metavar="PATH",
        help="Output dir for the trained checkpoint (default: checkpoints/student-<variant>/).",
    )
    p_train.set_defaults(func=cmd_train_student)

    p_eval_student = subparsers.add_parser(
        "evaluate-student",
        help="Routing accuracy vs teacher, latency, and reasoning diversity for a trained checkpoint. "
        "For cross-domain generalization, just pass a checkpoint trained on one dataset and "
        "--dataset for a different one — no separate mode needed.",
    )
    p_eval_student.add_argument(
        "--checkpoint-dir", required=True, metavar="PATH", help="Trained student checkpoint (from train-student).",
    )
    p_eval_student.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset(s) to evaluate on, e.g. --dataset gsm8k math",
    )
    p_eval_student.add_argument("--all", action="store_true", help="Evaluate on every registered dataset.")
    p_eval_student.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend the checkpoint was distilled from (for accuracy/latency comparison).",
    )
    p_eval_student.add_argument(
        "--split", default="validation", metavar="SPLIT",
        help="Split to evaluate on (default: validation — use a split the checkpoint wasn't trained on).",
    )
    p_eval_student.set_defaults(func=cmd_evaluate_student)

    p_select = subparsers.add_parser(
        "select-checkpoint",
        help="Evaluate every checkpoint-<step> saved by train-student (one per epoch) on validation "
        "and report the one with the highest macro F1. Pass --freeze-to to also copy the winner.",
    )
    p_select.add_argument(
        "--checkpoint-root", required=True, metavar="PATH",
        help="train-student's --checkpoint-dir (the directory containing checkpoint-<step> subdirs).",
    )
    p_select.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset(s) to pool for scoring, e.g. --dataset gsm8k math",
    )
    p_select.add_argument("--all", action="store_true", help="Pool every registered dataset.")
    p_select.add_argument(
        "--teacher", required=True, metavar="NAME", help="Teacher backend the checkpoints were distilled from.",
    )
    p_select.add_argument(
        "--split", default="validation", metavar="SPLIT", help="Split to score on (default: validation).",
    )
    p_select.add_argument(
        "--freeze-to", default=None, metavar="PATH",
        help="Copy the winning checkpoint here (e.g. checkpoints/student-best/). Deliberate, opt-in — "
        "omit to just see the report and freeze manually later.",
    )
    p_select.set_defaults(func=cmd_select_checkpoint)

    p_oracle = subparsers.add_parser(
        "oracle-label", help="Run every candidate tier on every query to derive ground-truth routing labels."
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
    p_oracle.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process the first N examples of each split (candidate inference has real cost). "
        "A later, larger --limit on the same split only computes the newly-added queries.",
    )
    p_oracle.set_defaults(func=oracle_label)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
