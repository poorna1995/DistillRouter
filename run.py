#!/usr/bin/env python3
"""DistillRouter — single entry point for the project.

Pipeline: oracle-label (ground truth) -> label-data (teacher) ->
build-student-data (join query + teacher output) -> train-student-{classifier,dual}
(route-only / reasoning-plus-route students) -> select-router-checkpoint
(validation-only selection) -> evaluate-router (test, once).

Usage:
    python run.py list-datasets
    python run.py prepare-data --dataset gsm8k
    python run.py prepare-data --all

    python run.py check-splits --all

    python run.py list-teachers
    python run.py label-data --dataset gsm8k --teacher qwen2.5-14b
    python run.py label-data --all --split train validation

    python run.py oracle-label --dataset gsm8k --split calibration
    python run.py oracle-label --all --split calibration

    python run.py build-student-data --all --teacher qwen2.5-14b

    python run.py train-student-classifier --all --teacher qwen2.5-14b
    python run.py train-student-dual --all --teacher qwen2.5-14b

    python run.py select-router-checkpoint --variant classifier \\
        --checkpoint-root checkpoints/student-classifier --all --teacher qwen2.5-14b
    python run.py evaluate-router --variant classifier \\
        --checkpoint-dir checkpoints/student-classifier-best --all --teacher qwen2.5-14b --split test

New subcommands go in build_parser() below rather than new top-level scripts.
"""
from __future__ import annotations

import argparse

from common.config import (
    STUDENT_MODEL_ID, EPOCHS, LAMBDA_REASON, LAMBDA_ROUTE, LEARNING_RATE, LR_SCHEDULER,
    OVERSAMPLE_RATIO, ROUTE_HEAD_DROPOUT, ROUTE_LOSS, WARMUP_RATIO,
)
from dataset import datasets, get_loader_class, training_datasets


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


def cmd_build_student_data(args: argparse.Namespace) -> None:
    from student.data import build_distillation_dataset
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = training_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    for name in targets:
        for split in args.split:
            build_distillation_dataset(
                name, teacher_cls.prompt_version, output_version=teacher_cls.output_version, split=split,
            )


def cmd_train_student_dual(args: argparse.Namespace) -> None:
    from pathlib import Path

    from common.config import PROJECT_ROOT
    from student.train_dual import train_student_dual
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = training_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        PROJECT_ROOT / "checkpoints" / "student-dual"
    )
    train_student_dual(
        targets,
        teacher_cls.output_version,
        checkpoint_dir,
        model_id=args.model_id or STUDENT_MODEL_ID,
        split=args.split,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lambda_route=args.lambda_route,
        lambda_reason=args.lambda_reason,
        route_head_dropout=args.route_head_dropout,
        oversample_ratio=args.oversample_ratio,
        loss_mode=args.route_loss,
        seed=args.seed,
    )


def cmd_train_student_classifier(args: argparse.Namespace) -> None:
    from pathlib import Path

    from common.config import PROJECT_ROOT
    from student.train_classifier import train_student_classifier
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    targets = training_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        PROJECT_ROOT / "checkpoints" / "student-classifier"
    )
    train_student_classifier(
        targets,
        teacher_cls.output_version,
        checkpoint_dir,
        model_id=args.model_id or STUDENT_MODEL_ID,
        split=args.split,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        route_head_dropout=args.route_head_dropout,
        oversample_ratio=args.oversample_ratio,
        loss_mode=args.route_loss,
        seed=args.seed,
    )


def cmd_evaluate_router(args: argparse.Namespace) -> None:
    from evaluation.router_eval import evaluate_router_checkpoint
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    if args.variant == "classifier":
        from student.train_classifier import load_classifier_checkpoint, predict_route
        load_fn = load_classifier_checkpoint
    else:
        from student.train_dual import load_dual_head_checkpoint, predict_route
        load_fn = load_dual_head_checkpoint

    targets = datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    for name in targets:
        print(f"\n=== Evaluating {args.variant} router on '{name}/{args.split}' (vs teacher '{args.teacher}') ===")
        report = evaluate_router_checkpoint(
            load_fn, predict_route, args.checkpoint_dir, name, teacher_cls.output_version, split=args.split,
        )
        print(f"  accuracy={report.get('accuracy')}, macro_f1={report.get('macro_f1')}, "
              f"avg_latency_seconds={report.get('latency', {}).get('student_avg_latency_seconds')}")


def cmd_select_router_checkpoint(args: argparse.Namespace) -> None:
    from evaluation.metrics import freeze_checkpoint
    from evaluation.router_eval import select_best_router_checkpoint
    from teacher.base import get_teacher_class  # lazy: pulls in torch/transformers at import time

    if args.variant == "classifier":
        from student.train_classifier import load_classifier_checkpoint, predict_route
        load_fn = load_classifier_checkpoint
    else:
        from student.train_dual import load_dual_head_checkpoint, predict_route
        load_fn = load_dual_head_checkpoint

    targets = training_datasets() if args.all else args.dataset
    if not targets:
        print("Nothing to do: pass --dataset <name> [<name> ...] or --all.")
        return
    teacher_cls = get_teacher_class(args.teacher)
    result = select_best_router_checkpoint(
        load_fn, predict_route, args.checkpoint_root, targets, teacher_cls.output_version, split=args.split,
    )
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

    p_build_student_data = subparsers.add_parser(
        "build-student-data",
        help="Join teacher output with query text into the {query, teacher_route, routing_reasoning} "
        "records both student variants train on.",
    )
    p_build_student_data.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to build, e.g. --dataset gsm8k math",
    )
    p_build_student_data.add_argument("--all", action="store_true", help="Build for every registered dataset.")
    p_build_student_data.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend whose output to use. See `list-teachers`.",
    )
    p_build_student_data.add_argument(
        "--split", nargs="+", default=["train"], metavar="SPLIT",
        help="Canonical split(s) to build (default: train).",
    )
    p_build_student_data.set_defaults(func=cmd_build_student_data)

    p_train_dual = subparsers.add_parser(
        "train-student-dual",
        help="Train the reasoning-plus-route student: a route classifier head + a "
        "reasoning LM head on one shared trunk, jointly -- reasoning as a training-only auxiliary "
        "signal, never generated at inference. See student/train_dual.py's module docstring.",
    )
    p_train_dual.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to pool into one training set, e.g. --dataset gsm8k math",
    )
    p_train_dual.add_argument("--all", action="store_true", help="Pool every registered dataset.")
    p_train_dual.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend whose distillation records to train on (must have run build-student-data first).",
    )
    p_train_dual.add_argument(
        "--split", default="train", metavar="SPLIT", help="Split to train on (default: train).",
    )
    p_train_dual.add_argument(
        "--model-id", default=None, metavar="HF_ID",
        help=f"Base model to fine-tune (default: {STUDENT_MODEL_ID}).",
    )
    p_train_dual.add_argument("--epochs", type=float, default=EPOCHS, help=f"Training epochs (default: {EPOCHS}).")
    p_train_dual.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, metavar="LR",
        help=f"AdamW peak learning rate (default: {LEARNING_RATE}).",
    )
    p_train_dual.add_argument(
        "--warmup-ratio", type=float, default=WARMUP_RATIO, metavar="R",
        help="Fraction of total steps to linearly ramp the LR up from 0 to --learning-rate before "
        "--lr-scheduler-type's decay takes over (default: 0.0 = no warmup, Trainer's own default).",
    )
    p_train_dual.add_argument(
        "--lr-scheduler-type", default=LR_SCHEDULER, metavar="TYPE",
        help="HF Trainer schedule name (default: linear -- decays --learning-rate to 0 over the full "
        "run). Common alternatives: cosine, constant, constant_with_warmup. The decay curve is computed "
        "against --epochs * steps-per-epoch, so it's specific to the epoch budget this run uses.",
    )
    p_train_dual.add_argument(
        "--lambda-route", type=float, default=LAMBDA_ROUTE, metavar="W", help="Weight on the route classifier loss.",
    )
    p_train_dual.add_argument(
        "--lambda-reason", type=float, default=LAMBDA_REASON, metavar="W", help="Weight on the reasoning LM loss.",
    )
    p_train_dual.add_argument(
        "--route-head-dropout", type=float, default=ROUTE_HEAD_DROPOUT, metavar="P",
        help="Dropout inside the route classifier's MLP (LayerNorm -> Linear -> GELU -> Dropout -> Linear).",
    )
    p_train_dual.add_argument(
        "--oversample-ratio", type=float, default=OVERSAMPLE_RATIO, metavar="R",
        help="Oversample minority routing tiers toward the majority tier's count before training "
        "(0.0=off, 1.0=full balance, 0.5=halfway). See student/data.py's "
        "oversample_minority_tiers -- duplicated examples carry their routing_reasoning too, so "
        "this affects the reasoning LM loss as well as the route classifier loss.",
    )
    p_train_dual.add_argument(
        "--route-loss", default=ROUTE_LOSS, choices=["soft", "hard", "hard+soft"],
        help="Routing loss against the teacher's votes (default: %(default)s). See experiment E6.",
    )
    p_train_dual.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed, set before the model (and route_head's random init) is constructed -- "
        "not just TrainingArguments' own seed, which is too late for that (default: 42).",
    )
    p_train_dual.add_argument(
        "--checkpoint-dir", default=None, metavar="PATH",
        help="Output dir for the trained checkpoint (default: checkpoints/student-dual/).",
    )
    p_train_dual.set_defaults(func=cmd_train_student_dual)

    p_train_classifier = subparsers.add_parser(
        "train-student-classifier",
        help="Train the route-only student: route_head -> "
        "CrossEntropy(route), no reasoning anywhere. The control for whether reasoning "
        "supervision (train-student-dual) helps. "
        "See student/train_classifier.py's module docstring.",
    )
    p_train_classifier.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset name(s) to pool into one training set, e.g. --dataset gsm8k math",
    )
    p_train_classifier.add_argument("--all", action="store_true", help="Pool every registered dataset.")
    p_train_classifier.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend whose distillation records to train on (must have run build-student-data first "
        "-- only teacher_route is used, routing_reasoning is read but ignored).",
    )
    p_train_classifier.add_argument(
        "--split", default="train", metavar="SPLIT", help="Split to train on (default: train).",
    )
    p_train_classifier.add_argument(
        "--model-id", default=None, metavar="HF_ID",
        help=f"Base model to fine-tune (default: {STUDENT_MODEL_ID}).",
    )
    p_train_classifier.add_argument("--epochs", type=float, default=EPOCHS, help=f"Training epochs (default: {EPOCHS}).")
    p_train_classifier.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, metavar="LR",
        help=f"AdamW peak learning rate (default: {LEARNING_RATE}).",
    )
    p_train_classifier.add_argument(
        "--warmup-ratio", type=float, default=WARMUP_RATIO, metavar="R",
        help="Fraction of total steps to linearly ramp the LR up from 0 to --learning-rate before "
        "--lr-scheduler-type's decay takes over (default: 0.0 = no warmup, Trainer's own default).",
    )
    p_train_classifier.add_argument(
        "--lr-scheduler-type", default=LR_SCHEDULER, metavar="TYPE",
        help="HF Trainer schedule name (default: linear -- decays --learning-rate to 0 over the full "
        "run). Common alternatives: cosine, constant, constant_with_warmup. The decay curve is computed "
        "against --epochs * steps-per-epoch, so it's specific to the epoch budget this run uses.",
    )
    p_train_classifier.add_argument(
        "--route-head-dropout", type=float, default=ROUTE_HEAD_DROPOUT, metavar="P",
        help="Dropout inside the route classifier's MLP (LayerNorm -> Linear -> GELU -> Dropout -> Linear).",
    )
    p_train_classifier.add_argument(
        "--oversample-ratio", type=float, default=OVERSAMPLE_RATIO, metavar="R",
        help="Oversample minority routing tiers toward the majority tier's count before training "
        "(0.0=off, 1.0=full balance, 0.5=halfway). See student/data.py's "
        "oversample_minority_tiers.",
    )
    p_train_classifier.add_argument(
        "--route-loss", default=ROUTE_LOSS, choices=["soft", "hard", "hard+soft"],
        help="Routing loss against the teacher's votes (default: %(default)s). See experiment E6.",
    )
    p_train_classifier.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed, set before the model (and route_head's random init) is constructed (default: 42).",
    )
    p_train_classifier.add_argument(
        "--checkpoint-dir", default=None, metavar="PATH",
        help="Output dir for the trained checkpoint (default: checkpoints/student-classifier/).",
    )
    p_train_classifier.set_defaults(func=cmd_train_student_classifier)

    p_eval_router = subparsers.add_parser(
        "evaluate-router",
        help="Routing accuracy vs teacher, per-tier P/R/F1, confusion matrix, and latency for a "
        "single-forward-pass router checkpoint (train-student-classifier or train-student-dual).",
    )
    p_eval_router.add_argument(
        "--variant", required=True, choices=["classifier", "dual"],
        help="Which router architecture the checkpoint is (classifier=train-student-classifier, "
        "dual=train-student-dual) -- picks the matching load_*_checkpoint/predict_route pair.",
    )
    p_eval_router.add_argument(
        "--checkpoint-dir", required=True, metavar="PATH",
        help="Trained router checkpoint (from train-student-classifier or train-student-dual).",
    )
    p_eval_router.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset(s) to evaluate on, e.g. --dataset gsm8k math",
    )
    p_eval_router.add_argument("--all", action="store_true", help="Evaluate on every registered dataset.")
    p_eval_router.add_argument(
        "--teacher", required=True, metavar="NAME",
        help="Teacher backend the checkpoint was distilled from (for accuracy/latency comparison).",
    )
    p_eval_router.add_argument(
        "--split", default="validation", metavar="SPLIT",
        help="Split to evaluate on (default: validation -- use a split the checkpoint wasn't trained on).",
    )
    p_eval_router.set_defaults(func=cmd_evaluate_router)

    p_select_router = subparsers.add_parser(
        "select-router-checkpoint",
        help="Evaluate every checkpoint-<step> saved by train-student-classifier or train-student-dual "
        "(one per epoch) on validation and report the one with the highest macro F1. Pass "
        "--freeze-to to also copy the winner.",
    )
    p_select_router.add_argument(
        "--variant", required=True, choices=["classifier", "dual"],
        help="Which router architecture the checkpoints are.",
    )
    p_select_router.add_argument(
        "--checkpoint-root", required=True, metavar="PATH",
        help="train-student-classifier's/train-student-dual's --checkpoint-dir "
        "(the directory containing checkpoint-<step> subdirs).",
    )
    p_select_router.add_argument(
        "--dataset", nargs="+", default=[], metavar="NAME",
        help="Dataset(s) to pool for scoring, e.g. --dataset gsm8k math",
    )
    p_select_router.add_argument("--all", action="store_true", help="Pool every registered dataset.")
    p_select_router.add_argument(
        "--teacher", required=True, metavar="NAME", help="Teacher backend the checkpoints were distilled from.",
    )
    p_select_router.add_argument(
        "--split", default="validation", metavar="SPLIT", help="Split to score on (default: validation).",
    )
    p_select_router.add_argument(
        "--freeze-to", default=None, metavar="PATH",
        help="Copy the winning checkpoint here. Deliberate, opt-in — omit to just see the report.",
    )
    p_select_router.set_defaults(func=cmd_select_router_checkpoint)

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
