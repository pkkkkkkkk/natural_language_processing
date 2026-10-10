"""Task 1: Binary Bag of Words / Word Frequency + Logistic Regression."""

import argparse
import csv
import platform
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


from common import (
    ROOT, TOKEN_PATTERN, default_data_path, evaluate, load_dataset,
    load_or_create_shared_split, make_split, save_json,
)


def run_experiment(name, binary, texts, labels, split, args):
    """只用训练集拟合词表和分类器，验证集和测试集只用于预测。"""
    train, validation, test = (split[key] for key in ("train", "validation", "test"))
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    binary=binary,
                    lowercase=True,
                    token_pattern=TOKEN_PATTERN,
                    dtype=np.float64,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    solver="lbfgs",
                    max_iter=args.max_iter,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    print(f"\nTraining {name} ...", flush=True)
    started = time.perf_counter()
    # 记录收敛状态，同时仍将训练中的所有警告显示出来。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(texts[train], labels[train])
    fit_seconds = time.perf_counter() - started
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    for item in caught:
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)

    validation_pred = pipeline.predict(texts[validation])
    test_pred = pipeline.predict(texts[test])
    validation_metrics = evaluate(labels[validation], validation_pred)
    test_metrics = evaluate(labels[test], test_pred)
    classifier = pipeline.named_steps["classifier"]
    result = {
        "method": name,
        "n_features": len(pipeline.named_steps["vectorizer"].vocabulary_),
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "fit_seconds": round(fit_seconds, 3),
        "n_iter": int(classifier.n_iter_.max()),
        "converged": converged,
    }

    classes = classifier.classes_.tolist()
    save_json(
        args.output / f"{name}_test_report.json",
        {
            "class_order": classes,
            "confusion_matrix": confusion_matrix(labels[test], test_pred, labels=classes).tolist(),
            "classification_report": classification_report(
                labels[test], test_pred, labels=classes, output_dict=True, zero_division=0
            ),
        },
    )
    with (args.output / f"{name}_test_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["row_index", "true_label", "predicted_label"])
        writer.writerows(zip(test.tolist(), labels[test].tolist(), test_pred.tolist()))
    joblib.dump(pipeline, args.model_dir / f"{name}_model.joblib", compress=3)

    print(
        f"{name}: vocabulary={result['n_features']}, "
        f"fit={fit_seconds:.1f}s, iterations={result['n_iter']}, converged={converged}\n"
        f"  Validation: Accuracy={validation_metrics['accuracy']:.4f}, "
        f"Macro-F1={validation_metrics['macro_f1']:.4f}\n"
        f"  Test:       Accuracy={test_metrics['accuracy']:.4f}, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path("nyt.csv"))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "task1")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data" / "split_indices.json",
                        help="已有划分校验后复用，缺失时自动生成")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "task1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="Logistic Regression 的 C 参数")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4, help="限制数值计算线程数")
    args = parser.parse_args()
    if args.c <= 0 or args.max_iter <= 0 or args.threads <= 0:
        parser.error("--c、--max-iter 和 --threads 必须为正数。")

    texts, labels = load_dataset(args.data)
    split, split_metadata = load_or_create_shared_split(args.split_file, args.data, labels, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = split_metadata["dataset_sha256"]
    class_counts = {
        name: dict(sorted(Counter(labels[indices].tolist()).items()))
        for name, indices in split.items()
    }
    print(f"Dataset: {args.data.resolve()} ({len(labels)} samples)", flush=True)
    print(f"Using shared split: {args.split_file.resolve()} (seed={split_metadata['seed']})", flush=True)
    for name, indices in split.items():
        print(f"  {name}: {len(indices)} samples, classes={class_counts[name]}", flush=True)

    # 两次实验复用同一份 split，CountVectorizer 始终输出稀疏矩阵。
    with threadpool_limits(limits=args.threads):
        results = [
            run_experiment(name, binary, texts, labels, split, args)
            for name, binary in (("binary_bow", True), ("word_frequency", False))
        ]

    with (args.output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_json(
        args.output / "results.json",
        {
            "config": {
                "dataset": str(args.data.resolve()),
                "dataset_sha256": fingerprint,
                "seed": args.seed,
                "split_seed": split_metadata["seed"],
                "split_sizes": {name: len(indices) for name, indices in split.items()},
                "class_counts": class_counts,
                "lowercase": True,
                "token_pattern": TOKEN_PATTERN,
                "classifier": "LogisticRegression",
                "solver": "lbfgs",
                "C": args.c,
                "max_iter": args.max_iter,
                "threads": args.threads,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            },
            "results": results,
        },
    )
    print(f"\nResults saved to: {args.output.resolve()}", flush=True)
    print(f"Models saved to: {args.model_dir.resolve()}", flush=True)
    if not all(result["converged"] for result in results):
        raise SystemExit("存在未收敛模型，请增大 --max-iter 后重新运行。")


if __name__ == "__main__":
    main()
