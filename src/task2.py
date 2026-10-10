"""Task 2: 100-dimensional GloVe / Word2Vec + Logistic Regression."""

import argparse
import csv
import platform
import re
import time
import warnings
import zlib
from pathlib import Path

import gensim
import joblib
import numpy as np
import sklearn
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from common import (
    ROOT, TOKEN_PATTERN, default_data_path, evaluate, file_sha256,
    load_dataset, load_or_create_shared_split, load_shared_split, save_json,
)


VECTOR_SIZE = 100
METHODS = ("glove", "word2vec_ag", "word2vec_nyt")
TOKENIZER = re.compile(TOKEN_PATTERN)




def tokenize(text):
    """复用 Task 1 的小写和分词规则，保留重复词。"""
    return TOKENIZER.findall(text.lower())


class CsvTextCorpus:
    """可重复遍历的 AG News 语料，每次只读取一条记录。"""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with self.path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if "text" not in (reader.fieldnames or []):
                raise ValueError("AG News CSV 必须包含 text 列。")
            for index, row in enumerate(reader):
                text = row.get("text")
                if not text or not text.strip():
                    raise ValueError(f"AG News 第 {index + 1} 条文本为空。")
                yield tokenize(text)


def stable_hash(word):
    """固定词初始化使用的哈希，避免 Python 进程随机哈希的影响。"""
    return zlib.crc32(word.encode("utf-8"))


class EpochProgress(CallbackAny2Vec):
    def __init__(self, name, epochs):
        self.name = name
        self.epochs = epochs
        self.epoch = 0

    def on_epoch_begin(self, model):
        self.started = time.perf_counter()
        print(f"  {self.name}: epoch {self.epoch + 1}/{self.epochs} ...", flush=True)

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"  epoch {self.epoch} finished in {time.perf_counter() - self.started:.1f}s", flush=True)


def train_word2vec(corpus, name, args):
    started = time.perf_counter()
    print(f"\nBuilding Word2Vec vocabulary: {name} ...", flush=True)
    model = Word2Vec(
        vector_size=VECTOR_SIZE,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        negative=5,
        sample=1e-3,
        workers=args.workers,
        seed=args.seed,
        hashfxn=stable_hash,
    )
    model.build_vocab(corpus)
    if not len(model.wv):
        raise ValueError(f"{name} 词表为空，请检查语料或降低 --min-count。")
    print(
        f"  documents={model.corpus_count}, tokens={model.corpus_total_words}, "
        f"vocabulary={len(model.wv)}",
        flush=True,
    )
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        callbacks=[EpochProgress(name, args.epochs)],
    )
    return model.wv, {
        "corpus_documents": int(model.corpus_count),
        "corpus_tokens": int(model.corpus_total_words),
        "embedding_seconds": round(time.perf_counter() - started, 3),
    }


def load_glove(path, needed_words):
    """读取已有 GloVe；仅保存 NYT 会查到的词以减少内存和模型体积。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 GloVe 文件：{path}\n请先运行 src/download_glove.py，"
            "或用 --glove-path 指定 glove.6B.100d.txt。"
        )
    print(f"\nLoading pretrained GloVe: {path.resolve()} ...", flush=True)
    started = time.perf_counter()
    words, rows = [], []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            word, separator, values = line.rstrip().partition(" ")
            if word not in needed_words:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if not separator or vector.size != VECTOR_SIZE or not np.isfinite(vector).all():
                raise ValueError(f"GloVe 第 {line_number} 行不是有效的 100 维词向量。")
            words.append(word)
            rows.append(vector)
    if not words:
        raise ValueError("GloVe 中没有找到任何 NYT token，请检查输入文件。")
    vectors = KeyedVectors(vector_size=VECTOR_SIZE)
    vectors.add_vectors(words, np.asarray(rows, dtype=np.float32))
    print(f"  loaded {len(vectors)} relevant word vectors", flush=True)
    return vectors, {"embedding_seconds": round(time.perf_counter() - started, 3)}


def mean_document_vectors(documents, vectors):
    """重复词重复计入平均；未知词跳过；无有效词的文档使用零向量。"""
    if vectors.vector_size != VECTOR_SIZE:
        raise ValueError("词向量维度必须为 100。")
    features = np.zeros((len(documents), VECTOR_SIZE), dtype=np.float64)
    total_tokens = known_tokens = zero_documents = 0
    for row, tokens in enumerate(documents):
        indices = [vectors.key_to_index[token] for token in tokens if token in vectors.key_to_index]
        total_tokens += len(tokens)
        known_tokens += len(indices)
        if indices:
            features[row] = vectors.vectors[indices].mean(axis=0, dtype=np.float64)
        else:
            zero_documents += 1
    return features, {
        "total_tokens": total_tokens,
        "known_tokens": known_tokens,
        "token_coverage": known_tokens / total_tokens if total_tokens else 0.0,
        "zero_documents": zero_documents,
    }


def run_classifier(name, vectors, documents, labels, split, args, embedding_info):
    features, coverage = {}, {}
    for subset, indices in split.items():
        features[subset], coverage[subset] = mean_document_vectors(
            [documents[index] for index in indices], vectors
        )
    if not coverage["train"]["known_tokens"]:
        raise ValueError(f"{name} 的训练集没有有效词向量，无法进行有意义的分类实验。")

    classifier = LogisticRegression(
        C=args.c, solver="lbfgs", max_iter=args.max_iter, random_state=args.seed
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features["train"], labels[split["train"]])
    fit_seconds = time.perf_counter() - started
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    for item in caught:
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)

    validation_pred = classifier.predict(features["validation"])
    test_pred = classifier.predict(features["test"])
    validation_metrics = evaluate(labels[split["validation"]], validation_pred)
    test_metrics = evaluate(labels[split["test"]], test_pred)
    result = {
        "method": name,
        "vector_size": VECTOR_SIZE,
        "embedding_vocabulary": len(vectors),
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_token_coverage": coverage["test"]["token_coverage"],
        "test_zero_documents": coverage["test"]["zero_documents"],
        "embedding_seconds": embedding_info["embedding_seconds"],
        "classifier_seconds": round(fit_seconds, 3),
        "n_iter": int(classifier.n_iter_.max()),
        "converged": converged,
    }
    classes = classifier.classes_.tolist()
    save_json(
        args.output / f"{name}_test_report.json",
        {
            "class_order": classes,
            "confusion_matrix": confusion_matrix(
                labels[split["test"]], test_pred, labels=classes
            ).tolist(),
            "classification_report": classification_report(
                labels[split["test"]], test_pred, labels=classes, output_dict=True, zero_division=0
            ),
            "embedding_info": embedding_info,
            "coverage": coverage,
        },
    )
    with (args.output / f"{name}_test_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["row_index", "true_label", "predicted_label"])
        writer.writerows(
            zip(split["test"].tolist(), labels[split["test"]].tolist(), test_pred.tolist())
        )
    vectors.save(str(args.model_dir / f"{name}_vectors.kv"), separately=[])
    joblib.dump(classifier, args.model_dir / f"{name}_classifier.joblib", compress=3)
    print(
        f"\n{name}: converged={converged}, iterations={result['n_iter']}\n"
        f"  Validation: Accuracy={validation_metrics['accuracy']:.4f}, "
        f"Macro-F1={validation_metrics['macro_f1']:.4f}\n"
        f"  Test:       Accuracy={test_metrics['accuracy']:.4f}, "
        f"Macro-F1={test_metrics['macro_f1']:.4f}\n"
        f"  Test token coverage={result['test_token_coverage']:.2%}, "
        f"zero-vector documents={result['test_zero_documents']}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path("nyt.csv"))
    parser.add_argument("--ag-data", type=Path, default=default_data_path("ag.csv"))
    parser.add_argument("--split-file", type=Path, default=ROOT / "data" / "split_indices.json",
                        help="已有划分校验后复用，缺失时自动生成")
    parser.add_argument(
        "--glove-path", type=Path, default=ROOT / "resources" / "glove" / "glove.6B.100d.txt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "task2")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "task2")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子，也用于首次生成划分")
    parser.add_argument("--epochs", type=int, default=5, help="Word2Vec 训练轮数")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Word2Vec 线程数；1 有利于复现")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4, help="分类器的数值计算线程数")
    args = parser.parse_args()
    if args.c <= 0 or min(
        args.epochs, args.window, args.min_count, args.workers, args.max_iter, args.threads
    ) <= 0:
        parser.error("训练参数必须为正数。")
    if "glove" in args.methods and not args.glove_path.is_file():
        parser.error("缺少 GloVe 文件。先运行 src/download_glove.py，或提供 --glove-path。")
    if "word2vec_ag" in args.methods and not args.ag_data.is_file():
        parser.error("找不到 AG News 数据文件。")
    texts, labels = load_dataset(args.data)
    split, split_metadata = load_or_create_shared_split(args.split_file, args.data, labels, args.seed)
    print(f"Using shared split: {args.split_file.resolve()} (seed={split_metadata['seed']})", flush=True)
    print(f"Split sizes: { {name: len(indices) for name, indices in split.items()} }", flush=True)
    print("Tokenizing NYT ...", flush=True)
    documents = [tokenize(text) for text in texts]
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "nyt_dataset_sha256": split_metadata["dataset_sha256"],
        "ag_dataset_sha256": file_sha256(args.ag_data) if "word2vec_ag" in args.methods else None,
        "glove_file_sha256": file_sha256(args.glove_path) if "glove" in args.methods else None,
        "split_file": str(args.split_file.resolve()),
        "split_file_sha256": file_sha256(args.split_file),
        "split_seed": split_metadata["seed"],
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "methods": args.methods,
        "seed": args.seed,
        "vector_size": VECTOR_SIZE,
        "token_pattern": TOKEN_PATTERN,
        "word2vec": {
            "sg": 1, "negative": 5, "sample": 1e-3,
            "epochs": args.epochs, "window": args.window,
            "min_count": args.min_count, "workers": args.workers,
            "nyt_corpus": "NYT training split only",
        },
        "classifier": {"name": "LogisticRegression", "solver": "lbfgs", "C": args.c,
                       "max_iter": args.max_iter, "threads": args.threads},
        "python_version": platform.python_version(),
        "gensim_version": gensim.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    results = []
    with threadpool_limits(limits=args.threads):
        for name in args.methods:
            if name == "glove":
                needed_words = {token for document in documents for token in document}
                vectors, embedding_info = load_glove(args.glove_path, needed_words)
                embedding_info["corpus"] = "External pretrained GloVe 6B"
            elif name == "word2vec_ag":
                vectors, embedding_info = train_word2vec(CsvTextCorpus(args.ag_data), name, args)
                embedding_info["corpus"] = "AG News (all provided texts)"
            else:
                corpus = [documents[index] for index in split["train"]]
                vectors, embedding_info = train_word2vec(corpus, name, args)
                embedding_info["corpus"] = "NYT training split only"
            results.append(run_classifier(name, vectors, documents, labels, split, args, embedding_info))
            # 每完成一组就保存汇总；单独运行 --methods 时汇总只包含本次方法。
            with (args.output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
            save_json(args.output / "results.json", {"config": config, "results": results})
            del vectors

    print(f"\nResults saved to: {args.output.resolve()}", flush=True)
    print(f"Models saved to: {args.model_dir.resolve()}", flush=True)
    if not all(result["converged"] for result in results):
        raise SystemExit("存在未收敛分类器，请增大 --max-iter 后重新运行。")


if __name__ == "__main__":
    main()
