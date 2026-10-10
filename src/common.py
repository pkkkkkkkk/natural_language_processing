"""三个文本分类实验共用的数据读取、划分和评价工具。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = r"(?u)\b\w+\b"


def default_data_path(name):
    """读取工程 data 目录中的原始数据。"""
    return ROOT / "data" / name


def load_dataset(path):
    """按 CSV 原始记录顺序读取数据，不静默删除或重排样本。"""
    texts, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("数据文件必须包含 text 和 label 两列。")
        for index, row in enumerate(reader):
            text, label = row.get("text"), row.get("label")
            if not text or not text.strip() or not label or not label.strip():
                raise ValueError(f"第 {index + 1} 条数据的 text 或 label 为空。")
            texts.append(text)
            labels.append(label.strip())
    if len(set(labels)) < 2:
        raise ValueError("分类数据至少需要两个类别。")
    return np.asarray(texts, dtype=object), np.asarray(labels)


def make_split(labels, seed):
    """先留出 20%，再均分为验证集和测试集；两次均按类别分层。"""
    indices = np.arange(len(labels))
    train, held_out = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, random_state=seed, stratify=labels[held_out]
    )
    return {"train": train, "validation": validation, "test": test}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_shared_split(path, data_path, n_samples):
    """校验已有划分，拒绝数据不匹配及重复、遗漏、越界的索引。"""
    saved = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_sha256", "n_samples", "row_index_base", "seed", "train", "validation", "test"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("划分文件缺少必需字段。")
    if saved["dataset_sha256"] != file_sha256(data_path):
        raise ValueError("NYT 数据与共享划分的 SHA-256 不一致，不能复用此划分。")
    if saved["n_samples"] != n_samples or saved["row_index_base"] != 0:
        raise ValueError("划分文件的样本数量或索引起点不正确。")
    if type(saved["seed"]) is not int:
        raise ValueError("划分文件的随机种子必须是整数。")
    split = {}
    for name in ("train", "validation", "test"):
        indices = saved[name]
        if not isinstance(indices, list) or not indices or not all(type(index) is int for index in indices):
            raise ValueError(f"{name} 必须是非空的整数索引列表。")
        if any(index < 0 or index >= n_samples for index in indices):
            raise ValueError("数据划分存在越界索引。")
        split[name] = np.asarray(indices, dtype=np.int64)
    combined = np.concatenate(list(split.values()))
    if not np.array_equal(np.sort(combined), np.arange(n_samples)):
        raise ValueError("数据划分存在重复、遗漏或越界索引。")
    return split, saved


def load_or_create_shared_split(path, data_path, labels, seed=42):
    """已有划分校验后复用；缺失时按统一规则生成并保存。"""
    if path.exists():
        return load_shared_split(path, data_path, len(labels))
    split = make_split(labels, seed)
    metadata = {
        "dataset_sha256": file_sha256(data_path),
        "n_samples": len(labels),
        "row_index_base": 0,
        "seed": seed,
        "stratified": True,
        **{name: indices.tolist() for name, indices in split.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, metadata)
    print(f"Created shared split: {path.resolve()} (seed={seed})", flush=True)
    return split, metadata


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
