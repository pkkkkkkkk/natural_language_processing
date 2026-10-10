"""Task 3: fine-tune BERT-base-uncased for 3 epochs with max_length=64."""

import argparse
import csv
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


from common import (
    ROOT, default_data_path, file_sha256, load_dataset,
    load_or_create_shared_split, save_json,
)


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
SMOKE_SIZES = {"train": 48, "validation": 16, "test": 16}


def load_data_and_split(data_path, split_path, seed=42):
    """读取 NYT；缺失时创建共享划分，已有划分校验后复用。"""
    texts, labels = load_dataset(data_path)
    split, metadata = load_or_create_shared_split(split_path, data_path, labels, seed)
    return texts, labels, split, metadata


def select_smoke_subset(split, labels, seed):
    """仅用于运行检查：从原有各子集内部进行分层抽样。"""
    selected = {}
    for name, indices in split.items():
        size = SMOKE_SIZES[name]
        if len(indices) <= size:
            selected[name] = indices.copy()
        else:
            selected[name], _ = train_test_split(
                indices, train_size=size, random_state=seed, stratify=labels[indices]
            )
    return selected


class TextDataset(Dataset):
    def __init__(self, texts, label_ids, indices, tokenizer):
        self.encodings = tokenizer(
            texts[indices].tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_ids[indices], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("没有可用的 CUDA GPU。请检查 NVIDIA 驱动及 CUDA 版 PyTorch，或使用 --device cpu。")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def evaluate(model, loader, device, use_amp, num_labels):
    model.eval()
    total_loss = total_samples = 0
    true_ids, predicted_ids = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: values.to(device) for name, values in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
            size = batch["labels"].shape[0]
            total_loss += float(output.loss.item()) * size
            total_samples += size
            true_ids.extend(batch["labels"].cpu().tolist())
            predicted_ids.extend(output.logits.argmax(dim=-1).cpu().tolist())
    return {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(
            true_ids, predicted_ids, labels=list(range(num_labels)), average="macro", zero_division=0
        )),
    }, true_ids, predicted_ids


def train_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, epoch, log_steps):
    model.train()
    total_loss = total_samples = 0
    for step, batch in enumerate(loader, 1):
        batch = {name: values.to(device) for name, values in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(**batch)
        if not torch.isfinite(output.loss).item():
            raise RuntimeError("训练损失不是有限值。请尝试 --fp32 或检查数据。")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # FP16 溢出时 GradScaler 会跳过参数更新，此时也不推进学习率。
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        size = batch["labels"].shape[0]
        total_loss += float(output.loss.item()) * size
        total_samples += size
        if step % log_steps == 0 or step == len(loader):
            print(
                f"  Epoch {epoch}/{NUM_EPOCHS}, step {step}/{len(loader)}, "
                f"mean loss={total_loss / total_samples:.4f}",
                flush=True,
            )
    return total_loss / total_samples


def write_table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path("nyt.csv"))
    parser.add_argument("--split-file", type=Path, default=ROOT / "data" / "split_indices.json",
                        help="已有划分校验后复用，缺失时自动生成")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "resources" / "huggingface")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子，也用于首次生成划分")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--fp32", action="store_true", help="禁用 CUDA FP16 混合精度")
    parser.add_argument("--local-files-only", action="store_true", help="只使用已缓存的预训练模型")
    parser.add_argument("--smoke-test", action="store_true", help="仅用小样本检查运行流程")
    args = parser.parse_args()
    if min(args.batch_size, args.eval_batch_size, args.threads, args.log_steps) <= 0:
        parser.error("batch size、threads 和 log steps 必须为正数。")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("学习率必须为正、weight decay 必须非负、warmup ratio 必须在 [0, 1) 内。")
    if args.output is None:
        args.output = (
            ROOT / "validation" / "task3_smoke_results"
            if args.smoke_test else ROOT / "results" / "task3"
        )
    if args.model_dir is None:
        args.model_dir = (
            ROOT / "validation" / "task3_smoke_results" / "final_model"
            if args.smoke_test else ROOT / "models" / "task3" / "final_model"
        )

    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.fp32
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    texts, labels, split, split_metadata = load_data_and_split(args.data, args.split_file, args.seed)
    if args.smoke_test:
        split = select_smoke_subset(split, labels, args.seed)
        print("SMOKE TEST: small subsets for execution checks only.", flush=True)
    classes = sorted(set(labels[split["train"]].tolist()))
    if len(classes) < 2 or not set(labels.tolist()).issubset(classes):
        raise ValueError("训练集需要包含完整类别集合，且至少有两个类别。")
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    label_ids = np.asarray([label2id[label] for label in labels], dtype=np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; mixed precision: {use_amp}", flush=True)
    print(f"Using shared split: {args.split_file.resolve()} (seed={split_metadata['seed']})", flush=True)
    print(f"Split sizes: { {name: len(indices) for name, indices in split.items()} }", flush=True)
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(args.cache_dir), use_fast=True, local_files_only=args.local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=str(args.cache_dir),
        num_labels=len(classes),
        label2id=label2id,
        id2label=id2label,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(device)
    datasets = {
        name: TextDataset(texts, label_ids, indices, tokenizer) for name, indices in split.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size if name == "train" else args.eval_batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    # 不对 bias 和 LayerNorm 的权重施加 weight decay；所有参数均参加微调。
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for name, p in model.named_parameters() if not any(s in name for s in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for name, p in model.named_parameters() if any(s in name for s in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    total_steps = len(loaders["train"]) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.floor(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "run_type": "smoke_test" if args.smoke_test else "full_experiment",
        "model_name": MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "max_length": MAX_LENGTH,
        "epochs": NUM_EPOCHS,
        "label2id": label2id,
        "seed": args.seed,
        "split_seed": split_metadata["seed"],
        "split_file": str(args.split_file.resolve()),
        "split_file_sha256": file_sha256(args.split_file),
        "dataset_sha256": split_metadata["dataset_sha256"],
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "class_counts": {
            name: dict(sorted(Counter(labels[indices].tolist()).items())) for name, indices in split.items()
        },
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "total_optimizer_steps": total_steps,
        "max_grad_norm": 1.0,
        "encoder_frozen": False,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "precision": "FP16 mixed precision" if use_amp else "FP32",
        "threads": args.threads,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    save_json(args.output / "run_config.json", config)
    if args.smoke_test:
        save_json(args.output / "smoke_indices.json", {name: indices.tolist() for name, indices in split.items()})
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        started = time.perf_counter()
        print(f"\nTraining epoch {epoch}/{NUM_EPOCHS} ...", flush=True)
        train_loss = train_epoch(
            model, loaders["train"], optimizer, scheduler, scaler, device, use_amp, epoch, args.log_steps
        )
        validation, _, _ = evaluate(model, loaders["validation"], device, use_amp, len(classes))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "epoch_seconds": round(time.perf_counter() - started, 3),
        })
        save_json(args.output / "training_history.json", history)
        write_table(args.output / "training_history.csv", history)
        print(
            f"Epoch {epoch}: train loss={train_loss:.4f}, "
            f"validation Accuracy={validation['accuracy']:.4f}, Macro-F1={validation['macro_f1']:.4f}",
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    # 使用训练满 3 轮后的最终模型，测试集仅在此处评估一次。
    test, true_ids, predicted_ids = evaluate(model, loaders["test"], device, use_amp, len(classes))
    true_labels = [id2label[index] for index in true_ids]
    predicted_labels = [id2label[index] for index in predicted_ids]
    result = {
        "method": "bert_smoke" if args.smoke_test else "bert_base_uncased",
        "run_type": config["run_type"],
        "epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "training_seconds": round(training_seconds, 3),
    }
    save_json(args.output / "bert_test_report.json", {
        "run_type": config["run_type"],
        "class_order": classes,
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=classes).tolist(),
        "classification_report": classification_report(
            true_labels, predicted_labels, labels=classes, output_dict=True, zero_division=0
        ),
    })
    with (args.output / "bert_test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_index", "true_label", "predicted_label"])
        writer.writerows(zip(split["test"].tolist(), true_labels, predicted_labels))
    final_model = args.model_dir
    model.save_pretrained(final_model, safe_serialization=True)
    tokenizer.save_pretrained(final_model)
    write_table(args.output / "results.csv", [result])
    save_json(args.output / "results.json", {"config": config, "results": [result]})
    print(
        f"\n{config['run_type']}: Test Accuracy={test['accuracy']:.4f}, Macro-F1={test['macro_f1']:.4f}\n"
        f"Results saved to: {args.output.resolve()}",
        flush=True,
    )
    print(f"Model saved to: {final_model.resolve()}", flush=True)


if __name__ == "__main__":
    main()
