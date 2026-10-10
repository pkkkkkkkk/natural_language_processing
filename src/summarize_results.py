"""汇总三个实验已有的结果，不重新训练模型。"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "task", "method", "validation_accuracy", "validation_macro_f1",
    "test_accuracy", "test_macro_f1",
)


def main():
    rows = []
    for task in ("task1", "task2", "task3"):
        path = ROOT / "results" / task / "results.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            for result in csv.DictReader(file):
                rows.append({"task": task, **{key: result[key] for key in FIELDS[1:]}})
    target = ROOT / "results" / "summary.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} methods to: {target}")


if __name__ == "__main__":
    main()
