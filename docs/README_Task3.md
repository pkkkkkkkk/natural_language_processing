# Task 3 BERT 文本分类实验

源码为 `src/task3.py`，使用 `google-bert/bert-base-uncased` 微调 NYT 文本分类模型，固定最大序列长度为 64、训练轮数为 3。BERT 编码器与分类头一起训练。代码使用 `data/split_indices.json` 中的共享划分，在第 3 轮完成后使用最终模型评估 NYT 测试集的 Accuracy 和 Macro-F1。

## 准备数据

脚本默认读取工程内的 `data/nyt.csv`，并使用 `data/split_indices.json` 中的共享划分。保留同一 `src/` 目录下的 `common.py`：划分文件缺失时自动生成，已有时校验后复用。

`--seed` 用于训练和首次生成划分，默认值为 42。已有划分不会因修改训练种子而被覆盖；若需要新划分，为三个任务指定相同的新 `--split-file` 路径。

## 命令行运行

使用 Python 3.11 或更高版本，以下命令均在工程根目录执行。完整 GPU 训练需要预先配置 NVIDIA 驱动与匹配的 CUDA 版 PyTorch。安装 Task 3 专用依赖：

```powershell
$taskPython = ".\.venv\Scripts\python.exe"
& $taskPython -m pip install -r configs/requirements_task3.txt
```

NVIDIA GPU 环境需要预先安装支持该 GPU 的 CUDA 版 PyTorch。已有正确 CUDA 版时，依赖文件中的版本范围允许保留它。

完整训练：

```powershell
& $taskPython src/task3.py --device cuda --batch-size 16
```

`--device auto` 默认优先选择 CUDA，否则使用 CPU。

## 数据和训练设置

- 复用 `data/split_indices.json` 中的原始划分，完整实验的训练、验证、测试集分别为 9,215、1,152、1,152 条。
- 校验 NYT 的 SHA-256、样本数、索引起点、索引是否完整且互不重叠。已有文件校验失败时会报错；只有文件缺失时才按统一规则生成新的分层划分。
- 类别 ID 映射从训练集类别建立，并保存到模型的 `label2id` / `id2label`。
- 使用 BERT 自带的 uncased WordPiece tokenizer；不使用前两部分的词袋分词规则。`max_length=64` 包含特殊 token，长文本截断，短文本 padding 到 64，并通过 attention mask 忽略 padding。
- 默认 batch size 为 16，评估 batch size 为 32，学习率为 `2e-5`，优化器为 AdamW，weight decay 为 `0.01`。bias 和 LayerNorm 权重不施加 weight decay。
- 学习率先在总更新步数的 10% 内 warmup，再线性下降；梯度范数裁剪为 1.0。
- CUDA 上默认使用 FP16 混合精度及 GradScaler；CPU 使用 FP32。`--fp32` 可以禁用 CUDA 混合精度。
- 随机种子默认 42。每轮打乱训练顺序，验证集与测试集保持索引顺序，不参与参数更新。
- 每轮进行验证；固定完成 3 轮后，使用第 3 轮最终模型进行一次测试。不根据测试指标选择模型或调参。

除模型、序列长度和轮数外，其他参数是本代码选择的实验设置，可以根据验证集表现调整。调整后应记录参数，避免依赖测试集挑选配置。

## 输出文件

完整实验的报告默认保存到 `results/task3/`，权重保存到 `models/task3/final_model/`，预训练缓存保存到 `resources/huggingface/`。`--output`、`--model-dir` 和 `--cache-dir` 可分别覆盖这三种路径：

| 文件 | 内容 |
| --- | --- |
| `results.csv` | 最终验证集和测试集的 Accuracy、Macro-F1 |
| `results.json` | 最终结果、模型版本、环境、数据划分及训练参数 |
| `run_config.json` | 训练开始时记录的运行配置 |
| `training_history.csv` / `.json` | 3 轮训练损失与验证指标 |
| `bert_test_report.json` | 每类 Precision、Recall、F1、Support 和混淆矩阵 |
| `bert_test_predictions.csv` | 测试集原始行索引、真实类别、预测类别 |
| `models/task3/final_model/` | 微调完成的模型权重、配置和 tokenizer |

`row_index` 是原 CSV 中从 0 开始的数据记录序号，不含表头。混淆矩阵的类别顺序由 `class_order` 指定，行是真实类别、列是预测类别。正式实验的结果记录为 `run_type="full_experiment"`。

预训练模型缓存和微调模型在工程内按类别保存，并由 `.gitignore` 排除。

## 加载模型预测

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "models/task3/final_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval()
inputs = tokenizer(
    ["The company announced higher quarterly profits."],
    truncation=True, padding="max_length", max_length=64, return_tensors="pt"
)
with torch.inference_mode():
    prediction = model(**inputs).logits.argmax(dim=-1).item()
print(model.config.id2label[prediction])
```

