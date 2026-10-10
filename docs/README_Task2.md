# Task 2 词向量文本分类实验

源码为 `src/task2.py`，完成作业要求的三组实验。每组均使用 100 维词向量，对文档中有效词的向量取平均，再训练 Logistic Regression。在与 Task 1 完全相同的 NYT 测试集上报告 Accuracy 和 Macro-F1。

| 方法 | 词向量来源 | 训练分类器的数据 |
| --- | --- | --- |
| `glove` | Stanford 预训练 `glove.6B.100d.txt` | NYT 训练集 |
| `word2vec_ag` | 当前提供的 AG News 全部文本，自行训练 Word2Vec | NYT 训练集 |
| `word2vec_nyt` | NYT 训练集文本，自行训练 Word2Vec | NYT 训练集 |

## 安装和运行

使用 Python 3.12。准备 `data/nyt.csv` 和 `data/ag.csv` 后，可直接在作业目录中运行，无需先运行 Task 1：

```powershell
$taskPython = ".\.venv\Scripts\python.exe"
& $taskPython -m pip install -r configs/requirements.txt
& $taskPython src/download_glove.py
& $taskPython src/task2.py
```

如果没有虚拟环境，先执行 `python -m venv .venv`。本脚本使用同一 `src/` 目录下 `common.py` 的数据读取、分词规则和评价函数。`data/split_indices.json` 缺失时自动按默认种子 42 生成 80% / 10% / 10% 的分层划分；已有时校验后复用。已有文件损坏或与 CSV 不匹配会报错，不会自动覆盖。

## GloVe 文件

下载脚本从 [Stanford 官方地址](https://nlp.stanford.edu/data/glove.6B.zip) 获取约 822 MiB 的 ZIP，仅解压 100 维文件到 `resources/glove/glove.6B.100d.txt`。ZIP 与解压后的文本共占约 1.2 GB。下载位置可用 `src/download_glove.py --directory` 指定。若网络中断，重新执行下载脚本；未完成的 `.part` 文件不会作为正式词向量使用。

已有 GloVe 文件时，无需下载，可以指定路径：

```powershell
& $taskPython src/task2.py --glove-path "D:/embeddings/glove.6B.100d.txt"
```

代码逐行读取词向量文件，只保留 NYT 文档可能使用的词。这个操作仅筛选已有的外部向量以减少存储，没有在 NYT 上训练或调整 GloVe；各文档均可查询预训练词表中已有的词，包括只在验证集、测试集出现的词。

## 文档表示和训练设置

- 分词规则与 Task 1 相同：转为小写，保留连续字母、数字、下划线组成的 token，也保留单字符词。不去停用词、不做词干化。
- 所有方法均使用 100 维词向量。文档向量是所有有效词出现位置对应向量的算术平均。重复词重复参与平均，未知词跳过。没有有效词的文档使用全零向量，并在报告中记录数量。
- Word2Vec 默认采用 Skip-gram：`sg=1`、`window=5`、`min_count=2`、`negative=5`、`sample=1e-3`、`epochs=5`、`seed=42`、`workers=1`。作业仅固定了 100 维，其他参数是本代码的实验设置。
- AG News 以可重复遍历的 CSV 语料迭代器训练，不需要把全部分词结果同时放入内存。
- NYT Word2Vec 仅使用 NYT 训练集文本；验证集和测试集不参加词向量训练。
- Logistic Regression 与 Task 1 使用相同参数：`solver="lbfgs"`、`C=1.0`、`max_iter=2000`，采用默认 L2 正则化。直接输入平均向量，不做额外标准化、归一化或类别加权。
- 验证集和测试集分别输出结果。默认参数固定，没有根据测试集指标调参。

## 输出文件

默认报告目录为 `results/task2/`，权重目录为 `models/task2/`（可用 `--model-dir` 修改）：

| 文件 | 内容 |
| --- | --- |
| `results.csv` | 三种方法的验证集和测试集指标、覆盖率、词表大小、训练时间及收敛状态 |
| `results.json` | 完整结果、训练参数、依赖版本和输入文件 SHA-256 |
| `<方法>_test_report.json` | 每类 Precision、Recall、F1、Support、混淆矩阵及各子集词向量覆盖情况 |
| `<方法>_test_predictions.csv` | 测试集原始数据索引、真实类别和预测类别 |
| `models/task2/<方法>_vectors.kv` | 本次实际使用的 100 维词向量 |
| `models/task2/<方法>_classifier.joblib` | 训练后的 Logistic Regression 分类器 |

词覆盖率按 token 的出现次数计算；全零文档数反映完全没有有效词向量的文档数量。混淆矩阵按 `class_order` 排列，行是真实类别、列是预测类别。

GloVe 原始资源保存在 `resources/`，词向量和分类器权重保存在 `models/task2/`，均由 `.gitignore` 排除。
## 加载模型预测新文本

在作业目录中运行：

```python
import joblib
import sys
from gensim.models import KeyedVectors

sys.path.insert(0, "src")
from task1 import ROOT
from task2 import mean_document_vectors, tokenize

name = "word2vec_nyt"  # 也可以是 glove 或 word2vec_ag
model_dir = ROOT / "models" / "task2"
vectors = KeyedVectors.load(str(model_dir / f"{name}_vectors.kv"))
classifier = joblib.load(model_dir / f"{name}_classifier.joblib")
documents = [tokenize("The company announced higher quarterly profits.")]
features, coverage = mean_document_vectors(documents, vectors)
print(classifier.predict(features)[0])
```

## 当前数据的实测结果

使用提供的数据和上述默认参数，三组完整实验均已运行，分类器均已收敛。

| 方法 | 测试集 Accuracy | 测试集 Macro-F1 | 测试集词覆盖率 |
| --- | ---: | ---: | ---: |
| GloVe 6B 100d | 0.9852 | 0.9646 | 99.79% |
| Word2Vec trained on AG News | 0.9757 | 0.9410 | 97.33% |
| Word2Vec trained on NYT training set | 0.9818 | 0.9542 | 99.29% |
