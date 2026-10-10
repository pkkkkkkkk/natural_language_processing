# Task 1 文本分类实验

源码为 `src/task1.py`，完成 Binary Bag of Words + Logistic Regression 和 Word Frequency + Logistic Regression 两组实验，最终报告同一份 NYT 测试集上的 Accuracy 和 Macro-F1。

## 安装和运行

使用 Python 3.12。打开终端并进入工程根目录 `作业一`，执行：

```powershell
python -m venv .venv
$taskPython = ".\.venv\Scripts\python.exe"
& $taskPython -m pip install -r configs/requirements.txt
& $taskPython src/task1.py
```

默认读取 `data/nyt.csv`；其他数据位置可用 `--data` 指定。三个任务使用 `src/common.py` 中相同的数据划分逻辑。`data/split_indices.json` 缺失时自动生成，已有时校验后复用。`--seed` 用于训练和首次生成划分；若需新划分，使用新的 `--split-file` 路径。

## 实现方法

1. 按 CSV 原始顺序读取 `text` 和 `label`。遇到空文本或空标签时明确报错，不静默删除样本。
2. 缺少划分文件时，默认随机种子为 42，使用分层随机抽样将 NYT 分成训练集 80%、验证集 10%、测试集 10%；已有时复用保存的划分。正式实验的样本数量因取整为 9,215、1,152、1,152。两组实验共享这些索引。
3. 使用 `CountVectorizer`：统一转为小写，以连续的字母、数字、下划线作为 token，保留单字符 token；不移除停用词，不做词干化，不限制词表大小。
4. 词表仅从训练集建立，验证集和测试集中的未知词会被忽略。特征矩阵采用稀疏存储。
5. Binary Bag of Words 设置 `binary=True`，每个特征是 0 或 1；Word Frequency 设置 `binary=False`，每个特征是词在文档中出现的原始次数。例如 `apple apple banana` 对应 `[1, 1]` 和 `[2, 1]`。
6. 两组实验均使用 Logistic Regression，`solver="lbfgs"`、`C=1.0`、`max_iter=2000`，使用 scikit-learn 默认的 L2 正则化。不设置类别权重，不归一化词频，不使用 TF-IDF。
7. 仅用训练集拟合模型，分别计算验证集与测试集的 Accuracy 和 Macro-F1。默认运行采用固定参数，没有根据测试集结果调参。若需要调参，应只比较验证集表现，确定参数后再做最终测试。


## 输出文件

默认报告输出到 `results/task1/`，共享划分写入 `data/`，模型写入 `models/task1/`：

| 文件 | 内容 |
| --- | --- |
| `results.csv` | 两组方法的验证集、测试集指标，以及词表大小、训练时间和收敛状态 |
| `results.json` | 实验参数、环境版本、数据指纹、各子集类别数量和结果 |
| `data/split_indices.json` | 三个任务共用的数据划分索引，任意任务都可在缺失时生成 |
| `binary_bow_test_report.json`、`word_frequency_test_report.json` | 每类 Precision、Recall、F1、Support 和混淆矩阵 |
| `binary_bow_test_predictions.csv`、`word_frequency_test_predictions.csv` | 每条测试数据的原始索引、真实类别与预测类别 |
| `models/task1/*_model.joblib` | 已训练的完整 Pipeline，包含词表与分类器 |

## 加载模型预测新文本

在作业目录下运行以下 Python 代码：

```python
import joblib

model = joblib.load("models/task1/binary_bow_model.joblib")
prediction = model.predict(["The company announced higher quarterly profits."])
print(prediction[0])
```
## 当前数据的实测结果

使用当前提供的 NYT 数据及上述默认参数，完整运行得到以下结果。两种方法的词表均为 61,021 个 token，模型均已收敛。

| 方法 | 验证集 Accuracy | 验证集 Macro-F1 | 测试集 Accuracy | 测试集 Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Binary Bag of Words | 0.9809 | 0.9551 | 0.9887 | 0.9718 |
| Word Frequency | 0.9835 | 0.9626 | 0.9887 | 0.9729 |
