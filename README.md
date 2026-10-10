# 作业一：文本分类实验

工程包含三个任务、六种文本表示方法的完整源码与正式实验结果。所有方法使用相同的 NYT 训练、验证、测试划分，最终比较测试集 Accuracy 和 Macro-F1。Task 1 按已确认的更正实现两种方法：Binary Bag of Words 和 Word Frequency。

## 工程目录

```text
作业一/
├── README.md                       # 工程入口、运行和提交说明
├── .gitignore                      # 排除数据、权重和临时文件
├── src/                            # 三个实验及辅助脚本
│   ├── common.py                   # 共用的数据读取、划分和评价工具
│   ├── task1.py
│   ├── task2.py
│   ├── task3.py
│   ├── download_glove.py
│   └── summarize_results.py
├── configs/                        # 依赖清单
│   ├── requirements.txt            # Task 1、Task 2
│   └── requirements_task3.txt       # Task 3
├── data/                           # 运行数据
│   ├── split_indices.json          # 任意实验首次运行时生成，三个任务共用
│   ├── nyt.csv                     # 原始 NYT 数据
│   └── ag.csv                      # 原始 AG News 数据
├── results/                        # 可用于报告的正式实验结果
│   ├── summary.csv                 # 由各任务 results.csv 生成的六方法汇总
│   ├── task1/
│   ├── task2/
│   └── task3/
├── docs/                           # 各任务的实现和运行说明
│   ├── README_Task1.md
│   ├── README_Task2.md
│   ├── README_Task3.md
├── models/                         # 按任务分类的训练权重
├── resources/                      # GloVe 和 BERT 下载缓存
├── .venv/                          # 本机 Python 环境
```

`.gitignore` 已排除权重、缓存以及环境，GitHub 提交内容以源码、运行说明、正式结果和 PDF 报告为主。

## 运行

以下命令均在工程根目录执行。Task 1、Task 2 使用 Python 3.12。

```powershell
$taskPython = ".\.venv\Scripts\python.exe"
& $taskPython src/task1.py
& $taskPython src/download_glove.py
& $taskPython src/task2.py
& $taskPython src/summarize_results.py
```

新电脑可以先创建环境并安装依赖：

```powershell
python -m venv .venv
$taskPython = ".\.venv\Scripts\python.exe"
& $taskPython -m pip install -r configs/requirements.txt
```

Task 3 在配置好 NVIDIA 驱动及 CUDA 版 PyTorch 的本地或云端 GPU 环境中运行。在工程根目录执行：

```bash
python -m pip install -r configs/requirements_task3.txt
python src/task3.py --device cuda --batch-size 16
```

源码按自身位置定位工程根目录，默认读取工程内的 `data/`、`resources/`，将结果写入 `results/`，权重写入 `models/`。输入文件和输出目录的显式相对参数按终端当前目录解释；其他位置可通过 `--data`、`--model-dir` 等命令行参数指定。

## 查看结果与实现

- [六种方法的汇总表](results/summary.csv)：由 `src/summarize_results.py` 读取三个任务的结果 CSV 生成，重新运行实验后执行汇总脚本即可更新。
- [Task 1 说明](docs/README_Task1.md)：两种词袋表示与 Logistic Regression。
- [Task 2 说明](docs/README_Task2.md)：GloVe、AG Word2Vec、NYT Word2Vec 与 Logistic Regression。
- [Task 3 说明](docs/README_Task3.md)：BERT-base-uncased、最大长度 64、固定训练 3 轮。

各任务目录保留结果 CSV / JSON、分类报告与混淆矩阵、逐条测试预测。Task 3 另保留三轮训练历史和运行配置。结果中的训练参数、环境版本和指标反映实际运行记录；当前文件路径以源码和文档为准。
