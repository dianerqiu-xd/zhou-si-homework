# A5 图像识别与神经网络 Vibe Coding

本目录是 `A5` 的完整 Streamlit 提交项目。

## 文件说明

- `streamlit_app.py`：Web 交互应用入口。
- `src/core.py`：核心算法与演示逻辑源码。
- `assets/default_image.jpg`：默认演示图片，可在应用中上传替换。
- `数据集/`：A5 默认数据集目录。
- `demo_datasets/`：可上传 GitHub 的小型演示数据集目录。
- `prepare_demo_datasets.py`：从本地完整数据集中抽取小型演示数据集。
- `prepare_hog_bow_dataset.py`：下载 CIFAR-10 并生成任务 1 的轻量数据集。
- `prepare_cnn_resnet_datasets.py`：下载 MNIST/CIFAR-10 并生成任务 3、4 的数据集。
- `outputs/screenshots/`：报告截图。
- `reports/final_report.html` 与 `reports/final_report.pdf`：实验报告。
- `reports/作业小结.md`：作业小结。
- `prompt.txt`：本次 Vibe Coding prompt 文本。

## 运行方式

```bash
/Users/qiudianer/周四作业/.venv/bin/python -m streamlit run streamlit_app.py
```

## 默认数据集路径

```text
/Users/qiudianer/周四作业/a5/数据集/hog_bow_dataset
/Users/qiudianer/周四作业/a5/数据集/cnn_lenet_mnist
/Users/qiudianer/周四作业/a5/数据集/resnet_cifar10_dataset
```

代码默认从 `a5/数据集/` 读取数据。替换自己的数据集时，保持 `train/类别名/*.图片` 与 `test/类别名/*.图片` 的目录结构即可。

## GitHub 上传说明

`数据集/` 和 `raw/` 目录体积较大，已加入 `.gitignore`，不建议上传到 GitHub。上传代码后，在本地重新生成数据集：

```bash
cd /Users/qiudianer/周四作业/a5
/Users/qiudianer/周四作业/.venv/bin/python prepare_hog_bow_dataset.py
/Users/qiudianer/周四作业/.venv/bin/python prepare_cnn_resnet_datasets.py
```

生成后会得到：

```text
a5/数据集/hog_bow_dataset
a5/数据集/cnn_lenet_mnist
a5/数据集/resnet_cifar10_dataset
```

部署到 Streamlit Cloud 生成公网 URL 时，可以不生成这些大数据集。应用检测不到 `数据集/` 时会自动进入云端展示模式：

- 任务 1 使用轻量形状分类演示 HOG + BOW + SVM 流程。
- 任务 3 展示 LeNet/CNN 风格训练曲线和轻量分类结果。
- 任务 4 展示不同深度 ResNet 的预置性能对比表。

本地运行且存在完整 `数据集/` 时，应用会优先读取真实数据集。

如果希望 Streamlit Cloud 也读取真实图片文件夹，可以上传小型 demo 数据集。先在本地完整数据集已存在的情况下运行：

```bash
cd /Users/qiudianer/周四作业/a5
/Users/qiudianer/周四作业/.venv/bin/python prepare_demo_datasets.py
```

这会生成：

```text
a5/demo_datasets/hog_bow_dataset
a5/demo_datasets/cnn_lenet_mnist
a5/demo_datasets/resnet_cifar10_dataset
```

`demo_datasets/` 体积较小，可以随代码上传 GitHub。应用读取顺序是：

```text
完整数据集 a5/数据集/
小型数据集 a5/demo_datasets/
合成/预置展示
```

## Streamlit Cloud

如果把本文件夹作为仓库根目录，Main file path 填 `streamlit_app.py`。
如果上传整个周四作业目录，Main file path 填 `a5/streamlit_app.py`。

## 使用的 Agent / LLM

- Codex GPT-5
