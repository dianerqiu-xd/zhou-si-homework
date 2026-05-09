# Vibe Coding A4

## 运行方式

```bash
/Users/qiudianer/Desktop/pycharm/.venv311/bin/python -m streamlit run app.py
```

运行后浏览器访问 Streamlit 给出的本地地址，通常是 `http://localhost:8501`。

## 文件说明

- `app.py`：核心源码，包含交互式 Web app 与四个机器学习演示模块。
- `make_report.py`：生成截图和 PDF 实验报告的脚本。
- `screenshots/`：报告中使用的截图。
- `outputs/Vibe_Coding_A4_Report.pdf`：PDF 实验报告。
- `assets/default_images/`：KNN 页面默认使用的图片，来自本地 `图片` 文件夹的压缩副本。
- `data/`：可选数据目录。如果有真实 CIFAR-10 Python batch 数据，可放在 `data/cifar-10-batches-py/`；没有时程序会使用离线 CIFAR-10 风格图像数据完成演示。

四个页面都接入了默认图片/上传图片：

- Least Squares：对图片列平均亮度做最小二乘直线拟合。
- KNN：把图片作为 query，展示最近邻与不同 K 的投票结果。
- Linear Classifier Templates：把图片送入 softmax 线性分类器，显示 top-5 类别分数。
- Optimizer and Loss：可使用图片生成的分类器 scores 演示 SVM hinge loss 与 softmax cross-entropy。

## 使用的 Agent / LLM

- Codex（GPT-5 系列）
