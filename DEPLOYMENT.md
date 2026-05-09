# 周四作业 A1-A8 云端部署说明

这个目录已经整理成一个 Streamlit 多页面总应用，可以通过一个公网 URL 展示 A1 到 A8 的交互效果。

## 部署方式

1. 把 `/Users/qiudianer/周四作业` 上传到 GitHub 仓库。
2. 打开 Streamlit Community Cloud，新建 app。
3. Repository 选择刚上传的仓库。
4. Branch 选择 `main`。
5. Main file path 填：

```text
Home.py
```

部署完成后，Streamlit Cloud 会生成一个公开 URL。以后只需要打开这个 URL，不需要在本地运行。

## 为什么不是 GitHub Pages

这 8 个作业都是 Python/Streamlit app。GitHub Pages 只能托管静态网页，不能执行 Python，所以不能直接展示这些 Streamlit 交互。GitHub 负责保存代码，Streamlit Cloud 负责在线运行代码并提供 URL。

## 上传时跳过的内容

`.gitignore` 已经排除了本地虚拟环境、大型数据集和生成结果，例如：

- `.venv/`
- `a5/数据集/`
- `outputs/`
- `reports/`
- `screenshots/`

A5 已经有云端展示模式，缺少完整本地数据集时会自动使用轻量 demo 或合成展示。
