# SAM3.1 部署

双目流程使用独立的 Python 3.12 `sam3` 环境，仿真采集使用已有的 Isaac Sim 环境。
第三方源码和模型权重由 `.gitignore` 排除，需要在本地准备。
以下提交、权重版本和兼容性组合沿用项目原部署记录。

## SAM3.1

源码：[facebookresearch/sam3](https://github.com/facebookresearch/sam3)。
记录的上游提交：`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`。

从 QuadWeld 根目录执行：

```bash
git clone https://github.com/facebookresearch/sam3.git thirdparty/sam3
git -C thirdparty/sam3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
git -C thirdparty/sam3 apply ../sam3-start-session.patch
conda env create -f environments/sam3.yml
conda activate sam3
python -m pip install -r environments/sam3-requirements.txt
python -m pip check
```

[start-session 补丁](sam3-start-session.patch) 修复 SAM3.1 会话启动传入模型不支持参数的问题。
双目流程使用 `metal panel`、`metal plate` 和 `steel plate` 文本提示生成左目板件候选。

依赖文件固定 PyTorch `2.10.0+cu128`、torchvision `0.25.0+cu128`、
NumPy `1.26.4`、OpenCV `4.11.0.86` 和 setuptools `80.9.0`。
保留 setuptools 是因为当前上游仍导入 `pkg_resources`。

SAM 权重应放在：

```text
thirdparty/sam3/assets/model/sam3.1_multiplex.pt
```

原部署使用 ModelScope 的 `facebook/sam3.1` 模型：

```bash
uvx --from modelscope modelscope download --model facebook/sam3.1 \
  sam3.1_multiplex.pt --local_dir ./thirdparty/sam3/assets/model
```

也可按上游 [SAM3.1 模型页](https://huggingface.co/facebook/sam3.1) 获取相同文件。
访问权限和许可证以源码及模型页面为准。

## 运行双目流程

在项目根目录执行：

```bash
.venv/bin/python main.py --station A25 --headless \
  --sam-python /path/to/envs/sam3/bin/python
```

仅分割一组已采集双目的左图：

```bash
.venv-sam/bin/python -m model_runtime.sam_adapter <case>/input.json --output-dir <sam_dir>
```

源码通过 editable 安装，运行时应保留 `thirdparty/sam3/`。
输入、输出和回放约定见 [项目 README](../README.md)。
