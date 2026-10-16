# 第三方代码

第三方项目保留独立 Git 仓库；主项目通过 `.gitignore` 排除其源码目录，仅记录来源和检出的版本。

## SAM 3 / SAM 3.1

- 本地目录：`thirdparty/sam3/`
- 官方仓库：[facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SSH 地址：`git@github.com:facebookresearch/sam3.git`
- 克隆日期：2026-09-18
- 分支：`main`
- 克隆时提交：`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`
- 模型发布说明：[SAM 3.1](sam3/RELEASE_SAM3p1.md)
- 上游许可证：[LICENSE](sam3/LICENSE)

初次使用 SSH 浅克隆获取当时最新版本的全部仓库文件。Conda 环境配置见下文，模型权重单独获取。

从项目根目录重新克隆最新 `main`：

```bash
git clone --depth 1 git@github.com:facebookresearch/sam3.git thirdparty/sam3
```

如需恢复到本次记录的提交，在克隆后执行：

```bash
git -C thirdparty/sam3 fetch --depth 1 origin 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
git -C thirdparty/sam3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
```

当前检出的上游源码在 SAM 3.1 `start_session()` 中会向模型传入不支持的
`offload_state_to_cpu` 参数。已按上游 [Issue #544](https://github.com/facebookresearch/sam3/issues/544)
和 [PR #543](https://github.com/facebookresearch/sam3/pull/543) 修复本地源码；重新克隆上述版本后，
可从项目根目录重新应用 [补丁](sam3-start-session.patch)：

```bash
git -C thirdparty/sam3 apply ../sam3-start-session.patch
```

另一个本地修复涉及纯点提示：首次创建对象时，`_build_sam2_output()` 因当前帧
尚无缓存而提前返回空结果，丢弃了模型已经生成的掩码。修复后从空缓存开始合并
新掩码。上游 [Issue #555](https://github.com/facebookresearch/sam3/issues/555)
报告了类似空输出现象；这里的修改来自本地定位，可通过以下补丁恢复：

```bash
git -C thirdparty/sam3 apply ../sam3-point-prompt.patch
```

该修复在独立 SAM3 仓库中的本地提交为 `6e45d92`；主项目保存上述补丁。

安装及权重获取方式见上游 [README](sam3/README.md)。

## SAM 3 Conda 环境

环境名为 `sam3`，使用 Python 3.12、PyTorch 2.10.0 + CUDA 12.8、torchvision 0.25.0。
环境配置与算法/Isaac Sim 环境独立，包含图像推理、点选交互和 Jupyter 图像示例所需依赖。

从 QuadWeld 根目录安装：

```bash
conda env create -f environments/sam3.yml
conda activate sam3
python -m pip install -r environments/sam3-requirements.txt
```

本机环境路径为 `/data/huangliqi/miniconda3/envs/sam3`。若当前 shell 尚未初始化 Conda，可先执行：

```bash
source /data/huangliqi/miniconda3/etc/profile.d/conda.sh
conda activate sam3
```

[依赖文件](../environments/sam3-requirements.txt) 固定以下兼容性组合：

- PyTorch 与 torchvision 使用匹配的 CUDA 12.8 wheel；对应版本见 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v2100)。
- SAM 3 要求 NumPy < 2，使用 NumPy 1.26.4 与 OpenCV 4.11.0.86。
- 当前上游代码仍导入 `pkg_resources`，使用保留该模块的 setuptools 80.9.0。
- 图像示例依赖逐项安装。上游 `notebooks` 额外依赖中的视频库 decord 0.6.0，其 wheel 内部标记为 CPython 3.6，与本环境的 Python 3.12 不匹配，因此不安装该视频库。
- SAM 3 以 editable 模式引用 `thirdparty/sam3/`，因此使用环境时应保留该源码目录。

检查依赖、CUDA 运算和 SAM 3 导入：

```bash
python -m pip check
CUDA_VISIBLE_DEVICES=0 python scripts/check_sam3_env.py
```

进一步验证模型构建、图像编码、文字提示和点提示的前向计算：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_sam3_env.py --forward-smoke \
  --report outputs/diagnostics/sam3/sam3_env_check.json
```

`--forward-smoke` 使用随机初始化权重，仅检查运行环境，不下载预训练权重，也不评估分割效果。
GPU 编号可按 `nvidia-smi` 显示的空闲情况修改。

2026-09-18 在本机 RTX 3090 上完成验证：`pip check` 无依赖错误，CUDA 矩阵运算、
torchvision CUDA NMS，以及 SAM 3 图像编码、点提示和文字提示前向计算均通过。
点提示输出形状为 `3 × 480 × 640`，本次检查峰值 GPU 内存约 5.22 GiB；
记录见 [环境验证报告](../outputs/diagnostics/sam3/sam3_env_check.json)。本次未加载预训练权重。

## 模型权重

本机现有的是 **SAM 3.1 视频模型**权重
`thirdparty/sam3/assets/model/sam3.1_multiplex.pt`（约 3.5 GB），
当时从 ModelScope 的 `facebook/sam3.1` 下载到本地。终端记录中的命令如下，
从 QuadWeld 根目录执行：

```bash
python -m pip install modelscope
modelscope download --model facebook/sam3.1 --local_dir ./thirdparty/sam3/assets/model
```

本地推理需在 `build_sam3_predictor(version="sam3.1", ...)` 中传入
`checkpoint_path="thirdparty/sam3/assets/model/sam3.1_multiplex.pt"`。
模型源码目录由主项目的 `.gitignore` 排除，因此该权重不会随主项目 Git 仓库保存。

原先 README 写的是上游默认方式：不指定 `checkpoint_path` 时，SAM 3.1 从
[Hugging Face `facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) 下载
`sam3.1_multiplex.pt`；SAM 3 从
[Hugging Face `facebook/sam3`](https://huggingface.co/facebook/sam3) 下载 `sam3.pt`。
使用 Hugging Face 时可能需要在模型页获得访问权限并执行 `hf auth login`。
项目内目前未发现 `sam3.pt`；终端历史中的 `hf download facebook/sam3 ...` 命令
不能证明该下载成功。

交互式示例可通过 `jupyter lab thirdparty/sam3/examples/` 启动。
