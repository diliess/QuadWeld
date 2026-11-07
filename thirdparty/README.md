# DA3 与 SAM3 部署

完整识别流程使用独立的 `depth-anything-3` 和 `sam3` Conda 环境，均为 Python 3.12。
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
完整流程只使用固定文字提示 `metal plate`、`weld`、`weld bead`，不需要点提示补丁或 Notebook 环境。

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

## Depth Anything 3

源码：[ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)。
记录的上游提交：`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`。

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git thirdparty/depth-anything-3
git -C thirdparty/depth-anything-3 checkout --detach 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
conda env create -f environments/depth-anything-3.yml
conda activate depth-anything-3
python -m pip install -r environments/depth-anything-3-requirements.txt
python -m pip check
```

依赖文件使用与 SAM 相同的 PyTorch、torchvision、NumPy、OpenCV 和 setuptools，
另固定 xFormers `0.0.34`，补充上游模型代码需要的 `addict==2.4.0` 及结果导出依赖。
两个模型环境分别安装，不向基础 CPU 环境叠加模型依赖。

模型为 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`，下载到：

```bash
uvx --from huggingface_hub hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  config.json model.safetensors README.md \
  --revision b2359bdf726fb44ef62acca04d629dcf158053e7 \
  --local-dir thirdparty/depth-anything-3/weights/DA3NESTED-GIANT-LARGE-1.1
```

原部署记录中的 `model.safetensors` 大小为 6,759,558,100 字节，SHA256 为
`8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c`。
代码许可证及模型权重许可证分别查看上游仓库和模型页。

## 运行完整流程

在项目根目录、DA3 环境中执行：

```bash
conda activate depth-anything-3
python scripts/auto_weld.py assets/weld_images \
  --sam-python /path/to/envs/sam3/bin/python --gpu 0
```

不传 `--sam-python` 时，程序查找 DA3 环境同级的 `sam3/bin/python`。
DA3 先执行，SAM3 随后执行，最后由 DA3 环境完成几何后处理及结果导出。
源码通过 editable 安装，运行时应保留 `thirdparty/` 下的两个源码目录。
输入、输出和缓存约定见 [项目 README](../README.md)。
