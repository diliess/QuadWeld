# QuadWeld

从完整 RGB 图片自动提取三维接缝和已有焊道两侧的焊趾曲线。
[`main.py`](main.py) 集成 A04 场景加载、HJ-90 拍照及自动识别；
[`scripts/auto_weld.py`](scripts/auto_weld.py) 也可独立处理 RGB 图片。
无需输入深度、人工选区、点提示或焊接状态。

```text
RGB → DA3 深度与内参 → SAM3 板件/焊道分割 → 板面拟合
    → 接缝交线 / 成对焊趾 → 几何有效性与连续区间 → 结果导出
```

## A04 一次拍照与识别

`main.py` 直接加载已保存的 `A04_initial_arm` 场景：机器狗位于 A04 桌子长边，
机械臂固定为原工程正常初始关节角 `[0, 1.2, 2.3, 0, 1.3, 0]` rad。
HJ-90 保持焊枪上的固定安装，使用该臂姿下的自然视野，不做朝向目标的 IK 调整。
此流程是静态摆位拍照，不执行机器狗行走或机械臂运动轨迹。

依次执行：加载场景 → 拍摄一张 640×1100 RGB → 退出 Isaac 进程释放显存 →
DA3 预测深度 → SAM3 分割和几何识别。相机内参从当前 USD 相机读取，传给识别流程；
不采集或使用渲染器深度，也不把场景中的焊缝位置传给识别算法。

准备已有的 Isaac Sim Python 3.12 环境，以及下文的 DA3、SAM3 环境，然后执行：

```bash
.venv/bin/python main.py --headless \
  --da3-python /path/to/envs/depth-anything-3/bin/python \
  --sam-python /path/to/envs/sam3/bin/python
```

默认 Isaac Python 为项目 `.venv/bin/python`；其他安装可通过 `--isaac-python` 指定。
可设置 `QUADWELD_ISAAC_PYTHON`、`QUADWELD_DA3_PYTHON`、`QUADWELD_SAM_PYTHON`，
以后直接运行 `python main.py --headless`。脚本也会尝试当前解释器、本地 `.venv-da3`、
`.venv-sam` 以及常见 Conda 环境目录。
不加 `--headless` 时，拍照阶段显示 HJ-90 视角；拍照结束后窗口关闭，随后执行模型推理。

```bash
# 只检查环境、模型文件和 GPU，不启动 Isaac
.venv/bin/python main.py --check

# 仅查看初始臂姿下相机实际拍到的 RGB
.venv/bin/python main.py --headless --capture-only
```

| 参数 | 用途 |
| --- | --- |
| `--gpu` | 物理 GPU 编号，默认 0 |
| `--output-dir` | 本次运行目录，要求为空；默认自动生成带台北时间戳的目录 |
| `--isaac-python`、`--da3-python`、`--sam-python` | 三个运行环境的 Python 路径 |
| `--model-cache-dir` | 可选的 DA3/SAM 模型输出缓存目录 |
| `--semantic-threshold` | SAM 初始阈值，默认 `0.25` |

默认输出为 `outputs/runs/a04_<时间戳>/`：

```text
rgb.png             初始臂姿下 HJ-90 拍到的一帧 RGB
camera.json         此帧分辨率下的相机内参
capture.json        实际臂姿、机器狗位置、相机位姿和图片哈希
capture.log         Isaac 拍照进程日志
run_report.json     整条流程的状态和结果索引
auto_weld.log       自动识别日志
auto_weld/          自动识别图片、三维曲线、JSON 和 NPZ
```

保持正常初始臂姿时，相机画面可能没有足够的焊缝证据。识别结果会如实保留
`no_candidate` 或无有效区间，不会为了获得候选而自动转动机械臂。

## 安装与运行

模型使用两个独立的 Python 3.12 Conda 环境。按
[模型部署说明](thirdparty/README.md) 安装源码、依赖和权重后，在项目根目录执行：

```bash
conda activate depth-anything-3
python scripts/auto_weld.py assets/weld_images/tacked_front_01.jpg

# 目录中的图片分别独立处理
python scripts/auto_weld.py assets/weld_images

# 指定 SAM 环境、GPU 和输出目录
python scripts/auto_weld.py assets/weld_images \
  --sam-python /path/to/envs/sam3/bin/python \
  --gpu 0 --output-dir outputs/welds
```

默认 SAM 环境位于当前 DA3 环境的同级 `sam3/`，可用 `--sam-python` 覆盖。
`--gpu auto` 为每个模型选择空闲显存最多的 GPU；DA3 和 SAM3 依次运行。
两份模型权重必须预先下载，具体路径见部署说明。

## 模型缓存和参数

```bash
python scripts/auto_weld.py assets/weld_images \
  --model-cache-dir /tmp/quadweld_auto_models
```

缓存校验输入图片 SHA256；SAM 缓存同时校验阈值与图片顺序。
更换模型权重、源码或推理设置后应使用新缓存目录。

| 参数 | 用途 |
| --- | --- |
| `images` | 一张或多张 RGB 图片，或图片目录 |
| `--output-dir` | 结果目录，默认 `outputs/offline/depth_anything_3` |
| `--gpu` | 物理 GPU 编号，或 `auto` |
| `--sam-python` | SAM3 环境的 Python 路径 |
| `--semantic-threshold` | SAM 初始阈值，默认 `0.25` |
| `--joint-band-width` | 原图中交线展示带宽，默认 5 像素 |
| `--model-cache-dir` | 保存并复用 DA3 和 SAM 推理结果 |
| `--reuse-depth-dir` | 复用本流程生成、且图片哈希匹配的 DA3 深度缓存 |
| `--intrinsics` | 可选的针孔相机标定 JSON；默认使用 DA3 预测内参 |

`--intrinsics` 只替换内参，深度仍由 DA3 生成。JSON 格式：

```json
{
  "image_size": [640, 480],
  "camera_matrix": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
  "distortion_coefficients": [0, 0, 0, 0, 0]
}
```

图片必须与标定尺寸一致，并已去畸变。程序将内参换算到 DA3 的处理分辨率。

## 输出

| 文件 | 内容 |
| --- | --- |
| `auto_overview.jpg` | 全图接缝和焊道候选 |
| `weld_toes_overview.jpg` | 成对焊趾、待确认点和截面证据 |
| `seam_paths.csv` | 两种分支统一导出的三维曲线 |
| `data/auto_report.json` | 板面、区段状态、曲线索引和连续有效区间 |
| `data/auto_results.npz` | 深度、内参、掩码及二维/三维曲线数组 |
| `data/weld_toes.csv` | 焊趾表面估计和逐点有效性标记 |
| `data/sam_plate_overview.jpg`、`data/sam_weld_overview.jpg` | 几何过滤前的 SAM 分割结果 |
| `README.md` | 本次结果索引 |

三维坐标位于估计相机光学坐标系：X 向右、Y 向下、Z 向前，单位米。
CSV 按 `image,path_id,geometry_run` 分组，排除 `geometry_run=-1`，按 `sample` 排序；
不同有效区间不能直接连接。

未接受焊道覆盖的区段保留板面交线候选；接受的已有焊道输出两侧焊趾。
未检出焊道不代表工件未焊接。DA3 深度与三维曲线是模型估计，尚无毫米级定位精度验收。
完整算法、阈值和结果字段见 [流程说明](docs/sam3_da3_weld_detection.md)。

## 工业焊接场景

### DA3 多样本深度评价

```bash
.venv/bin/python scripts/benchmark_da3.py --gpu 0
```

从 A01、A02、A04、A06 各采集三个独立视角，使用同一 HJ-90 内参和
0.8、1.2、1.8 m 目标距离。不同距离同时改变视角，不单独归因于距离。
机械臂保持正常初始姿态，新增视角由独立评测相机获得。
DA3 只接收 RGB；渲染器光轴深度仅用于离线评价。
脚本排除夹具、遮挡及板面边界，输出逐帧 MAE、RMSE、相对误差及总览图，
保存到 `outputs/benchmarks/da3_<时间戳>/`。
如存在此前 A04 原始深度评价，也将其作为缓存基线单独列出。
此测试不执行 SAM3 或接缝识别。

`--resume --output-dir <目录>` 可恢复已校验的采集和推理；
`--evaluate-only --output-dir <目录>` 可在 CPU 上重新计算已有结果。

保留 [`assets/scenes/industrial-welding/`](assets/scenes/industrial-welding/README.md) 完整场景包，
包括 B2/E05L-Pro/HJ-90、工位、相机、贴图和四个拍摄姿态。

在已有的 Isaac Sim 中通过 File > Open 打开
`assets/scenes/industrial-welding/industrial_welding_full.usda` 查看完整场景。

[`scripts/import_industrial_welding.py`](scripts/import_industrial_welding.py) 保留用于从相邻
RoboSimStudio 工程同步场景更新。导入需要 USD Python 绑定：

```bash
uv run --locked --extra usd python scripts/import_industrial_welding.py
```

## 代码结构

```text
main.py                   A04 正常初始臂姿加载、RGB 拍照与识别总入口
scripts/auto_weld.py       图片输入、SAM 推理、流程编排和结果导出
scripts/predict_depth.py   DA3 模型工作进程
scripts/import_industrial_welding.py  工业场景导入
vision/weld_detector.py    板面拟合、接触交线、焊道筛选及状态分段
vision/weld_toes.py        成对焊趾定位
vision/weld_surfaces.py    局部承接表面和三维连续性检查
vision/skeleton.py         焊道掩码骨架细化
assets/weld_images/        RGB 样本及来源
assets/scenes/industrial-welding/  工业焊接场景包
```

基础 `pyproject.toml` 用于 CPU 后处理依赖；完整识别使用模型环境的依赖文件。
`usd` 是场景导入脚本的可选依赖，RGB 识别流程无需安装它。
