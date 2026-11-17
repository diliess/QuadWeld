# QuadWeld

从标定双目图像提取焊缝根线和已有焊道两侧的三维焊趾。`main.py` 只运行双目扫查，默认工位 A25，也支持 A26。

## 双目扫查

1. 末端双目相机在约 2 m 处拍摄全接头。
2. SAM3.1 使用文本提示生成板件候选掩码，双目视差平面拟合确定根线位置和距离。
3. 根据整体测量结果规划约 0.60 m 工作距离的重叠近拍位姿。
4. 逐视角采集、分割并测量根线和两侧焊趾。
5. 使用相机外参进行多视角融合，输出世界坐标折线。

米制尺度来自双目标定；识别阶段使用 RGB、标定、SAM 候选和相机位姿。渲染深度与场景焊缝网格只供独立评价。

定位算法已拆为两个可独立调用的 Python 功能包：`weld_vision.coarse_localization` 负责远距离粗定位，
`weld_vision.multiview_localization` 负责近距离逐视角测量和融合。输入输出与调用方式见
[两阶段定位功能包](docs/localization_packages.md)。

```bash
# A25 在线采集与识别
.venv/bin/python main.py --station A25 --headless

# 只检查 A26 所需环境和模型
.venv/bin/python main.py --station A26 --check

# 用已保存的图像、标定和 SAM 掩码复跑，无需 GPU 或模型环境
.venv/bin/python main.py --station A26 --replay outputs/benchmarks/20261116_02__stereo__A25-A26_front__sweep
```

在线采集使用 Isaac Sim 和独立的 SAM3 Python 3.12 环境。位姿规划还需要 MuJoCo、USD Python 绑定和碰撞模型，
默认使用相邻 `robosimstudio` 仓库中的工业焊接场景。采集按离散位姿停拍，并检查位姿间的静态插值点；
该检查不运行机身行走、关节跟踪或振动的动力学。

SAM3.1 的源码、权重和环境准备见 [模型部署说明](thirdparty/README.md)。
基础 `pyproject.toml` 定义 CPU 后处理依赖；Isaac Sim 与 SAM3 分别使用已有的独立运行环境。
默认优先查找项目 `.venv/bin/python` 作为 Isaac Python，SAM 可使用 `.venv-sam/bin/python` 或 Conda `sam3` 环境。
也可通过 `QUADWELD_ISAAC_PYTHON`、`QUADWELD_SAM_PYTHON` 或对应命令行参数指定解释器。

## 参数

| 参数 | 用途 |
| --- | --- |
| `--station A25/A26` | 选择工位，默认 A25 |
| `--replay PATH` | 读取保存的双目采集数据与 SAM 候选，重新测量及融合 |
| `--collision-model PATH` | 在线位姿规划使用的 MuJoCo 碰撞 XML |
| `--min-views N` | 融合焊趾点所需最少支持视图数，默认 2 |
| `--output-dir PATH` | 本次输出目录，要求不存在或为空 |
| `--gpu N` | 在线采集和分割使用的物理 GPU 编号，默认 0 |
| `--isaac-python PATH`、`--sam-python PATH` | 两个运行环境的 Python 路径 |
| `--semantic-threshold VALUE` | SAM 分割阈值，默认 0.25 |
| `--headless` | 兼容现有命令；双目采集始终不打开 GUI |
| `--check` | 检查环境或回放文件后退出，不执行测量 |

默认输出到 `outputs/runs/<台北日期时间>__stereo__<工位>_front__sweep/`；回放运行使用 `__replay` 后缀，结果写入新目录。
`--pipeline` 及单张 RGB、深度模型和激光扫描分支参数已从主入口移除。

## 输出

```text
run_report.json                 运行状态、来源和结果索引
overview_plan.json              整体观察位姿计划（在线采集）
overview/captures/<视图>/        整体双目 RGB 与 input.json 标定
overview/sam/                   整体观察的 SAM 候选
close_plan.json                 近距离扫查计划（在线采集）
close/captures/<视图>/           近距离双目 RGB 与标定
close/sam/                      近距离 SAM 候选
*_capture.log、*_sam.log         在线采集和分割日志
results/predictions.json         逐视图测量与融合状态
results/<工位>/fused.json        成功融合的根线与两侧焊趾
results/<工位>/*_toe.csv          两侧焊趾、有效标记和支持视图数
results/<工位>/overview_overlay.png  整体观察图上的融合结果
```

融合曲线位于世界坐标系，单位米。有效性不足的区段不会被当作有效测量点。
`run_report.json` 保存执行状态；双目融合状态为 `ok` 时返回 0，结果不完整时返回 2，入口捕获的运行异常返回 1。

## 单组双目与评价

```bash
# 从左目生成板件候选
.venv-sam/bin/python -m model_runtime.sam_adapter <case>/input.json --output-dir <sam_dir>

# 单组双目：根线与两侧焊趾
.venv/bin/python -m weld_vision.stereo_io <case>/input.json \
  --candidates <sam_dir>/<case>.npz --bead --output-dir outputs/bead_toe_demo/<case>

# 保存数据的独立真值评价
.venv/bin/python -m tests.benchmarks.measure_bead_sweep --run-dir <run_dir>

# 单元测试
.venv/bin/python -m unittest discover -s tests -t .
```

算法、仿真验证、误差统计及限制见 [带焊道接头的焊趾线](docs/bead_toe_seam.md) 和
[双目平面交线](docs/stereo_plane_seam.md)。
[八区域分阶段双目实验](outputs/benchmarks/20261117_01__stereo__multi8__staged/README.md)记录初始视角、粗定位、三个近拍视角、SAM3.1 分割、板面拟合、融合和独立误差评价。
历史 DA3、Nested、注册 RGB-D 和单图识别记录已移入 [非当前双目实验归档](outputs/archive/non_stereo_experiments_20261117/README.md)，原位置映射和逐文件校验清单保存在归档内。
输出目录的当前索引、命名规则与旧路径映射见 [outputs 说明](outputs/README.md)；本次清理情况见 [保留与清理记录](docs/output_retention.md)。

## 场景

完整场景可在 Isaac Sim 中打开 `assets/scenes/industrial-welding/industrial_welding_full.usda`。
场景同步工具继续使用独立模块入口，需要 USD Python 绑定：

```bash
uv run --locked --extra usd python -m weld_scene.import_scene
```

## 代码结构

| 路径 | 职责 |
| --- | --- |
| [`main.py`](main.py) | 解析双目运行参数、检查环境、调度采集和测量、记录运行状态 |
| [`weld_vision/`](weld_vision) | 双目标定读入、视差平面拟合、根线与焊趾提取、多视角融合 |
| [`weld_vision/coarse_localization/`](weld_vision/coarse_localization) | 远距离双目粗根线、板面法向与距离 |
| [`weld_vision/multiview_localization/`](weld_vision/multiview_localization) | 近距离逐视角测量、世界坐标融合和结果导出 |
| [`weld_scene/`](weld_scene) | 场景同步、相机标定、机械臂运动学、位姿规划和仿真采集 |
| [`model_runtime/`](model_runtime) | SAM3.1 适配器、Isaac/SAM 独立解释器与进程管理 |
| [`thirdparty/`](thirdparty) | SAM3 上游源码、权重、部署说明和补丁 |
| [`tests/`](tests) | 单元测试、独立真值和双目评估工具；共享评价函数位于 `tests/benchmarks/seam_metrics.py` |
| [`assets/`](assets) | 工业场景 USD、贴图、来源记录及历史 RGB 样本 |
| [`calibration/`](calibration) | HJ-90 仿真双目参数与厂商 RGB 标定 |
| [`environments/`](environments) | SAM3.1 环境定义及依赖版本 |
| [`docs/`](docs) | 当前算法说明与双目实验记录 |
| `outputs/` | 本地采集、缓存、预测、图像和报告，由 Git 忽略 |
