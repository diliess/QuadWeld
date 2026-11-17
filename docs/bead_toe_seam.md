# 带焊道接头的焊趾线（焊道与两侧板面的交线）

接头上已有一道有体积的焊道（鱼鳞纹角焊缝）时，要的不再是两块板的交线，而是焊道表面与两侧板面相交的两条线（焊趾）。
它们不一定是直线，这里输出为两条三维折线。实现在 `weld_vision/bead_seam.py`，建立在 [双目平面交线](stereo_plane_seam.md) 之上。
本文不判断接头属于“无焊道”还是“有焊道”，由调用方选择。

## 思路

焊趾在板面上，所以不需要重建焊道曲面：板面由双目拟合成平面，焊趾只要在这个平面内找出“焊道从哪里开始”的边界。

1. **两块板的平面**。和无焊道时一样：SAM3 给出两块板的掩码，各自拟合视差平面，两平面的交线是被焊道盖住的根线。
   焊道只占掩码边上很窄的一条，掩码把焊道划给哪块板都可以。
2. **板面展开条带**。对每块板，以根线为横轴、板面内垂直根线的方向为纵轴，把左、右目图像各自重采样成“板面正视”的条带
   （采样步长为半个像素对应的板面长度）。条带里根线是一条水平直线，焊趾是离它 3–20 mm 的一条近似水平的曲线。
3. **在条带里找焊趾**。靠近根线的几行是焊道的外观，远离根线（焊脚上限以外）的几行是板面的外观，两者都按沿焊缝的滑动窗口局部估计。
   每个像素按“像焊道还是像板”打分，沿焊缝做动态规划得到一条连续的分界线，再从分界线向板一侧走到分数落到底的位置，
   即焊道侧面的坡脚。左目和右目各做一遍。
4. **有效性**。一个点只有同时满足以下条件才输出：左右目得到的位置相差不超过 0.6 mm；焊道与板面的外观差至少是各自波动的 3 倍；
   该处一个像素对应的板面长度不超过 1.6 mm（板面太斜时拒绝）。通过的点按板面方程直接得到三维坐标，所以严格落在板面上。
5. **避开焊道再拟合一次**。焊道旁边的板面会映出焊道、被焊道遮光，这一带不贴着板面，会把平面带偏。
   第一遍得到焊趾后，把焊道和它外侧一圈（1.5 倍焊脚且不少于 10 mm）从掩码里去掉，重新拟合平面，再找一次焊趾。
6. **多视角融合**（可选）。`fuse_toes` 把各视图的焊趾变到世界坐标，沿融合后的根线每 1 mm 取各视图的中位数，至少两个视图有值才输出。

选板：`pick_plate_pair` 从 SAM3.1 文本提示（`metal panel` 等）的候选里去掉细长的掩码（焊道本身、工作台边），
取公共边界最长的一对。本批 36 组全部选对。直接用文本提示分割焊道不稳定，所以没有依赖焊道掩码。

## 拍摄要求

- **基线不要与焊缝平行。** 板面纹理和鱼鳞纹都是周期性的，基线沿着焊缝时，差一个周期的深度同样对得上，
  本批正对焊缝、相机不转的 4 组全部被判为 `ambiguous_stereo_match`。把相机绕光轴转 35°–90°，或从斜向 30° 看过去即可。
  转 35°–60° 或斜视的视图比转 90° 的更准（见下表）。
- 相机到两块板的仰角都保持在 25° 以上，30°–55° 较好；太贴着某块板时，该板上的焊趾会被判无效或偏差变大。
- 距离 0.6 m 左右（一个像素约 0.8 mm）。更远时焊趾仍可输出，但精度按像素大小变差。
- 实在只能正对拍时，把其他视图得到的根线距离用 `expected_distance` 传入；这个距离要准到百分之几，名义工作距离不够用。

## 用法

```bash
# 1. 板件候选掩码（SAM3.1 文本提示，需要 GPU）
.venv-sam/bin/python -m model_runtime.sam_adapter <case>/input.json --output-dir <sam_dir>

# 2. 单组双目：根线 + 两条焊趾
.venv/bin/python -m weld_vision.stereo_io <case>/input.json \
  --candidates <sam_dir>/<case>.npz --bead --output-dir outputs/bead_toe_demo/<case>
```

除平面交线的输出外，增加：

- `toe_plate0.csv`、`toe_plate1.csv`：每 1 mm 一个点，`x_m,y_m,z_m`（左相机光学坐标系）、`valid`、
  `distance_from_root_m`（该处焊脚长度）、左图像素坐标。`valid=0` 的点是由邻近有效段带过来的占位值，不能当测量值用。
- `toes.json`：每条焊趾按有效段切开的折线、覆盖率、平均焊脚和检查量；`toes_overlay.png`：叠加图。

`--leg-range MIN MAX` 给出焊脚的搜索范围（默认 0.003–0.02 m），`--expected-distance` 传入已知的接缝距离。
已有两块板的掩码时仍可用 `--masks` 代替 `--candidates`。

在代码中调用：

```python
from weld_vision import bead_seam as bs
i, j = bs.pick_plate_pair(candidate_masks, scores)
result = bs.locate_bead_toes(left_rgb, right_rgb, K, baseline_m, candidate_masks[i], candidate_masks[j])
if result['status'] == 'ok':
    root = result['seam']['xyz']                     # 被焊道盖住的根线
    for toe in result['toes']:                       # 两块板各一条
        points = toe['xyz'][toe['valid']]            # N×3，米，左相机坐标系
        segments = bs.polyline_segments(toe)         # 按有效段切开的折线
```

`status` 除平面交线的各种状态外，还有 `no_bead_toe_found`（两块板上都没有找到通过检查的焊趾，例如接头上没有焊道）。

## 实测

数据：仿真场景中 A25（焊脚约 8 mm）、A26（约 10 mm）两个带双面角焊道的 T 形接头，每个工位 18 个相机位姿
（距离 0.45–0.75 m，仰角 22°–55°，绕光轴转 0°/±35°/60°/90°，斜视 ±30°，正面 16 个、背面 2 个），共 36 组 640×1100 的双目图像。
真值取自焊道网格与板面相接的顶点，只在评价阶段读取。

```bash
.venv/bin/python -m weld_scene.stereo_capture --output-dir outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare --gpu 0
.venv-sam/bin/python -m model_runtime.sam_adapter outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare/captures/*/input.json \
  --output-dir outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare/sam
.venv/bin/python -m tests.benchmarks.benchmark_bead_toes --run-dir outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare
```

单视图（SAM3.1 掩码，误差为输出点到真值焊趾折线的三维距离）：

| 项目 | 结果 |
|---|---|
| 输出 / 拒绝 | 32 / 36 输出；4 组正对且不转的视图拒绝（`ambiguous_stereo_match`） |
| 根线误差（各视图均值的中位数 / 最差视图） | 0.19 mm / 1.36 mm |
| 底板焊趾（均值的中位数 / 最差视图 / P95 的中位数 / 覆盖率中位数） | 0.30 mm / 1.53 mm / 0.54 mm / 79% |
| 立板焊趾（同上） | 0.20 mm / 1.19 mm / 0.32 mm / 91% |
| 每组耗时（单线程 CPU，不含 SAM3） | 4.8–8.9 s |

按拍摄方式分：转 35°/60° 和斜视的 18 组，焊趾均值误差 0.08–0.50 mm；转 90° 的 14 组为 0.21–1.53 mm，
其中最差的是仰角最低（22°）的 `r90_low`（A25 底板焊趾只剩 6% 有效，A26 整体偏 1.2–1.5 mm）。
被拒绝的 4 组在给定其余视图融合出的根线距离后都得到正确结果（根线 0.10–0.20 mm，焊趾均值 0.13–0.41 mm）。

多视角融合（世界坐标）：

| 接头 | 视图数 | 根线 mm | 底板焊趾 均值 / P95 / 最大 mm | 立板焊趾 均值 / P95 / 最大 mm | 输出长度（真值 0.900 m） |
|---|---:|---:|---:|---:|---:|
| A25 正面 | 14 | 0.08 | 0.15 / 0.39 / 0.68 | 0.17 / 0.56 / 0.71 | 0.842 / 0.870 m |
| A26 正面 | 14 | 0.09 | 0.43 / 0.90 / 1.12 | 0.16 / 0.26 / 0.79 | 0.846 / 0.866 m |
| A25 背面 | 2 | 0.12 | 0.31 / 0.61 / 0.71 | 0.22 / 0.34 / 0.58 | 0.447 / 0.441 m |
| A26 背面 | 2 | 0.11 | 0.37 / 0.59 / 0.74 | 0.20 / 0.30 / 0.79 | 0.346 / 0.462 m |

背面只拍了两个位姿，只覆盖了约一半长度。把 SAM 掩码换成仿真标签掩码，结果相近
（根线 0.31 mm，底板 0.32 mm，立板 0.29 mm，均为各视图均值的中位数），说明误差主要不来自分割。

逐视图表格、叠加图和融合折线见 `outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare/toes/README.md`
（标签掩码对照在同目录的 `toes_labels/`）。

## 限制

- 只在仿真图像上验证：理想针孔、无标定和位姿误差；相机位姿按夹具坐标系自由摆放，没有检查机械臂可达性。
- 焊趾位置取“焊道侧面坡脚”，其中的阈值（`toe_level`）是按仿真里焊道与板面的外观标定的。
  实际焊道的咬边、飞溅、氧化色和打磨痕迹会改变这个外观，需要用实拍重新标定和验证；精度量级为 0.3–0.6 个像素。
- 板面只当作平面。板有明显焊接变形或是曲面件时，焊趾仍会被找出，但三维坐标会带上板面拟合的偏差。
- 基线与焊缝平行的正对视图无法单独使用（见拍摄要求）；贴近板面的视角下该板的焊趾退化。
- 纹理只有一个方向（纯条纹）时，平面拟合可能在任何转角下都给出自信的错误深度；这是合成测试中见到的，场景图像里没有出现。
- 焊道两端（起弧、收弧处）没有输出：正面融合后的折线连续无断口，但比 0.900 m 的真值短 30–58 mm，都缺在两端。
  只有两个视图的背面，折线中间还有 5–42 mm 的断口。
- 没有覆盖 A27/A28 一类的坡口多层焊道，也没有自动区分有无焊道。
- 没有用 DA3：在这类近距离、大片均匀钢板的视图上，它的米制尺度不够准，不能用来消除上面的深度多解。

## 约 2 米粗定位与 0.6 米近拍扫查

末端双目先从约 2 米处拍摄全接头，使用两板的视差平面交线估计根线位置和距离。近拍以这条估计根线为依据，从左到右生成互相重叠的拍摄位姿。
相机中心相对估计根线的目标距离为 0.60 m，仰角为 35°，绕光轴转 −35°。将粗定位的板面投影到近拍图像，用于匹配两块板的分割候选。
两侧焊趾按实际拍摄外参转到世界坐标，每隔 1 mm 融合一次，默认要求至少两个视图支持。
参照组把中央视图改为 0.45 m 和 0.75 m，保持瞄准位置和朝向，比较有效长度和焊趾误差。

A25/A26 仿真中，约 2 m 的距离估计偏差为 +5.58 / +2.09 mm。0.60 m 中央视图两侧焊趾各有效覆盖 464–472 mm，0.45 m 时为 309–319 mm，0.75 m 时底板为 210–221 mm。
各工位 7 个 0.60 m 视图全部参与融合，根线平均误差为 0.15 / 0.06 mm，两侧焊趾同时覆盖约 93% 的 0.9 m 真值长度。
[本次实验报告](../outputs/benchmarks/20261116_02__stereo__A25-A26_front__sweep/results/README.md)

在仓库根目录用新的结果目录运行：

```bash
RUN=outputs/benchmarks/20261118_01__stereo__A25-A26_front__sweep
.venv/bin/python -m tests.benchmarks.plan_bead_sweep plan-overview --run-dir "$RUN"
.venv/bin/python -m weld_scene.stereo_capture --pose-plan "$RUN/overview_plan.json" --output-dir "$RUN/overview" --gpu 0
.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/overview/captures/*/input.json --output-dir "$RUN/overview/sam" --gpu 0
.venv/bin/python -m tests.benchmarks.plan_bead_sweep plan-close --run-dir "$RUN"
.venv/bin/python -m weld_scene.stereo_capture --pose-plan "$RUN/close_plan.json" --output-dir "$RUN/close" --gpu 0
.venv-sam/bin/python -m model_runtime.sam_adapter "$RUN"/close/captures/*/input.json --output-dir "$RUN/close/sam" --gpu 0
.venv/bin/python -m tests.benchmarks.measure_bead_sweep --run-dir "$RUN"
```

`results/README.md` 汇总粗定位距离偏差、根线与两侧焊趾的三维误差及覆盖率。逐视图误差和端点偏差见 `results/metrics.json`。各工位目录保存世界坐标 JSON、带有效标记的 CSV 及叠加图。
近拍预测只使用双目图像、分割候选、标定和相机外参。网格仅用于评估。相机刚性安装在机械臂末端，使用仿真中既有的 50 mm 支架偏置。
仿真按离散位姿停拍，检查了相邻位姿间的静态插值点。该检查没有运行机身行走、关节跟踪和振动的动力学。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -p test_bead_seam.py
```

用光线追踪生成焊脚按正弦变化的带焊道 T 形接头双目图像，检查：焊趾误差小于 1 mm 且确实不是直线；
无焊道接头返回 `no_bead_toe_found`；掩码把焊道划给任一块板都能工作；条纹纹理下要么拒绝要么正确，给定距离后正确；
选板忽略细长掩码；多视角融合的中位数和最少视图数。

## 其他接头与无焊道负例（2026-11-17）

已测试 A01、A03、A23、A06、A04、B01 共 18 组双目，并补测 A25/A26 背面 4 组。
无焊道区域的融合根线平均线位置误差为 0.053–0.279 mm，但 A04 端点出现约 84.5 mm 外延误差。
将无焊道区域强制送入焊趾函数时，14/18 组仍输出短段候选，说明该函数不能用作有无焊道分类器。
当前调用方仍须选择根线或已有焊道模式。条件、完整指标、原图和世界坐标轨迹见 [本轮测试报告](candidate_regions_test.md)。

## 分阶段候选区域复测（2026-11-17）

新增 [8 区域完整实验](candidate_regions_test.md)，覆盖初始双目、粗定位、三个测量驱动近拍视角、SAM3.1、板面拟合、世界坐标融合和独立评价；A25/A26 背面另保留角平分线视角与主流程 35° 观察方向的对照。35° 下有效焊趾平均误差约 0.15–0.27 mm，但三个视角且每点要求两视角支持时，全长覆盖只有约 58%–60%，需要端部补拍。原始失败与重试过程全部保留。
