# 整体观察距离下的多视角焊缝识别

结果报告：[原图、预测焊缝投影和平均误差](../outputs/benchmarks/t_station_full_joint_20261110_01/README.md)。

## 采集与推理

以 `t_station_full_frame_20261110_01` 的三个静态站位为基准。各工位的机器狗 `base_link` 位姿完全固定，相机保持刚性安装在机械臂末端；通过实际六关节 IK 改变拍摄位置及朝向，不是移动独立虚拟相机。
每工件采集左侧、中心、右侧三个臂姿，各保存同步左右目，共六张 640×1100 RGB。三工位共 9 个臂姿、18 张图。全部双目保留至少 4% 画幅余量，焊枪像素为 0；相机内参、90 mm 基线及此前的仿真安装偏置保持一致。

| 工位 | 相机至工件中心 | 相对中心的观察方位角 |
|---|---:|---|
| A15 | 1.96 m | -8° / +0° / +8° |
| A03 | 1.98 m | -6° / +0° / +6° |
| A21 | 1.90 m | -8° / +0° / +8° |

上表角度表示相机绕工件的观察方位变化，不是某个机械臂关节的转角。各关节角与每帧相机位姿见 `station_plan.json` 和对应采集目录的 `evaluation/capture.json`。这些是通过静态接触、支撑和关节限制检查的独立位姿，未执行行走或关节运动轨迹。

处理顺序：

1. 每工件六张 RGB 与已知内参、FK 外参一次输入 `DA3NESTED-GIANT-LARGE-1.1`。前向钩子确认输入为 `[1,6,3,504,294]`，不是六次独立单图推理。
2. 沿用相机间已知距离的最小二乘尺度恢复。只使用基线与相机位姿，不使用工件尺寸、渲染深度或真实焊缝对齐深度。
3. 用预测深度和相机重力方向自动寻找前方竖板与其底部相邻横板，自动生成两组正负点，调用 SAM3.1 分开分割横板、竖板。整体距离的预测欧氏距离范围为 1.2–2.8 m；原近距离流程仍默认 0.3–0.8 m。
4. 水平候选与重力方向的法向余弦至少 0.97，排除倾斜窄边；横板按预测高度与竖板底部的接近程度选择，避免大面积地面优先。这个规则只用于开启 `--contact-pair` 的整体视角。
5. 运行现有 SAM 板面拟合和交线识别。18 目中 2 目直接由 SAM 内部拟合得到接缝，16 目启用了现有整图几何回退；不能把本次结果描述为全部直接由 SAM 掩码拟合成功。
6. 保留原始检测输出，另保存前景筛选结果：预测线点需距两张独立板件掩码各不超过 7 个处理像素，并处于 1.2–2.8 m 的预测距离内。该规则用于排除背景接缝，不按照真实误差删点，也不保证完全消除误检。
7. 最后才用 USD 名义焊缝和 RTX 深度进行独立评价；所有预测、原始掩码和筛选结果均已先保存。

## 评价结果

焊缝误差为保留预测点到画内有限名义焊缝线段的平均三维距离，各视角采样点汇总加权。不是端点配准误差，也不是只统计低于某个真实误差阈值的点。部分视角仍仅恢复局部线段，平均误差不代表整条焊缝已完整识别。

| 工位 | 焊缝平均误差 | 保留点 / 原始点 |
|---|---:|---:|
| A15 | 15.43 mm | 800 / 904 |
| A03 | 58.69 mm | 705 / 737 |
| A21 | 14.53 mm | 932 / 1386 |

同一中心视角再做双目两图推理作为对照，与六图联合的中心输出比较。两者使用相同中心 RGB、相同深度评价像素及同一权重。深度指标为可见底板/腹板内部的轴向 Z MAE：独立 RTX/CAD 面匹配 5 mm 内，逐面腐蚀 3 个处理像素；不是全画面指标。

| 工位 | 中心双目两图 MAE | 六图联合中心 MAE |
|---|---:|---:|
| A15 | 69.78 mm | 15.23 mm |
| A03 | 47.00 mm | 59.00 mm |
| A21 | 216.20 mm | 17.96 mm |

A15、A21 改善，A03 反而变差。多视角没有保证此次所有工件都提高精度。当前约 1.9–2.0 m 的观察距离超出 HJ-90 文档标称的 300–800 mm 测量范围，本次结果不能作为厂商标称测量精度的验证。

## 复现

在项目根目录执行，使用新的空结果目录：

```bash
RUN=outputs/benchmarks/t_station_full_joint_new
.venv/bin/python scripts/prepare_full_multiview.py --output-dir "$RUN"
.venv/bin/python scripts/capture_station_overviews.py --resume --output-dir "$RUN" --gpu 0
.venv-da3/bin/python scripts/benchmark_full_station_views.py --stage depth --run-dir "$RUN" --gpu 0
.venv-sam/bin/python scripts/segment_foreground_plates.py --run-dir "$RUN" --output-dir "$RUN/plate_parts" --distance-range 1.2 2.8 --contact-pair --gpu 0
.venv/bin/python scripts/benchmark_full_station_views.py --stage seams --run-dir "$RUN"
.venv/bin/python scripts/benchmark_full_station_views.py --stage report --run-dir "$RUN"
```

`seams/predictions/` 保存全部原始候选，`seams/foreground_predictions/` 保存仅用预测信息筛选的候选；`plate_parts/` 保存原始 SAM 输出、横竖板最终掩码与自动提示，`metrics.json` 保存逐目误差、覆盖率与汇总。
