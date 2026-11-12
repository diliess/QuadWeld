# Nested 双目图像与相机参数深度推理

2026-11-10：已在 QuadWeld 接入本地 `DA3NESTED-GIANT-LARGE-1.1`，同步输入左右 RGB、各自内参 K 和外参 R/t，输出左右目各自的米制光轴深度，并用独立保存的 RTX 深度评价。未修改模型结构、权重或第三方源码。

## 运行

在 QuadWeld 根目录执行。前两步分别使用已安装的 Isaac Sim 和 DA3 环境，第三步为 CPU 评价。

```bash
stereo_run="outputs/benchmarks/nested_stereo_$(date +%Y%m%d_%H%M%S)"

# 同步采集，当前默认仅 A04 正常初始臂姿
.venv/bin/python scripts/capture_hj90_stereo.py \
  --output-dir "$stereo_run/captures"

# 同一权重下对照：独立单目、无标定双目、有标定双目
.venv-da3/bin/python scripts/predict_depth_stereo.py \
  "$stereo_run/captures/A04_initial_arm/input.json" \
  --stream-blocks --output-dir "$stereo_run/predictions"

# 真值只在这一步参与数值评价
.venv/bin/python scripts/evaluate_depth_stereo.py "$stereo_run"
```

`--gpu 0` 可指定前两步使用的显卡。`--stream-blocks` 在每个 transformer block 计算时才调入显存，计算后移回 CPU，保留原权重 dtype 和 DA3 原生 autocast。本次在其他程序占用显存的 RTX 5090 上实际完成推理，有标定双目推理峰值约 3.2 GiB；显存充足时可省略此参数。

只处理已有双目 RGB 与标定、无需 Isaac 或真值时：

```bash
.venv-da3/bin/python scripts/predict_depth_stereo.py \
  /path/to/pair/input.json --modes stereo_calibrated \
  --stream-blocks --output-dir outputs/stereo_prediction
```

三组对照全部存在时，才能运行上述比较评价脚本。外部实拍图像可以推理，但该仿真评价脚本需要采集时保存的 RTX 真值和工件几何；它不会为实拍图像生成真值。

## 输入与输出约定

`input.json` 采用 `quadweld-stereo-v1`，采集脚本直接生成。完整示例保存在本次运行的 `captures/A04_initial_arm/input.json`。

| 字段 | 含义 |
|---|---|
| `synchronized: true` | 两张图像为同一时刻；本次暂停时间轴后同时渲染两目 |
| `views` | 按 left、right 排列的两个视图 |
| `views[].image` | 相对 JSON 所在目录的 RGB 文件路径 |
| `views[].image_sha256` | RGB 文件 SHA-256，防止误配标定与图像 |
| `views[].image_size_wh` | 原始图像宽、高，本次为 `[640, 1100]` |
| `views[].intrinsics` | 原图对应的 3×3 像素内参 K |
| `views[].extrinsics_w2c` | 4×4 世界到相机变换，平移以米为单位；不能填相机到世界变换 |
| `views[].distortion_coefficients` | 当前要求为零；实机图像需先去畸变并更新 K |
| `baseline_m` | 两个光心之间的距离，必须与外参计算一致 |
| `length_unit` | `m` |

相机坐标采用 OpenCV：X 向右、Y 向下、Z 向前。当前平行双目以左目光学系为世界参考，左目外参为单位阵，右目外参平移为 `[-0.09, 0, 0]`。内参约为 fx=767.616、fy=767.888、cx=320、cy=550，精确值以每次 USD 读取记录为准。这些是**仿真标定**，不是厂商实机标定。

推理调用把两张图片放入同一次 `model.inference(images, intrinsics=K, extrinsics=E)`。K 由 DA3 自己随缩放处理；外参采用其原生归一化。前向 hook 记录真正进入相机编码器的 K、归一化外参和图像尺寸，防止只在深度预测后替换内参而未向模型提供参数。

默认 `--process-res 504`，本次输出为高 504、宽 294；深度不是原始分辨率。每个 `stereo_calibrated/left_depth.npz`、`right_depth.npz` 包含：

- `depth_m`：该目光轴 Z 深度，float32、米；不是欧氏射线距离。
- `intrinsics`、`extrinsics_w2c`：与输出网格对应的标定参数。
- `processed_rgb`、`original_hw`、`image_sha256`：处理网格和输入对应关系。
- 模型 `confidence`、`sky`（若存在）。

同目录保存深度 PNG、处理后 RGB、前向参数审计 `report.json`，以及基线对齐前的预测深度和预测位姿 `before_baseline_alignment.npz`。PNG 每张使用各自百分位色阶，仅供预览；跨模式的数值对比请使用带统一米制色条的 `*_comparison.png`。

## 双目尺度恢复

本地 DA3 API 最后通过相机中心做通用 Sim(3) 位姿对齐。只有两个中心时协方差秩不足，会使该求解退化。`da3_stereo.py` 仅对本次模型实例的两视图末端对齐使用以下比例：

```text
C_i = -R_i.T @ t_i
s = ||C_right_calibrated - C_left_calibrated||
    / ||C_right_predicted - C_left_predicted||
depth_m = nested_depth_before_alignment * s
```

深度输出无需拟合全局旋转；已知相机位姿直接作为结果外参。输入标定参与前向计算，已知基线负责恢复米制尺度。RTX 深度、工件尺寸、目标距离均不进入推理，也不用于拟合尺度或偏移。预测基线塌缩会报错。

本次 A04/A02 的尺度因子分别为 1.071418 / 0.779948；标定前向完成、尚未按基线对齐时，工件 MAE 分别约 63.7–63.9 mm / 160.7–164.7 mm。因此下面的低误差属于**相机条件输入加已知基线尺度恢复**的完整结果。

## 本次实际核对

结果目录：`outputs/benchmarks/nested_stereo_20261110_01/`。Isaac Sim 6.0.0.1；PyTorch 2.10.0+cu128；RTX 5090；原 float32 权重，采用 DA3 原生推理精度设置。每次推理重设随机种子 0。

同一视图的三种模式使用相同 RGB、相同处理网格和相同可见工件掩码。掩码来自独立工件几何和 RTX 遮挡检查，每个面边界向内去除 3 个处理像素，评价过程中不按预测误差或模型置信度筛选像素。原始 RTX 深度通过实际 K 变换采样到处理像素中心，不跨边缘混合插值。

| 工位/视图 | 工件像素 | 单目 MAE mm | 无标定双目 MAE mm | 标定双目 MAE mm | 标定 RMSE mm | 标定 P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| A02 左 | 91330 | 390.79 | 922.26 | **4.32** | 5.10 | 8.61 |
| A02 右 | 90970 | 604.96 | 957.30 | **3.89** | 4.83 | 8.64 |
| A04 初始臂姿 左 | 41672 | 238.37 | 298.05 | **2.93** | 3.88 | 7.29 |
| A04 初始臂姿 右 | 52323 | 670.07 | 294.57 | **4.31** | 5.18 | 9.82 |

结果表仅评价工件面内部。**全画面有限深度像素**仍有明显误差：标定双目 MAE 在 A02 左/右为 88.55 / 182.34 mm，A04 左/右为 33.23 / 11.26 mm；包含机器人自身、边缘和其他场景表面，不能把工件内部结果推广到这些区域。

A02 工件真值平均深度约 572/593 mm，全部评价工件像素在 300–800 mm 内。A04 正常初始臂姿工件平均深度约 953/945 mm，超过 HJ-90 文档工作距离；其 300–800 mm 子集只有 1633/2453 像素，不能据此宣称全工件处于有效测量范围。`metrics.csv` 同时列出工作距离子集和全画面指标。

本次激光关闭，测试的是 RGB 双目输入 Nested；中央激光三角测量仍使用 [HJ-90 主动扫描入口](hj90_active_stereo.md)，尚未与 Nested 融合。A04 保持正常初始臂姿，A02 使用已有拍摄预设；场景源文件和机器人状态未写回。这是两个静态仿真预设的光轴深度核对，不能代表实机精度、0.03 mm 重复精度或焊缝三维定位精度。

结果索引：

- `README.md`、`metrics.csv`、`summary.json`：逐目、逐模式、逐区域统计。
- `A02_left_comparison.png` 等四张对照图：RGB、真值、三种预测及标定误差图。
- `predictions/model.json`：权重、代码哈希、第三方源码 commit、运行版本。
- `captures/*/evaluation/`：独立评价真值；推理输入 JSON 不引用它。
- `evaluation/*.npz`：共同网格的真值、预测和评价掩码，可独立重算指标。

验证：11 项单元检查通过，覆盖基线尺度、旋转参考系、退化基线、图像与标定一致性及像素中心采样等；已额外核对真实前向 K、保存的缩放前后深度和独立重算误差。
