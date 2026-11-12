# A15 / A03 / A21：拉近至650 mm工作距离

2026-11-10，结果见 [本次完整报告](../outputs/benchmarks/t_station_close_20261110_01/README.md)。

沿用上一轮50 mm侧向支架偏置、90 mm双目基线、内参和无焊枪视野，实际改变仿真机械臂关节与底座站位，把局部焊缝中心拉近至650 mm。逐目验证画内焊缝真实欧氏距离约 **650–704 mm**，位于300–800 mm工作范围内。不是数字放大、修改焦距或裁剪RGB。

近距离无法一次看全840 mm前侧名义焊缝，每工位按CAD几何选择左、中、右三个重叠拍摄姿态，共9组同步双目、18张原图。三组视野并集均覆盖完整名义线，18目焊枪像素均为0。规划先于模型运行，不按预测误差选择视角。

| 工位 | 俯角 | 工件额定范围深度MAE | 范围筛选后焊缝点均距 | 整缝覆盖@10mm |
|---|---:|---:|---:|---:|
| A15 | 35° | 4.06 mm | 8.83 mm | 82.9% |
| A03 | 25° | 26.91 mm | 33.12 mm | 51.8% |
| A21 | 35° | 18.66 mm | 22.03 mm | 33.5% |

深度误差评价独立RTX真值300–800 mm内、可见且逐面腐蚀3像素后的底板/腹板内部；完整可见工件指标及额定像素比例也保存在summary.json。画内焊缝全部位于工作距离内，不能将此条件推广到画面中的所有背景。

按本次要求，报告主表剔除模型估计欧氏距离不在300–800 mm内的候选点。筛选只使用预测点到当前镜头的距离，范围内的点全部计入，不根据真值或定位误差大小删点。A15和A21中部视角的远处背景接触线候选因此被剔除。范围内仍可能存在误检，以上数值准确地表示范围筛选后候选的误差。

筛选结果保存在 [范围筛选统计](../outputs/benchmarks/t_station_close_20261110_01/seam_range_filtered_metrics.json)，与既有 [距离分组诊断](../outputs/benchmarks/t_station_close_20261110_01/candidate_range_diagnostic.json) 逐项核对一致：

| 工位 | 模型估计300–800mm内的预测点：数量 / 均距 | 范围外预测点：数量 / 均距 |
|---|---:|---:|
| A15 | 746 / 8.83 mm | 129 / 2108.55 mm |
| A03 | 838 / 33.12 mm | 0 / N/A |
| A21 | 575 / 22.03 mm | 102 / 2536.44 mm |

原始全候选均距318.39 / 33.12 / 400.86 mm及所有原始预测继续保存在summary.json和seams目录中。此次只重算报告统计，没有重新推理或修改原始预测。范围筛选后重新核对整缝覆盖率，三个工位均与筛选前一致。

整缝覆盖率是各视图独立核对后按同一名义弧长求逻辑并集：每个名义样本在至少一张实际画内图像中距保留的预测线段≤10 mm才计入；剔除点前后的线段不跨越连接。没有执行深度融合、ICP或按真值拟合尺度，理想FK和相机外参仅用于仿真采集/评价。不同距离的视角与像素不同，不属于同像素消融。

## 查看与复跑

- [机械臂姿态、原图、深度和焊缝](../outputs/benchmarks/t_station_close_20261110_01/close_results.png)
- [左中右三个重叠视角](../outputs/benchmarks/t_station_close_20261110_01/overlapping_views.png)
- [逐段整缝覆盖图](../outputs/benchmarks/t_station_close_20261110_01/full_seam_coverage.png)
- [全部18目详情与姿态USD](../outputs/benchmarks/t_station_close_20261110_01/README.md)

```bash
.venv/bin/python scripts/benchmark_close_station_views.py --gpu 0

# 复用当前结果重新核对和汇总
.venv/bin/python scripts/benchmark_close_station_views.py --evaluate-only \
  --output-dir outputs/benchmarks/t_station_close_20261110_01

# 仅复用已有预测，更新范围筛选统计与报告
.venv/bin/python scripts/benchmark_close_station_views.py --report-only \
  --output-dir outputs/benchmarks/t_station_close_20261110_01
```

每个captures/*/arm_pose.usda显式声明米制/Z轴向上，可在Isaac Sim打开；依赖上一轮无焊枪安装场景。已核对关节限位、静态碰撞、足部接地、质心支撑、相机刚性、深度单位及像素标注，未执行行走或关节控制轨迹。原场景USDC与用户GUI没有写回，新增支架质量/刚度仍未知。

28项单元检查通过；`validate_run.py`独立重算18个视图的距离、深度MAE与整缝并集，并核对导出姿态、单位、哈希和静态检查，结果记录于`validation.json`。
