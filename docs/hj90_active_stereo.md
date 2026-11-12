# HJ-90 双目与中央激光功能仿真

2026-11-09：QuadWeld 新增两路独立相机、中央线激光投射、摆动扫描、RGB 条纹检测和双目三角测量。按用户确认采用仿真标定参数，配置为 `calibration/hj90_sim.json`。原 DA3/SAM 单张 RGB 工作流继续保留；本入口不调用 DA3，也不以 RTX 深度作为传感器输出。

2026-11-12：按新提供的厂商资料接入独立 RGB 通道，采用 `calibration/hj90_rgb_vendor.json` 的已校正 RGB 内参和左目到 RGB 的外参。该记录不提供双目左右目内参或激光标定；本页扫描链路仍使用原有仿真双目参数。原生 RGB 分辨率、坐标及使用方式见 [资料对齐说明](hj90_material_alignment.md)。

这是对文档所述**测量链路的功能复现**。缺少厂家左右目内外参、激光参数、光功率、曝光响应和扫描机构标定，不能称为与实机完全等价，不能宣称达到宣传册的 0.03 mm 重复精度或测量精度。

## 已实现的链路

| 文档要求 | 本次实现 |
|---|---|
| 左右目，90 mm 基线 | 两个独立 RTX render product、相同仿真内参；光心分别在中点左右 45 mm，与焊枪刚性连接 |
| 中央激光器与摆动机构 | 中央窗口位置的 RTX `RectLight isProjector`，窄高斯线光纹理，绕竖轴左右扫描；不是画在 RGB 上的蓝线或贴在工件上的几何条 |
| 标配 450 nm 蓝光 | 记录 450 nm，RGB 蓝色辐射近似；没有相干光、散斑或波长级光谱响应模型 |
| 静态扫描 | 相机/工件静止，激光角度从 −23° 到 +23°；每个角度同一次渲染更新取得左右目曝光 |
| 动态扫描 | 激光角固定，每个采样重新取得同一目标位姿下的 on/off 双目图像；提供移动台阶试件验证。工业场景本身不含运动控制器，没有给机器人编造行走动作 |
| 深度和点云 | 蓝光 on/off 差分 → 亚像素条纹 → 同行匹配 → `Z=fx×0.09/disparity` → 激光平面一致性筛除 → 300–800 mm 有效区；输出原始条纹点云与稀疏深度 |
| 灰度图 | 静态基准左右目 RGB 另转换保存灰度 PNG；这是彩色仿真图的灰度转换 |
| 采集速率 | 约束虚拟时间的线速不超过 5000 线/s，静态全帧至少 0.2 s；默认 640 线/0.4 s。离线 RTX 渲染耗时不代表实机帧率 |

曝光采用理想同步、零运动模糊。静态模式共用一组无激光背景，动态模式每个目标位姿重新拍背景。该 on/off 方案是仿真实现选择，不声称厂家采用同一曝光时序。物体遮挡、材质反射和投影阴影由 RTX 渲染；弱回波、多重亮条、异常视差和几何不一致会无效。有效掩码仅表示通过这些检查，仍可能包含反射误匹配。

640×1100 和内参由旧工程 800 mm 档视场推导，不是厂家公布的真实传感器分辨率或标定。激光位置相对双目中点 `[0,0.004,0]` m（OpenCV 坐标）、线宽、±23° 扫角和光强均为可配置假设。当前两镜头采用相同内参、平行零畸变；加载非零畸变会明确报错，未伪装支持实机畸变。电动翻盖仍是源场景的静态开启几何；热过程、防护性能、电气接口和原厂固件未仿真。

## 使用

在 QuadWeld 根目录运行，复用本机已安装的 Isaac 环境：

```bash
# 保持 A04 长边站位和正常初始臂姿，执行双目+激光扫描
.venv/bin/python main.py --sensor hj90 --headless

# 工件 A02 的已有拍摄预设；128 线快速检查
.venv/bin/python scripts/scan_hj90.py --pose A02 --headless --lines 128

# 标准 640 线扫描；只把 RTX 深度用作评价
.venv/bin/python scripts/scan_hj90.py --pose A02 --headless --lines 640 --evaluate

# 固定激光角，移动合成试件；验证动态采样，不是机器人动力学
.venv/bin/python scripts/scan_hj90.py --fixture step --mode dynamic \
  --target-speed -0.1 --lines 24 --headless --evaluate

# 仅用保存的 RGB 和标定重新计算；不需要 Isaac、场景或评价真值
.venv/bin/python scripts/reconstruct_hj90.py outputs/hj90/<本次目录>
```

`--isaac-python /path/to/python` 可以显式指定渲染环境。主入口 `--sensor rgb` 是默认的原 DA3/SAM 流程；`--sensor hj90` 直接采集米制扫描结果，不执行焊缝识别。两条链路尚未融合。

另有 [Nested 双目与相机参数推理](nested_stereo_depth.md)：关闭激光后，同步左右目 RGB 连同 K/R/t 输入 Nested，输出两目各自的稠密深度并独立核对误差。该入口尚未融合这里的激光扫描点云。

完整场景 File > Open 入口仍为 `assets/scenes/industrial-welding/industrial_welding_full.usda`，现在包含更强的 `industrial_welding_hj90.usda`：

- 左目 `/World/Robot/e05l_welding_torch/hj90_active/left`
- 右目 `/World/Robot/e05l_welding_torch/hj90_active/right`
- 激光 `/World/Robot/e05l_welding_torch/hj90_active/laser`
- 厂商已校正 RGB `/World/Robot/e05l_welding_torch/hj90_active/rgb`

原 `/.../hj90` 留作中点 RGB 兼容相机。`main.py` 默认改用厂商 RGB 通道，拍照时在会话层关闭激光。打开 USD 可看到零角度的投射，自动摆扫由 `scan_hj90.py` 驱动；不会仅凭打开文件自动执行机器人动作。所有原始 USDC、拍摄层和机器人初始位姿保持不变。更新仿真或 RGB 标定后运行 `scripts/build_hj90_sensor.py` 重建便携传感器层；导入脚本也会重建它。

投影使用 [NVIDIA 的 RTX projector light 接口](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html#projector-light)。采样前重置渲染历史，避免把上一条光纹拖影当作新条纹；参考 [Isaac Sim SDG 渲染历史说明](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html)。

## 输出和坐标

`outputs/hj90/<时间>/` 保存：

- `left_off.png`、`right_off.png`：静态双目背景；`left_gray.png`、`right_gray.png` 为灰度转换。
- `frames/*.npz`：每条扫描线的 `left_on`、`right_on` RGB；动态模式还保存逐帧 `left_off`、`right_off`。
- `frames.json`：角度、虚拟采样时间、当帧中点相机到世界的矩阵、点数及独立评价。
- `camera.json`：仿真标定和相机路径。
- `scan.npz`：三维点、左目亚像素坐标、蓝光强度评分、帧索引；静态模式另含 `depth_m` 与 `valid_depth_mask`。
- `cloud.ply`、`cloud.pcd`、`cloud.txt`：世界坐标的米制点云。
- `depth.png`：静态扫描稀疏深度，孔洞不插值、不填补成理想真值。
- `report.json`：结果与模型边界；`runtime.log`：渲染日志。
- 可选 `evaluation/*.npz`：RTX 左目光轴深度，只用于独立评价，RGB 重建程序不读取。

`points_mid_optical_m` 使用双目中点 OpenCV 坐标：+X 向右、+Y 向下、+Z 向前。`points_world_m` 使用场景米制 Z 向上坐标。静态深度在**中点相机**上栅格化，未观测位置为 NaN；它不同于左右目各自的深度。动态点云保留帧索引与当帧外参，不把不同时刻数据伪装成单幅静态深度。运动目标的点云可出现时间畸变。

## 本次验证

7 项 CPU 检查覆盖工作区三档距离、视差符号、激光关闭、遮挡、多重反射亮峰、空洞、90 mm 刚性基线、USD 本地依赖和采集限速。Isaac Sim 6.0.0.1 / RTX 5090 实际运行了平面、台阶/遮挡、A04 初始臂姿和 A02；未执行机器人动力学。

| 数据 | 线数 | 重建点数 | 深度 MAE | P95 |
|---|---:|---:|---:|---:|
| 平面中心光纹 `plane_probe_02` | 1 | 1100 | 1.85 mm | 2.74 mm |
| 台阶/遮挡 `step_scan_02` | 48 | 17082 | 1.57 mm | 3.74 mm |
| A04 初始臂姿 `A04_initial_arm_01` | 128 | 9532 | 3.84 mm | 8.92 mm |
| A02 `A02_01` | 128 | 81167 | 2.68 mm | 7.05 mm |
| 移动台阶固定激光 `dynamic_01` | 24 | 5393 | 0.34 mm | 1.34 mm |

台阶 `step_scan_02` 使用早期 4 mm 激光平面筛选容差；当前容差收紧为 0.5 mm。A04/A02 使用当前 0.5 mm。每项指标只评价最终保留点在**左目观测像素**的轴向深度与 RTX 真值，包含量化/边缘和反射误匹配，不是全画面精度，也不是焊缝定位精度。128 线为稀疏测试，不等同厂家全帧。A04 当前朝向有大量目标超出 800 mm 或落在自身上，9532 点不能解释为工件有效覆盖。

A04、A02 和动态测试已从保存的 RGB 离线重建，全部输出数组与在线结果完全一致，未读场景或评价深度。运行数据保留在 `outputs/hj90/`，由 Git 忽略。没有按真实厂家标定、真实工件或硬件采集进行精度认证。
