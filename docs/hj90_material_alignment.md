# HJ-90 相机资料与仿真设计对齐

2026-11-12。核对用户提供的 `../document/伟景相机资料/`：HJ-90 宣传册 V2.2 / 2026.10、DetectLaser 3.6.20.119 的公开 SDK 头文件和示例，以及 `intrinsics.vendor.json`。当前场景采用已知的机身规格、双目与中央激光结构，并新增独立的厂商标定 RGB 光学通道。

## 采用的资料

| 信息 | 资料中的依据 | 当前设计 |
|---|---|---|
| 主动双目、中央激光及摆动机构 | 宣传册结构示意与工作模式 | 左右目独立成像，90 mm 基线，中央 RTX 线激光；静态模式左右摆扫，动态模式固定角度并采样相对运动 |
| 机身 120×92×86.5 mm，1056 g | 宣传册产品规格与尺寸图 | 保留 1.056 kg 刚性负载与包络；相机局部 XYZ 分别为 120 / 86.5 / 92 mm；支架仍为场景定制 |
| 300–800 mm 工作距离 | 宣传册规格 | 激光重建有效深度筛选 0.3–0.8 m；RGB 渲染可以看到更远，但不代表厂商保证计量精度 |
| 450 nm 标配；5000 线/s；全帧最大 6 FPS | 宣传册规格 | 保留蓝光辐射近似与虚拟采样时间约束；不把离线渲染速度当作实机采集速度 |
| 左目 3D 坐标到 RGB 的变换 | `SDK/VzNLSDK/Inc/VZNL_RGBConfig.h` 的 `VZNL_GetRGBCalibParam` | RGB 使用独立内参与外参；变换方向由 SDK 注释确定 |
| 独立 RGB 通道 | 同一头文件的 `VzNL_GetRGBImage`；`vizumsdk_save_color_image_and_depth_map.cpp` 注明第三目图像示例 | 在已有机身上增加 RGB 光学通道，不凭空增画新的机壳开孔 |
| RGB 原生分辨率、内参及外参 | `intrinsics.vendor.json`，序列号 `Jh26B196` | 原样保存标定文件，USD 与拍照入口实际采用这些参数 |

SDK 是通用包；提供的 JSON 未给出可独立核对的设备型号绑定证明。因此这是按所提供资料配置的 `Jh26B196` RGB 通道，仍需实机确认其与 HJ-90 的具体选型关系。双目基线由 HJ-90 宣传册给定，**这份 RGB JSON 不是一组实测左右目标定**。

SDK 的两份说明文档虽然扩展名为 `.pdf`，内容带 `%TSD-Header`，不能用普通 PDF 工具读取。本次接口解释取自可读的头文件和源码示例，没有推测这些受保护文件的内容。

## RGB 参数与坐标

已将原始 JSON 逐字节保存为 `calibration/hj90_rgb_vendor.json`，便携场景另带 `source/hj90_rgb_vendor.json`。SHA256 为 `28423b34f9ab8cf2bc036e4f70e359ef0e51d4fb44e3a0d1e11a486da7348ea5`。

厂商 RGB 图像宽×高为 **3120×4096**。原始内参为：

```text
K_vendor = [[3142.534912109375, 0, 1556.43310546875],
            [0, 3142.534912109375, 2117.60546875],
            [0, 0, 1]]
```

主点不是画幅中心。USD 通过 `horizontalApertureOffset` 和 `verticalApertureOffset` 保留它，读取内参时也计算这两个偏移。不得简单替换成 `(width/2, height/2)`。

SDK 说明 `T_optical_camera` 把左目点转换到 RGB 光学坐标：

```text
P_rgb_m = R_vendor @ P_left_m + t_vendor_mm / 1000
T_midpoint_from_rgb = Translate(-baseline/2, 0, 0) @ inverse(T_rgb_from_left)
```

只把平移从毫米转换为米，保留原始旋转。该矩阵不是 RGB 相机在世界坐标系中的位姿，也不是左右目之间的变换。它与 USD 焊枪安装变换组合后，RGB 通道随机械臂刚性运动。

标定文件以 OpenCV 整数像素中心约定解释；原文件和 `camera_matrix_opencv` 保留该约定。QuadWeld 现有几何程序与 USD 使用图像边界为原点、像素中心为 `index+0.5`，因此内部 `camera_matrix` 的 `cx/cy` 加 0.5。实际缩放先缩放边界坐标内参，再减 0.5 得到对应的 OpenCV 内参，例如 `cx'=(cx+0.5)*scale_x-0.5`。不会把两种坐标约定混用。

JSON 的零畸变只适用于它描述的**已校正 RGB 图像**。它引用的 `rectified-to-raw.uint16.bin` 没有随资料提供，当前记录 `rectification_map_supplied=false`，不支持以零畸变参数处理未经校正的实机原图。原文件记录的投影一致性检查也不是独立米制精度验证；实机仍需标定板重投影核对。

## 场景与使用

完整场景入口为 `assets/scenes/industrial-welding/industrial_welding_full.usda`。

| 通道 | USD 路径末尾 | 参数来源 |
|---|---|---|
| 左目、右目 | `hj90_active/left`、`hj90_active/right` | 90 mm 基线来自规格；640×1100 与内参仍为仿真采样、按 800 mm 测量视场估算 |
| RGB | `hj90_active/rgb` | 本次提供的 3120×4096 已校正 RGB 标定 |
| 激光 | `hj90_active/laser` | 中央结构与波长来自规格；精确激光外参、光强与摆角仍为仿真假设 |
| 旧中点视角 | `hj90` | 保留此前仿真相机，供历史评测复现 |

路径共同前缀为 `/World/Robot/e05l_welding_torch/`。RGB 标定原生视场与宣传册的三维扫描视场属于不同通道，不强行把两者设为同一组内参。机身外形包络、质量、原始场景与机械臂初始关节角继续采用已校核的规格；没有厂家 CAD、重心/惯量、左右目与激光标定的部分保留明确的仿真假设。

```bash
# 重建包含 RGB / 双目 / 激光的便携覆盖层与哈希清单
.venv/bin/python -m weld_scene.build_sensor

# 当前双目焊缝扫查（左右目使用独立仿真标定）
.venv/bin/python main.py --station A25 --headless

# 独立的双目激光仿真工具
.venv/bin/python -m weld_scene.active_scan --pose original --headless
```

原单张 RGB 拍照入口已移除。下文拍照记录和输出字段属于历史验证；厂商 RGB 标定与便携 USD 相机仍保留。当前双目采集的 `input.json` 保存左右目内外参，其标定独立于厂商 RGB 通道。

## 验证

57 项 CPU 测试通过；其中新增 7 项检查 RGB 标定值、单位和坐标方向、缩放主点、非法输入拒绝、USD 投影、随臂运动与拍照入口。便携 USD 的全部依赖与文件哈希检查通过。

Isaac Sim 6.0.0.1 实际完成 `780×1024` 和原生 `3120×4096` 两次 RGB 拍照，输出分别在 `outputs/validation/20261112_01__rgb_calibration__HJ90__vendor_capture/` 与 `outputs/validation/20261112_02__rgb_calibration__HJ90__native_capture/`。两次均成功获得非恒定图像，激光关闭，未执行物理运动；本次未重跑 DA3/SAM3 或进行速度基准评价。该检查确认参数配置和渲染链路有效，不代表硬件计量精度认证。
