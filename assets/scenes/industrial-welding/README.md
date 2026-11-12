# 工业焊接厂房：B2 + E05L-Pro + HJ-90

来自 RoboSimStudio 的 `industrial_welding` 场景，源提交为 `e226fa4`。本目录已包含所有几何、三个相对路径 PBR 贴图和拍摄姿态，不需要运行 RoboSimStudio，也不需要其 Python 环境。可使用已有的 Isaac Sim 打开完整场景 USD。

厂房为 32×32 m，保留 A 区 24 个工位、B 区 4 个大板架、工件、夹具和焊缝显示。米制、Z 向上，世界坐标与源工程一致，地面上表面 Z=0。

| 入口 | 内容 | 用途 |
|---|---|---|
| `industrial_welding_full.usda` | 完整场景、灯光、相机及 B2/E05L-Pro/HJ-90 | Isaac File > Open；可在此层保存编辑 |
| `industrial_welding.usda` | 厂房、工件、材质和 1087 个静态环境碰撞体 | 供其他 USD 工程引用的纯环境层 |
| `industrial_welding.usdc` | 原始完整底层，逐字节保留源文件 | 原始站位或其他覆盖层的共同底层 |
| `industrial_welding_*_photo.usda` | 四个独立拍摄姿态覆盖层 | A02、A04 正面、长边和初始臂姿 |

完整场景默认采用最后确认的 A04 长边站位：基座 `[3.65,3.20,0.396486]` m、yaw `-2.928654` rad，机械臂 `[0,1.2,2.3,0,1.3,0]` rad。HJ-90 刚性安装在焊枪上，质量 1.056 kg，名义尺寸 120×92×86.5 mm。左右目为 90 mm 基线、640×1100 的仿真采样；独立 RGB 光学通道采用提供的 3120×4096 已校正标定。300–800 mm 为标称计量区，渲染裁剪为 0.005–100 m。

## 打开完整场景

在已有的 Isaac Sim 中通过 File > Open 打开本目录的 `industrial_welding_full.usda`。
默认采用 A04 长边站位和预设初始臂姿。其他姿态可分别打开对应的 `*_photo.usda`。

厂商标定 RGB 通道为 `/World/Robot/e05l_welding_torch/hj90_active/rgb`；分辨率应设为 `3120×4096`，或同比缩小为 `780×1024`。
原中点相机 `/World/Robot/e05l_welding_torch/hj90` 作为兼容入口保留；完整场景包含 90 mm 基线左右目和中央 RTX 激光投射。
自动摆扫、RGB 三角重建和仿真精度范围见 [HJ-90 双目激光说明](../../../docs/hj90_active_stereo.md)。
RGB 标定与双目假设的区分见 [资料对齐说明](../../../docs/hj90_material_alignment.md)。仍缺少左右目及激光实测标定，未模拟真实噪声、翻盖动作或厂商计量精度。

## 从正常初始臂姿拍照识别

项目根目录的 [`main.py`](../../../main.py) 默认使用 `industrial_welding_full.usda` 中的厂商标定 RGB 通道，
保留 A04 长边站位和机械臂 `[0,1.2,2.3,0,1.3,0]` rad 的正常初始姿态。
相机随焊枪自然朝向，不采用另外两个 A04 对准内角的拍摄臂姿。

```bash
.venv/bin/python main.py --headless \
  --da3-python /path/to/envs/depth-anything-3/bin/python \
  --sam-python /path/to/envs/sam3/bin/python
```

脚本只采集 RGB，退出渲染进程后调用 DA3/SAM 自动识别。运行方式和输出见
[项目说明](../../../README.md)。

## 用作纯环境资产

`industrial_welding.usda` 显式停用原机器人、相机、灯光和 PhysicsScene，保留工件和静态碰撞，
可由其他 USD 工程直接引用。`industrial_welding_full.usda` 和拍摄姿态层保留完整机器人及 HJ-90。

从场景获得的 RGB 图片可以送入完整识别流程，深度由 DA3 预测：

```bash
conda activate depth-anything-3
python scripts/auto_weld.py /path/to/captured_rgb.png
```

## 来源与更新

`manifest.json` 保存源提交、底层 SHA256、各入口和文件哈希。`source/` 保留源配置、USD 摘要、相机参数校核和用户提供的规格资料；`hj90_rgb_vendor.json` 是原始 RGB 标定副本，RGB 拍照入口把对应内参传给识别流程。工件真值和额定精度不传给感知算法。原工程的源码、焊缝任务 JSON、导航地图与候选站位仍由 RoboSimStudio 工程管理。

重新导入时，在项目根目录运行：

```bash
uv run --locked --extra usd python scripts/import_industrial_welding.py \
  --source ../robosimstudio/examples/live/twins/industrial_welding
```

默认更新本目录的底层、贴图和预览层，并重新生成完整场景与纯环境入口；更新前应另存这些文件中的手工编辑。许可和第三方通知保留在 `licenses/`。

历史验证记录（2026-11-07）：已核对目录搬迁、本地贴图依赖、机器人/相机隔离和默认初姿。QuadWeld 的 Isaac Sim 6.0.0.1 实际加载完整场景并采集 HJ-90 RGB/深度，24 个刚体、18 个转动关节、1155 个碰撞体一致；深度及工作区掩码与源 Isaac Sim 6.2 的 A04 初始臂姿结果逐像素相同，中心约 815.7 mm。整机观察视角及 WebRTC 启动记录见 `outputs/runs/industrial_welding_context/`；尚未验证远端客户端视频接收或动态执行。
