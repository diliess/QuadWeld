# 工业焊接厂房：B2 + E05L-Pro + HJ-90

2026-11-12 同步 RoboSimStudio 合并提交 `a14ad12`。当前为 **32×40 m 厂房、A 区 28 工位（4 排 7 列）、B 区 4 块板架**。A25/A26 含双面 T 型首层焊道及焊纹；A27/A28 含厚板 V 型对接首层焊道及焊纹。其余原试件保留。所有几何和贴图均为本地资产，不依赖 RoboSimStudio 环境。

| 入口 | 内容 |
|---|---|
| `industrial_welding_full.usda` | 完整场景、机器人和 QuadWeld 双目/激光/RGB 通道，Isaac File > Open 入口 |
| `industrial_welding_hj90.usda` | 叠加传感器的覆盖层，引用当前源 USDC |
| `industrial_welding.usda` | 停用机器人、相机、灯光与 PhysicsScene 的纯环境引用层 |
| `industrial_welding.usdc` | 源场景底层，逐字节保留源文件 |

## 打开与默认姿态

在 Isaac Sim 中通过 **File > Open** 打开 `industrial_welding_full.usda`。视口选择 `/World/Cameras/overview` 看全景，`robot` 看机器人；`first_layer_T_back_macro` 和 `first_layer_butt_macro` 看焊纹。

默认使用源场景的正常初始站姿：基座约 `[-14,0,0.522968]` m，机械臂 `[0,1.2,2.3,0,1.3,0]` rad。相机保持随焊枪刚性安装。该初姿尚未对准工件，不能用初始相机画面评价焊缝识别效果。

旧 24 工位布局的 A02/A04 四个拍摄预设已退出当前场景，完整旧包备份在本机 `outputs/imports/industrial-welding_20261112_174825_211229/`（Git 忽略）。旧覆盖层不能直接套用新底层。当前 `manifest.json` 的 `full_scene.poses` 仅列有效预设；`main.py`、双目采集和扫描入口使用其中的默认姿态。A04 机械臂多角度脚本在缺少新 A04 站位时会明确报错，需要重新规划站位后再运行。A15/A03/A21 的站位规划脚本读取当前工件几何，旧报告和旧误差仍只属于旧场景。

## 相机配置

- 中点兼容视点：`/World/Robot/e05l_welding_torch/hj90`。
- 左右目：`/World/Robot/e05l_welding_torch/hj90_active/left`、`right`，90 mm 基线，640×1100 仿真采样。
- 中央 RTX 激光：`hj90_active/laser`，450 nm 的 RGB 近似及可配置扫描参数。
- 独立 RGB：`hj90_active/rgb`，保留 Jh26B196 的实测 K/T；3120×4096，或同比缩小为 780×1024。

相机载荷仍为 1.056 kg；额定测量区 0.3–0.8 m，渲染裁剪为 0.005–100 m。左右目内参、激光外参与辐射参数仍为仿真假设，未声明实机计量精度。见 [双目激光说明](../../../docs/hj90_active_stereo.md) 与 [资料对齐说明](../../../docs/hj90_material_alignment.md)。

## 拍照与更新

在项目根目录仅验证当前默认 RGB 采集：

```bash
.venv/bin/python main.py --headless --capture-only --rgb-resolution 780 1024
```

识别流程仍是 RGB → DA3 → SAM3 → 几何推断，模型不接收场景焊缝真值或渲染器深度。工件识别测试须先选定针对当前工位的新视角。

```bash
.venv/bin/python scripts/import_industrial_welding.py \
  --source ../robosimstudio/examples/live/twins/industrial_welding
```

也可按项目依赖使用 `uv run --locked --extra usd python` 执行导入。导入先在临时目录验证源哈希、当前预设和完整传感器层的本地依赖，通过后备份旧包再替换。没有当前拍摄预设时回到源初姿，历史目录不参与导入。`source/` 保存源配置、USD 摘要和静态验证；`metadata/` 保存焊缝侧录及材质清单，仅供场景信息/评价，不输入感知模型。许可保留在 `licenses/`。

源场景静态参数与几何已校核；新工位作业站位的 IK、连续碰撞、步态和动力学不在此次同步验证范围内。

## 本次验证

60 项 CPU 测试通过，包含新布局、默认姿态和导入失败时保留原包的回归检查。QuadWeld 自带 Isaac Sim 6.0.0.1 实际加载通过：24 个刚体、18 个转动关节、1685 个碰撞体；正常初姿、左右目 90 mm 基线、RGB 内参与刚性外参一致。全景、机器人、T 型及对接焊纹、左右目与 RGB 均已渲染并打开检查，激光开关产生实际投射响应。日志和截图保留在 `outputs/imports/scene_sync_20261112/`。验证保持时间轴暂停，不代表新工位的焊缝识别精度或作业可行性。
