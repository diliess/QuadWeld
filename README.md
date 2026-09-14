# QuadWeld：第一版视觉轨迹与机械臂控制

按 `plan.md` 实现固定 UR5e 的第一阶段：拍摄钢板 → OpenCV 提取闭合黑色轮廓 → 相机射线与钢板平面求交 → 每约 10 mm 生成路径点 → IK 预检查 → 接近、下降、沿轮廓走一圈、抬起。默认运行完整流程，有界面时完成后保留窗口。路径来自相机图像，场景中的矩形尺寸和角点不参与轨迹构造。

## 场景参数

所有位置和尺寸使用米；世界坐标系为 Z 轴向上。

| 对象 | 参数 |
| --- | --- |
| UR5e 固定底座 | 世界位置 `(0.45, -0.62, 0.45)`；下方有 `0.26 × 0.26 × 0.45` 支座；官方资产 `/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` |
| 钢板 | 中心 `(0.45, 0, 0.50)`；长 × 宽 × 厚 `0.80 × 0.60 × 0.01`；上表面 `z=0.505` |
| 黑色矩形 | 位于钢板中央；外尺寸 `0.30 × 0.20`；线宽 `0.008`；线条高出钢板 `0.0001` |
| RGB 相机 | 世界位置 `(0.45, 0, 1.75)`；正对钢板；`640 × 480`；焦距 `0.024`；水平光圈 `0.020955` |
| TCP 离板间隙 | `0.01`；工具 +Z 指向钢板，姿态固定 |
| 简化工具长度 | 法兰到 TCP 为 `0.12`，与离板间隙分别配置 |
| 路径 | 间距不超过 `0.01`；接近与抬起高度为钢板上方 `0.10` |
| 速度上限 | TCP 插值时间按 `0.04 m/s` 设置；关节 `0.5 rad/s`、`1.0 rad/s²` |

机械臂位于钢板的 `-Y` 侧，底座轴线到钢板近边缘有 `0.32 m` 的水平距离，支座边缘到钢板还有 `0.19 m` 的间隙。相机到钢板约 `1.245 m`，在该焦距与光圈下，钢板平面的可见范围约为 `1.09 × 0.81 m`，因此 `0.80 × 0.60 m` 的钢板四周都有余量。Isaac Sim 6 的 `RtxCamera` 使用 USD 相机坐标约定：局部 `-Z` 向前、局部 `+Y` 向上。相机姿态为单位四元数，正对下方钢板；图像向上对应世界 `+Y`，图像向右对应世界 `+X`。`/World/Workpiece/PlateFrame` 的原点位于钢板上表面中央，轴向与世界一致。

## 运行

需要支持 RTX 光线追踪的 NVIDIA GPU、可用驱动，以及 **Isaac Sim 6.0**。本机使用全小写的 `quadweld` Conda 环境（Python 3.12）。首次安装可按 [NVIDIA 的 Python 环境说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html) 执行：

```bash
conda create -n quadweld python=3.12 pip -y
conda activate quadweld
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==6.0.0.1' --extra-index-url https://pypi.nvidia.com
```

进入项目目录后，有界面运行：

```bash
conda activate quadweld
cd /data/huangliqi/QuadWeld
python main.py
```

无界面执行完整一圈并退出：

```bash
conda activate quadweld
cd /data/huangliqi/QuadWeld
python main.py --headless
```

用 `--gpu 4` 等参数指定一张 GPU（默认 GPU 0，关闭多 GPU 渲染）。只检查视觉与整条 IK 路径、暂不执行任务运动：

```bash
python main.py --headless --plan-only
```

`--plan-only` 会将机器人初始化到拍照用的 home 姿态。使用 `--scene-only` 保留仅建场景、拍照的流程。
除 Isaac Sim 外，算法使用 NumPy、SciPy、OpenCV 和 Pillow；可用以下命令检查当前环境是否已包含它们：

```bash
python -c "import numpy, scipy, cv2, PIL"
python -m unittest discover -s tests -v
```

若缺少算法依赖，可在相同环境中安装 `numpy scipy opencv-python-headless pillow`。不要在已装另一种 OpenCV Python 包的环境中重复安装多个 OpenCV 变体。

在 Windows WebRTC 客户端实时观看完整任务，服务器运行：

```bash
conda activate quadweld
cd /data/huangliqi/QuadWeld
python main.py --webrtc --gpu 4
```

`--webrtc` 在运行算法的 Python 进程中开启无本地窗口的串流，同时保留传输画面中的 UI。等待终端出现 `WebRTC ready`，在 Windows 客户端填写 Linux 服务器 IP 并连接。看到画面后，在 **QuadWeld Task** 面板点击 **Start task**（也可在运行脚本的 SSH 终端按回车）。随后同一个进程完成拍照、轮廓检测、规划、接近、下降、走圈和抬起；面板显示当前阶段。等待启动和 IK 预检查期间仍更新画面。

任务结束后保持串流，便于观察结果；可从界面 File → Exit 退出，或在服务器终端按 Ctrl+C。`--auto-start` 可跳过开始按钮，`--exit-on-complete` 可在任务结束后退出，适合自动验证。

已在独立测试端口验证串流进程启动、等待开始、终端触发及完整 181 个运动目标执行，报告位于 `outputs/webrtc_validation/execution_report.json`；9 项 CPU 测试通过。Windows 客户端的视频接收和远程按钮操作需在实际网络中连接确认。

需要指定客户端可达的入口 IP 时：

```bash
python main.py --webrtc --gpu 4 --public-ip 192.168.1.100
```

将示例 IP 换成服务器内网/VPN IP 或实际映射入口 IP。默认放行 `49100/TCP`（信令）和 `47998/UDP`（视频）；SSH `-L` 只能转发 TCP，不能单独承载视频。支持 `--signal-port`、`--stream-port` 指定端口，但客户端与网络映射也需匹配。不要同时启动使用相同端口的 `main.py --webrtc` 和独立串流进程。

如果只想打开已经保存的 USD 场景，可使用以下独立串流命令；它不会执行或回放轨迹算法：

```bash
conda activate quadweld
cd /data/huangliqi/QuadWeld
mkdir -p outputs/webrtc
cd outputs/webrtc
isaacsim isaacsim.exp.full.streaming --no-window \
  --/exts/omni.kit.livestream.app/primaryStream/enableEventTracing=false \
  --exec /data/huangliqi/QuadWeld/stream_scene.py
```

串流启动后连接运行机器的 IP，默认信令端口为 `49100`、视频端口为 `47998`。`stream_scene.py` 会打开刚生成的 USD 并把界面视角对准机械臂与钢板；俯视 RGB 相机仍保存在场景中。默认启用的串流事件追踪会生成占空间的 `NvStreamer-*.etli` 文件；上面的命令关闭追踪，并将其他运行文件集中在 `outputs/webrtc/`。

若默认 Isaac Sim 资产根目录不可访问，先取得完整的 UR5e USD 资产及其依赖，再指定主 USD 文件：

```bash
python main.py --headless --robot-usd /absolute/path/to/ur5e.usd
```

成功后生成的文件如下。USD 仍引用 UR5e 资产；复制到其他机器时需保证资产可访问。

| 文件（`outputs/` 下） | 内容 |
| --- | --- |
| `overhead_rgb.png` | 在 home 姿态采集的原始 RGB，未加入调试线 |
| `contour_mask.png`、`contour_overlay.png` | 阈值图、轮廓叠加；红点为起点，箭头标识方向 |
| `waypoints_world.csv` | TCP 世界位置，包含接近、下降、闭合轮廓和抬起 |
| `trajectory.npz` | 图像点、表面点、4×4 TCP 位姿、关节解、运动阶段和坐标变换 |
| `robot_kinematics.npz` | 从当前 USD 提取的运动学模型，可独立在 CPU 上加载 |
| `tcp_execution.csv` | 每步实际与参考 TCP 位置、位置/姿态误差 |
| `execution_report.json` | 是否完成、跟踪误差统计；失败不会标记完成 |
| `trajectory_validation.png` | 用下面的绘图命令生成：俯视轨迹对照与离板高度曲线 |
| `quadweld_scene.usda` | 场景与规划调试图；绿线为表面轮廓，蓝线为 TCP 参考路径 |
| `quadweld_result.usda` | 执行后额外保存橙色实际 TCP 轨迹 |

USD 保存的是场景及轨迹可视化，不包含 Python 控制器；打开 USD 或点击播放不会自动重跑任务。请运行 `main.py` 执行控制。关闭 GUI 窗口可退出。

运行后可在 CPU 上绘制实际轨迹（需 Matplotlib）：

```bash
python plot_execution.py
```

## 算法与坐标约定

`vision/contour_detector.py` 将 RGB 转成灰度并阈值化，使用轮廓层级寻找包围亮色内部的闭合暗色线条，拒绝触碰图像边界的区域。多目标相近时报告歧义；未检测到目标时终止，不使用已知矩形角点作为回退。第一版跟随可见线条的**外边缘**，不是线条中心。阈值、最低轮廓面积可在 `config.py` 调整。

轮廓从图像左上角附近开始，按图像顺时针排序；俯视图中对应世界坐标的左上 → 右上 → 右下 → 左下。多边形简化后逐边采样，保留拐角和明确的闭合末点。细线边缘的像素锯齿可能产生额外短边，不强行把检测结果拟合成手工矩形。

`vision/geometry.py` 从实际 USD Camera 的焦距、光圈、变换及 `PlateFrame` 读取标定。采用相机内参 + 射线平面求交，因为该 MVP 已知钢板平面，且无需额外标定四个对应角点。所有 4×4 数组采用列向量约定，`T_world_camera` 将相机点变换到世界。USD 相机射线为 `[(u-cx)/fx, -(v-cy)/fy, -1]`；`u` 向右、`v` 向下。求交结果位于钢板上表面，加 `0.01 × 表面法向` 得到 TCP 目标。

`robot/kinematics.py` 从配置的六个 USD 旋转关节读取父/子关节坐标系、轴和限位，建立 `T_world_base × T_base_tool × T_tool_TCP`。官方 UR5e 缺少 `tool0`，场景构造时根据其夹具挂载坐标建立 `wrist_3_link/tool0/TCP`。IK 使用有界数值最小二乘，以上一个关节解作为初值，同时约束位置和姿态。初始化后用物理引擎的 link 位姿检查 FK 一致性。

`planning/waypoint_generator.py` 负责几何路径；`robot/robot_controller.py` 提供 `move_to_pose` 和 `follow_cartesian_path`。预先求解整条路径，拒绝不可达目标和明显关节跳变。执行采用五次时间插值的关节位置目标，限制速度和加速度；每个采样点到达后继续，矩形拐角停速，不做切角平滑。相邻关节插值段近似 Cartesian 直线，第一版不保证严格恒定焊接速度。末点与首点闭合后沿法向抬起到接近高度。

只有初始化 home 状态时调用 `set_dof_positions`；任务运动全部调用 `set_dof_position_targets`，通过实际 link 位姿检查到位、超时和跟踪发散。任何失败都会停止继续下发路径并保存已有执行记录。简化工具只有视觉模型，无真实焊接过程。

## 运动调试与限制

- 没找到轮廓：查看 RGB 和遮挡，调整 `dark_threshold`；不要用矩形配置代替视觉结果。
- 反投影位置不对：检查 USD 相机方向、光圈比例、`PlateFrame`；当前采用无畸变针孔相机和已知平面。
- IK 无解或关节跳变：检查 home 姿态、机器人与钢板距离、工具长度和工具 +Z 方向；整个预检查通过前不开始走圈。
- 运动超时或抖动：查看执行 CSV 的误差，检查驱动参数、碰撞及关节速度；不要通过瞬移关节掩盖失败。
- 绿色路径、蓝色 TCP 路径和橙色实际轨迹应在俯视方向基本重合，蓝/橙轨迹比钢板高 10 mm。

预检查对机械臂连杆中心线及工具做钢板/地面干涉检查；它不是完整的网格碰撞、自碰撞或障碍物规划器。当前范围仍是固定底座、固定水平钢板、单个高对比度闭合轮廓，不包含四足、真实焊接和实时视觉伺服。更换机械臂需要适配其关节链、工具挂载、home 姿态与驱动参数。

本机已使用 Isaac Sim 6、RTX 3090（`--gpu 4`）完成整圈运行：107 个闭合轮廓采样点、181 个运动目标、5,828 个执行采样，完成后抬起。走圈阶段相对图像生成路径的 RMS 距离误差约 0.53 mm、最大约 0.84 mm；TCP 离板高度为 9.17–9.76 mm。该统计表示仿真控制跟踪误差，不代表真实焊接精度或视觉标定精度。7 项 CPU 测试通过，覆盖视觉变化、相机反投影、闭合路径、IK 不可达和失败后停止执行。

若图片没有钢板，先检查相机 prim `/World/Sensors/OverheadRGB` 的位置、旋转和裁剪范围，以及 `config.py` 中的钢板位置。若默认资产加载失败，检查 Isaac Sim 资产根目录或使用 `--robot-usd`。若报 `No device could be created`、`NVML_ERROR_DRIVER_NOT_LOADED`，需要在能访问 NVIDIA 驱动和 GPU 的环境运行；无界面模式同样需要 GPU 才能渲染 RGB 图像。

代码面向 Isaac Sim 6，使用 `isaacsim.sensors.experimental.rtx.RtxCamera` 和 `CameraSensor` 获取 RGB，使用 `SimulationManager` 管理仿真，避免 6.0 已弃用的 `isaacsim.sensors.camera` 与 `isaacsim.core.api`。相机运行接口的分辨率顺序为 `(高, 宽)`；配置仍写作 `(宽, 高)`。参数均在 `config.py`；以后切换机械臂时改资产路径和机器人坐标系参数即可。

Isaac Sim 首次运行会提示接受 NVIDIA Omniverse 许可协议；请阅读提示并自行决定是否接受。许可协议是运行 Isaac Sim 的前提。

参考：[Isaac Sim 6 UR5e 资产](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html)、[Isaac Sim 6 Python 运行方式](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)、[Isaac Sim 6 相机 API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html)。
