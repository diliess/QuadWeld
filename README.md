# QuadWeld：第一版视觉轨迹与机械臂控制

在 `plan.md` 第一阶段控制器上，将感知改为机身固定 RGB-D：采集对齐的 RGB 和深度 → 检测闭合黑色轮廓 → 深度点云 RANSAC 拟合板面 → 恢复三维轮廓 → 用安装标定变换到机械臂基座系 → 生成粗轨迹 → IK 预检查 → 接近、下降、走圈、抬起。**生成轨迹不读取 USD 的钢板位置、法向、尺寸或角点**，深度不可用时终止。本版是静态观测后执行，尚未实现移动中的在线重规划。

## 场景参数

所有位置和尺寸使用米；世界坐标系为 Z 轴向上。

| 对象 | 参数 |
| --- | --- |
| UR5e 固定底座 | 世界位置 `(0.45, -0.62, 0.45)`；下方有 `0.26 × 0.26 × 0.45` 支座；官方资产 `/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` |
| 钢板 | 中心 `(0.45, 0, 0.50)`；长 × 宽 × 厚 `0.80 × 0.60 × 0.01`；上表面 `z=0.505` |
| 黑色矩形 | 位于钢板中央；外尺寸 `0.30 × 0.20`；线宽 `0.008`；线条高出钢板 `0.0001` |
| 机身 RGB-D | 相对机械臂基座位置 `(0, 0.12, 0.55)`，向前下方观察；`640 × 480`，`fx=fy=400 px`；安装与内参见 `calibration/body_rgbd.json` |
| TCP 离板间隙 | `0.01`；工具 +Z 指向钢板，姿态固定 |
| 简化工具长度 | 法兰到 TCP 为 `0.12`，与离板间隙分别配置 |
| 路径 | 间距不超过 `0.01`；接近与抬起高度为钢板上方 `0.10` |
| 速度上限 | TCP 插值时间按 `0.04 m/s` 设置；关节 `0.5 rad/s`、`1.0 rad/s²` |

相机 prim 为 `/World/RobotMount/Sensors/BodyRGBD`，与机械臂共享安装座，模拟将来的机身安装关系；目前没有四足模型。上述工件参数仅用于生成测试场景。相机安装变换来自独立标定文件，仿真按该文件放置传感器，算法也以该文件作为输入。实机需要测量相机内参和相机到机械臂基座的外参，不能直接沿用示例数值。机身整体位置无需已知，只要相机与机械臂基座保持刚性连接，外参就不随狗的整体移动改变；但本版要求采集到执行结束期间相对工件保持静止。

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

此前的 RGB 版本已在独立端口验证串流启动、等待开始、终端触发及完整运动执行。RGB-D 版本沿用同一串流入口。Windows 客户端的视频接收和远程按钮操作仍需在实际网络中连接确认。

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

串流启动后连接运行机器的 IP，默认信令端口为 `49100`、视频端口为 `47998`。`stream_scene.py` 会打开刚生成的 USD 并把界面视角对准机械臂与钢板；机身 RGB-D 相机保存在场景中。默认启用的串流事件追踪会生成占空间的 `NvStreamer-*.etli` 文件；上面的命令关闭追踪，并将其他运行文件集中在 `outputs/webrtc/`。

若默认 Isaac Sim 资产根目录不可访问，先取得完整的 UR5e USD 资产及其依赖，再指定主 USD 文件：

```bash
python main.py --headless --robot-usd /absolute/path/to/ur5e.usd
```

成功后生成的文件如下。USD 仍引用 UR5e 资产；复制到其他机器时需保证资产可访问。

| 文件（`outputs/` 下） | 内容 |
| --- | --- |
| `body_rgb.png` | 在 home 姿态采集的机身 RGB，未加入调试线 |
| `body_depth_m.npy`、`body_depth_preview.png` | 对齐的光轴 Z 深度（米）和仅供查看的灰度预览 |
| `rgbd_calibration.json` | 本次使用的安装与内参标定副本 |
| `contour_mask.png`、`contour_overlay.png` | 阈值图、轮廓叠加；红点为起点，箭头标识方向 |
| `waypoints_base.csv`、`rgbd_trajectory.npz` | 粗轨迹：基座系 TCP 位置/位姿、相机系与基座系轮廓、运动阶段、估计板面 |
| `vision_report.json` | 深度平面拟合内点比例、残差、法向及路径点数 |
| `trajectory.npz` | 基座系位姿 `poses_base`、表面点、关节解、运动阶段及相机安装变换 |
| `robot_kinematics.npz` | 从当前 USD 提取的运动学模型，可独立在 CPU 上加载 |
| `tcp_execution.csv` | 每步实际与参考 TCP 基座系位置、位置/姿态误差 |
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

不启动 Isaac Sim，也可以在 CPU 上从文件生成同样的粗轨迹：

```bash
python generate_rgbd_trajectory.py \
  --rgb outputs/body_rgb.png \
  --depth outputs/body_depth_m.npy \
  --calibration outputs/rgbd_calibration.json \
  --output-dir outputs/rgbd_offline
```

实机输入为 RGB、对齐到该 RGB 图像坐标的深度 `.npy` 或单通道 PNG，以及标定 JSON。深度必须是光轴 Z 距离，不能是到光心的径向距离。米制浮点深度使用 `depth_scale_to_m=1`，毫米制 uint16 使用 `0.001`；预览 PNG 不能用于规划。内参必须匹配当前分辨率和畸变/校正状态。实机数据需要先由设备 SDK 完成 RGB/深度配准；此脚本不接入相机驱动。

`vision/contour_detector.py` 将 RGB 转成灰度并阈值化，使用轮廓层级寻找包围亮色内部的闭合暗色线条，拒绝触碰图像边界的区域。多目标相近时报告歧义；未检测到目标时终止，不使用已知矩形角点作为回退。第一版跟随可见线条的**外边缘**，不是线条中心。阈值、最低轮廓面积可在 `config.py` 调整。

轮廓从图像左上角附近开始，按图像顺时针排序。多边形简化后逐边采样，保留拐角和明确的闭合末点。细线边缘的像素锯齿可能产生额外短边，不强行把检测结果拟合成手工矩形。

`vision/rgbd.py` 只接受 RGB、深度和标定。将轮廓内部有效深度反投影为点云，RANSAC 去除离群点，再用 SVD 拟合平面。轮廓边缘与这个**观测估计的平面**求交，避免直接采用黑线边缘容易缺失的深度。默认至少 40% 的内部像素有有效深度，拟合内点比例至少 70%，内点距离阈值 3 mm；不满足则拒绝生成。相机坐标为 OpenCV 光学系：+X 向右、+Y 向下、+Z 向前。`p_base = base_from_camera @ p_camera`（齐次列向量）；法向朝向相机，工具 +Z 朝向板面。TCP 沿估计法向留出 10 mm 间隙。旧的 USD 已知平面投影模块已移除。

本版直接恢复三维路径及板面法向，不依赖板子绝对世界位姿，也没有建立带稳定编号的板坐标系。可见的 AprilTag/ArUco 标记目前不参与计算；后续需要跨次观测保持同一起点、板坐标系或关联 CAD 路径时，再加入标记识别及其到任务轮廓的标定。当前仍要求可见的高对比度闭合任务轮廓，只有定位标记并不足以推断任意焊缝。

`robot/kinematics.py` 仍从机器人 USD 提取六个关节的运动学结构；主流程将模型置于基座系后求 IK。官方 UR5e 缺少 `tool0`，场景构造时根据其夹具挂载坐标建立 `wrist_3_link/tool0/TCP`。IK 使用有界数值最小二乘，以上一个关节解为初值，同时约束位置和姿态。世界变换只用于仿真显示和实际 link 位姿回读适配。实机应替换为机器人模型、编码器/FK 和控制接口；本次移除的是**工件位置特权信息**，并非把整个仿真控制适配器也替换为硬件驱动。

`planning/waypoint_generator.py` 负责几何路径；`robot/robot_controller.py` 提供 `move_to_pose` 和 `follow_cartesian_path`。预先求解整条路径，拒绝不可达目标和明显关节跳变。执行采用五次时间插值的关节位置目标，限制速度和加速度；每个采样点到达后继续，矩形拐角停速，不做切角平滑。相邻关节插值段近似 Cartesian 直线，第一版不保证严格恒定焊接速度。末点与首点闭合后沿法向抬起到接近高度。

只有初始化 home 状态时调用 `set_dof_positions`；任务运动全部调用 `set_dof_position_targets`，通过实际 link 位姿检查到位、超时和跟踪发散。任何失败都会停止继续下发路径并保存已有执行记录。简化工具只有视觉模型，无真实焊接过程。

## 运动调试与限制

- 没找到轮廓：查看 RGB 和遮挡，调整 `dark_threshold`；不要用矩形配置代替视觉结果。
- 反投影位置不对：检查 RGB/深度配准、光轴 Z 定义、深度单位、分辨率和外参方向；查看 `vision_report.json`。
- IK 无解或关节跳变：检查 home 姿态、机器人与钢板距离、工具长度和工具 +Z 方向；整个预检查通过前不开始走圈。
- 运动超时或抖动：查看执行 CSV 的误差，检查驱动参数、碰撞及关节速度；不要通过瞬移关节掩盖失败。
- 绿色路径、蓝色 TCP 路径和橙色实际轨迹应在俯视方向基本重合，蓝/橙轨迹比钢板高 10 mm。

预检查对机械臂连杆中心线及工具做观测板面的近似干涉检查：将同一平面的有效深度点凸包下方视为实体，不读取板子尺寸或地面高度。该检查不能覆盖未观察到的障碍、地面、自碰撞，也不是完整网格碰撞规划器。当前范围是静态、近似平面、单个高对比度闭合轮廓，不包含四足运动、真实焊接、定位标记检测、RL 训练和实时视觉伺服。`rgbd_trajectory.npz` 的 `poses_base` 可作为以后残差策略的参考轨迹，但策略与训练环境尚未实现。

CPU 测试覆盖图像平移、深度高度变化、倾斜板面、噪声/缺失/离群点、毫米深度单位、外参变换和无有效观测时拒绝规划，以及原有 IK、控制失败停止、串流开始门控。仿真使用理想渲染深度；单元测试加入了合成噪声，尚未验证真实 RGB-D 在反光金属上的数据质量。执行报告的跟踪误差是相对视觉生成路径的控制误差；离板间隙也是相对估计平面，不代表真实工件测量精度。

本机 RGB-D 版本已通过 17 项 CPU 测试，并在 Isaac Sim 6 / RTX 3090（`--gpu 4`）完成 179 个运动目标、5,774 个执行采样，走圈后成功抬起。走圈相对视觉路径的 RMS 跟踪误差约 0.53 mm，最大约 0.84 mm；相对估计板面的间隙为 9.18–9.76 mm。验证结果在 `outputs/rgbd_validation/`。同一 RGB-D 经独立 CPU 入口生成的 124 个粗轨迹位姿与主流程逐项一致；179 个运动目标还包括控制器补充的 home 到接近点的运动。

若图片没有钢板，检查机身相机视野、机械臂遮挡及裁剪范围。若默认资产加载失败，检查 Isaac Sim 资产根目录或使用 `--robot-usd`。若报 `No device could be created`、`NVML_ERROR_DRIVER_NOT_LOADED`，需要在能访问 NVIDIA 驱动和 GPU 的环境运行；无界面模式同样需要 GPU 渲染 RGB-D。离线生成器不需要 GPU。

代码面向 Isaac Sim 6，使用 `RtxCamera` 和 `CameraSensor` 的 `rgb`、`distance_to_image_plane` 获取同一渲染产品的 RGB-D。相机接口分辨率顺序为 `(高, 宽)`；标定文件使用 `(宽, 高)`。仿真传感器当前要求中心主点、零畸变、米制深度；离线算法支持标定文件中的 OpenCV 畸变和深度比例。场景/控制参数在 `config.py`，感知阈值默认值在 `vision/rgbd.py`，相机标定在 `calibration/body_rgbd.json`。

Isaac Sim 首次运行会提示接受 NVIDIA Omniverse 许可协议；请阅读提示并自行决定是否接受。许可协议是运行 Isaac Sim 的前提。

参考：[Isaac Sim 6 UR5e 资产](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html)、[Isaac Sim 6 Python 运行方式](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)、[Isaac Sim 6 相机 API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html)。
