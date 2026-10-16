# QuadWeld：第一版视觉轨迹与机械臂控制

在 `plan.md` 第一阶段控制器上，将感知改为机身固定 RGB-D：采集对齐的 RGB 和深度 → 检测闭合轮廓或开放线条 → 深度点云 RANSAC 拟合板面 → 恢复三维路径 → 用安装标定变换到机械臂基座系 → 生成粗轨迹 → IK 预检查 → 接近、下降、沿线运动、抬起。**生成轨迹不读取 USD 的钢板位置、法向、尺寸或角点**，深度不可用时终止。本版是静态观测后执行，尚未实现移动中的在线重规划。

## 拼板与开口盒子

现在可直接在 Isaac Sim 中创建两种工件：`--workpiece plate-grid` 为多个独立方板组成的拼板，`--workpiece open-box` 为底板和四面侧壁组成的开口盒子。省略参数时仍使用原来的 `outline` 黑色轮廓示例。

新工件自动进入**场景预览与 RGB-D 采集模式**：机械臂初始化到拍照用的 home 姿态，随后保持，不执行焊缝识别或沿线运动。当前检测器不支持拼板的交叉分支，单平面规划器也未处理盒壁；因此新工件不能与 `--plan-only` 或 `--demo-open` 一起使用。

进入 `quadweld` 环境后，通过 Windows WebRTC 客户端观看（两条命令分别运行，切换前退出上一实例；GPU 编号按本机空闲情况调整）：

```bash
# 2×2 方板，真实 3 mm 间隙；连接后可直接查看
python main.py --workpiece plate-grid --webrtc --auto-start --gpu 1 --output-dir outputs/runs/plate_grid

# 开口盒子，观察底板与侧壁的交接处
python main.py --workpiece open-box --webrtc --auto-start --gpu 1 --output-dir outputs/runs/open_box
```

也支持 `--robot rm65`。去掉 `--auto-start` 后，在串流面板点击 **Capture RGB-D** 或在终端按回车，才初始化机械臂并采集图像。使用本地图形界面时去掉 `--webrtc`；只生成、拍照并退出时将其替换为 `--headless`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--grid-count` | `2` | 每一边的板件数，总共 N×N 块 |
| `--tile-size` | `0.20` | 单块正方形板边长，米 |
| `--seam-gap` | `0.003` | 相邻板之间的真实间隙，米 |
| `--wall-height` | `0.12` | 盒内底面到侧壁顶端的高度，米 |
| `WorkpieceConfig.thickness_m` | `0.006` | 板件厚度，米，在 `config.py` 设置 |
| `WorkpieceConfig.box_size_m` | `0.40` | 盒子外部边长，米，在 `config.py` 设置 |

例如生成 3×3 拼板，保持整体尺寸接近默认工件：

```bash
python main.py --workpiece plate-grid --grid-count 3 --tile-size 0.14 --seam-gap 0.004 \
  --headless --gpu 1 --output-dir outputs/runs/plate_grid_3x3
```

默认拼板外尺寸为 `0.403 × 0.403 m`，盒子外尺寸为 `0.40 × 0.40 × 0.126 m`（包含底板厚度）。每块板具有独立静态碰撞体，盒内空腔保留。拼板下方有支撑垫，间隙不是用黑色几何条填出来的；缝中深度可能来自下面的支架。材质为简单金属材质，并增加斜上方区域光。这些是理想几何工件，尚未加入锈蚀、坡口、焊道或加工纹理。

每次运行保存 `workpiece.usda`（独立、可复用的工件）、`quadweld_scene.usda`（包含机器人与相机的完整场景）、`body_rgb.png`、`body_depth_m.npy` 和 `body_depth_preview.png`。配置保存在 `robot_config.json` 的 `workpiece` 字段中。`execution_report.json` 的 `status=scene_only` 表示预览，不代表完成焊接任务。

两份默认资产也保存在 [assets/workpieces/](assets/workpieces/README.md)，可直接导入其他 Isaac Sim 场景。资产使用米、Z 向上，默认根节点为 `/Workpiece`；原点为拼板上表面中心或盒内底面中心。独立资产不含相机、灯光和底座，不依赖外部贴图。建模尺寸仅供场景生成使用，没有传给感知或轨迹生成器。

本次验证通过 27 项 CPU 测试，并在 Isaac Sim 6 / RTX 3090 上生成两种工件的 RGB-D。拼板同时验证了 WebRTC 服务与预览面板启动，盒子验证了无界面采集，均正常退出；客户端实际视频接收仍取决于网络连接。原矩形场景回归通过 106 个路径采样点、179 个运动目标的视觉与 IK 预检查，本次未重复沿线执行。预览图片见 [拼板](outputs/runs/plate_grid/body_rgb.png) 和 [盒子](outputs/runs/open_box/body_rgb.png)。

## 场景参数

所有位置和尺寸使用米；世界坐标系为 Z 轴向上。默认使用 UR5e，以下为其场景参数；RM65 的独立配置见下文。

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

### 切换机械臂

通过 `--robot ur5e|rm65` 选择机器人，省略时使用 UR5e。两者共用 RGB-D 感知、轨迹生成和控制流程：

```bash
# RM65：无界面执行完整任务
python main.py --robot rm65 --headless --gpu 4 --output-dir outputs/runs/rm65

# RM65：通过 Windows WebRTC 观看，连接后点击 Start task
python main.py --robot rm65 --webrtc --gpu 4 --output-dir outputs/runs/rm65

# 切回 UR5e
python main.py --robot ur5e --webrtc --gpu 4 --output-dir outputs/runs/ur5e
```

串流命令每次只运行一个实例。机器人在启动时选择，切换需要退出当前进程后重新启动。`--plan-only`、`--demo-open` 和 `--path-mode` 对两种机器人均适用。RM65 开放 U 形路径也已通过视觉与 IK 预检查（42 个路径采样点、103 个运动目标），结果见 `outputs/runs/rm65_open_validation/`。

| 配置 | UR5e | RM65-6F |
| --- | --- | --- |
| USD | Isaac Sim 官方 UR5e 资产 | `assets/robots/rm65/RM65-6F/RM65-6F.usda` |
| 末端连杆 | `wrist_3_link` | `link_6` |
| 基座世界位置 | `(0.45, -0.62, 0.45)` | `(0.45, -0.38, 0.60)` |
| 测试矩形尺寸 | `0.30 × 0.20 m` | `0.20 × 0.10 m` |
| 相机标定 | `calibration/body_rgbd.json` | `calibration/rm65_body_rgbd.json` |
| 关节驱动刚度 / 阻尼 | `100000 / 5000` | `10000 / 500` |

RM65 的安装座更近、更高，测试轮廓更小，以满足当前工具朝下姿态的可达范围；这些配置只决定仿真测试场景，轨迹仍由视觉观测生成。各自的 home 关节角定义在 `config.py`，拍照时避开任务轮廓。RM65 相机相对基座位于 `(0, 0.08, 0.55)`，向前下方观察。

RM65 暂以 `link_6` 原点作为工具安装原点，沿其 +Z 添加 120 mm 简化工具。实际工具需校准 `flange_translation`、`flange_quaternion_wxyz` 和 `tool_length_m`；示例相机外参也需替换为实测标定。

`--robot-usd /path/to/model.usda` 只覆盖所选机器人的资产路径，不自动识别机型；RM65 应同时指定 `--robot rm65`。`--calibration /path/to/calibration.json` 可覆盖该机器人的默认相机标定。RM65 资产来源及目录说明见 [assets/README.md](assets/README.md)。

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

`--plan-only` 与 `--scene-only` 都会将机器人初始化到拍照用的 home 姿态。`--scene-only` 随后仅拍照，不规划或执行路径。
默认 `--path-mode auto` 自动区分闭合轮廓和开放线条，也可用 `--path-mode closed` 或 `--path-mode open` 限定类型。运行 `python main.py --headless --plan-only --demo-open` 会把测试矩形改成开放 U 形线，用于检查端点到端点规划。
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
mkdir -p outputs/streaming/webrtc
cd outputs/streaming/webrtc
isaacsim isaacsim.exp.full.streaming --no-window \
  --/exts/omni.kit.livestream.app/primaryStream/enableEventTracing=false \
  --exec /data/huangliqi/QuadWeld/stream_scene.py
```

串流启动后连接运行机器的 IP，默认信令端口为 `49100`、视频端口为 `47998`。`stream_scene.py` 会打开 `outputs/runs/default/quadweld_scene.usda` 并把界面视角对准机械臂与钢板；机身 RGB-D 相机保存在场景中。默认启用的串流事件追踪会生成占空间的 `NvStreamer-*.etli` 文件；上面的命令关闭追踪，并将其他运行文件集中在 `outputs/streaming/webrtc/`。

若默认 Isaac Sim 资产根目录不可访问，先取得完整的 UR5e USD 资产及其依赖，再指定主 USD 文件：

```bash
python main.py --headless --robot-usd /absolute/path/to/ur5e.usd
```

`outputs/` 按用途保存：`runs/` 为每次仿真运行与验证（默认运行写入 `runs/default/`），
`offline/` 为离线 RGB-D 处理与 SAM3 分割实验，`diagnostics/sam3/` 为 SAM3 环境报告，
`streaming/webrtc/` 为独立串流日志与追踪文件。每次运行的图像、轨迹、场景和报告保存在同一个运行目录。

成功后在运行目录生成的文件如下。USD 仍引用所选机器人的资产；复制到其他机器时需保证资产及其依赖可访问。

| 文件（默认在 `outputs/runs/default/` 下） | 内容 |
| --- | --- |
| `body_rgb.png` | 在 home 姿态采集的机身 RGB，未加入调试线 |
| `body_depth_m.npy`、`body_depth_preview.png` | 对齐的光轴 Z 深度（米）和仅供查看的灰度预览 |
| `rgbd_calibration.json` | 本次使用的安装与内参标定副本 |
| `robot_config.json` | 本次选择的机器人、资产路径、安装与控制参数 |
| `contour_mask.png`、`contour_overlay.png` | 阈值图、路径叠加；红点为起点，开放路径的蓝点为终点 |
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

## SAM3 分割实验

原图统一放在 [assets/weld_images/](assets/weld_images/)，共 12 张，来源见该目录的
`sources.csv`，点提示见 `point_prompts.csv`。全部分割实验集中在
[outputs/offline/sam3/](outputs/offline/sam3/)：

| 文件或目录 | 内容 |
| --- | --- |
| `overview.jpg`、`results.csv` | 每张原图一个代表结果，共 12 张，含人工核对说明 |
| `text_comparison.jpg`、`point_comparison.jpg` | 近正面图片的文字提示和三种点提示对照 |
| `masks/`、`overlays/` | 代表结果的原尺寸二值掩码和叠加图，与原图同名 |
| `trials.csv` | 80 次历史推理的统一索引；9 次接口修复前的点提示标为无效评估记录 |
| `history.zip` | 整理前全部原图、实验文件和日志，保留原路径及 SHA-256 迁移记录 |

`trials.csv` 中的 `archive` 指向历史压缩包，`mask_member`、`overlay_member` 和
`report_member` 是包内路径。`results.csv` 中的图像、掩码和叠加图路径可直接访问。
点提示的对象得分由接口固定为 1.0，不代表分割置信度。

当前结果：部分已有焊道可用文字或点提示分割；两张待焊主交线图片的文字提示未检出目标，
点提示主要覆盖板件区域，尚未得到正确交界线窄带掩码。
SAM3 环境、权重和本地接口补丁见 [thirdparty/README.md](thirdparty/README.md)。

在 `sam3` Conda 环境中运行新实验：

```bash
python scripts/segment_weld_images.py --prompt 'weld seam' --threshold 0.1
python scripts/segment_weld_images.py --points-csv assets/weld_images/point_prompts.csv
```

默认结果仍写入 `outputs/offline/sam3/`：新实验表与对照图为 `latest.csv`、`latest.jpg`，
掩码和叠加图的文件名包含提示词或点方案及文字检测阈值，多提示词共用上述两个图像目录。

## 算法与坐标约定

不启动 Isaac Sim，也可以在 CPU 上从文件生成同样的粗轨迹：

```bash
python generate_rgbd_trajectory.py \
  --rgb outputs/runs/default/body_rgb.png \
  --depth outputs/runs/default/body_depth_m.npy \
  --calibration outputs/runs/default/rgbd_calibration.json \
  --output-dir outputs/offline/rgbd_offline
```

实机输入为 RGB、对齐到该 RGB 图像坐标的深度 `.npy` 或单通道 PNG，以及标定 JSON。深度必须是光轴 Z 距离，不能是到光心的径向距离。米制浮点深度使用 `depth_scale_to_m=1`，毫米制 uint16 使用 `0.001`；预览 PNG 不能用于规划。内参必须匹配当前分辨率和畸变/校正状态。实机数据需要先由设备 SDK 完成 RGB/深度配准；此脚本不接入相机驱动。

`vision/contour_detector.py` 将 RGB 转成灰度并阈值化。闭合暗线仍通过轮廓层级检测并跟随可见线条的**外边缘**；开放暗线经过细化得到单像素中心线，从两个端点之一开始按连通关系排序。分叉线没有唯一的端点到端点顺序，因此会被拒绝。多目标相近时报告歧义；未检测到目标时终止，不使用已知矩形角点作为回退。阈值、最低轮廓面积可在 `config.py` 调整。

闭合轮廓从图像左上角附近开始，按图像顺时针排序，并把首点追加为末点。开放路径从图像位置较靠上的端点开始，只走到另一个端点；到达后原地沿板面法向抬起，不连接回起点。路径简化后逐段采样并保留拐角。

`vision/rgbd.py` 只接受 RGB、深度和标定。闭合路径使用轮廓内部深度，开放路径使用线条两侧的环带深度；将有效深度反投影为点云，RANSAC 去除离群点，再用 SVD 拟合平面。路径像素射线与这个**观测估计的平面**求交，避免直接采用黑线处容易缺失的深度。默认至少 40% 的支持区域有有效深度，拟合内点比例至少 70%，内点距离阈值 3 mm；不满足则拒绝生成。相机坐标为 OpenCV 光学系：+X 向右、+Y 向下、+Z 向前。`p_base = base_from_camera @ p_camera`（齐次列向量）；法向朝向相机，工具 +Z 朝向板面。TCP 沿估计法向留出 10 mm 间隙。旧的 USD 已知平面投影模块已移除。

本版直接恢复三维路径及板面法向，不依赖板子绝对世界位姿，也没有建立带稳定编号的板坐标系。可见的 AprilTag/ArUco 标记目前不参与计算；后续需要跨次观测保持同一起点、板坐标系或关联 CAD 路径时，再加入标记识别及其到任务轮廓的标定。当前仍要求可见的高对比度闭合任务轮廓，只有定位标记并不足以推断任意焊缝。

`robot/kinematics.py` 仍从机器人 USD 提取六个关节的运动学结构；主流程将模型置于基座系后求 IK。`robot/asset.py` 根据 USD 物理属性查找刚体和 articulation 根，兼容 UR5e 与 RM65 的不同层级，并在所选末端连杆下建立 `tool0/TCP`。UR5e 的工具安装坐标来自资产中的夹具挂载关节；RM65 使用配置中的安装变换。IK 使用有界数值最小二乘，以上一个关节解为初值，同时约束位置和姿态。世界变换只用于仿真显示和实际 link 位姿回读适配。实机应替换为机器人模型、编码器/FK 和控制接口；本次移除的是**工件位置特权信息**，并非把整个仿真控制适配器也替换为硬件驱动。

`planning/waypoint_generator.py` 负责几何路径；`robot/robot_controller.py` 提供 `move_to_pose` 和 `follow_cartesian_path`。预先求解整条路径，拒绝不可达目标和明显关节跳变。执行采用五次时间插值的关节位置目标，限制速度和加速度；每个采样点到达后继续，拐角停速，不做切角平滑。相邻关节插值段近似 Cartesian 直线，第一版不保证严格恒定焊接速度。闭合路径走回首点后抬起；开放路径在终点抬起。

只有初始化 home 状态时调用 `set_dof_positions`；任务运动全部调用 `set_dof_position_targets`，通过实际 link 位姿检查到位、超时和跟踪发散。任何失败都会停止继续下发路径并保存已有执行记录。简化工具只有视觉模型，无真实焊接过程。

## 运动调试与限制

- 没找到轮廓：查看 RGB 和遮挡，调整 `dark_threshold`；不要用矩形配置代替视觉结果。
- 反投影位置不对：检查 RGB/深度配准、光轴 Z 定义、深度单位、分辨率和外参方向；查看 `vision_report.json`。
- IK 无解或关节跳变：检查 home 姿态、机器人与钢板距离、工具长度和工具 +Z 方向；整个预检查通过前不开始走圈。
- 运动超时或抖动：查看执行 CSV 的误差，检查驱动参数、碰撞及关节速度；不要通过瞬移关节掩盖失败。
- 绿色路径、蓝色 TCP 路径和橙色实际轨迹应在俯视方向基本重合，蓝/橙轨迹比钢板高 10 mm。

预检查对机械臂连杆中心线及工具做观测板面的近似干涉检查：将同一平面的有效深度点凸包下方视为实体，不读取板子尺寸或地面高度。该检查不能覆盖未观察到的障碍、地面、自碰撞，也不是完整网格碰撞规划器。当前范围是静态、近似平面、单个高对比度闭合轮廓或无分叉开放线条，不包含四足运动、真实焊接、定位标记检测、RL 训练和实时视觉伺服。`rgbd_trajectory.npz` 的 `poses_base` 可作为以后残差策略的参考轨迹，但策略与训练环境尚未实现。

CPU 测试覆盖图像平移、深度高度变化、倾斜板面、噪声/缺失/离群点、毫米深度单位、外参变换和无有效观测时拒绝规划，以及原有 IK、控制失败停止、串流开始门控。仿真使用理想渲染深度；单元测试加入了合成噪声，尚未验证真实 RGB-D 在反光金属上的数据质量。执行报告的跟踪误差是相对视觉生成路径的控制误差；离板间隙也是相对估计平面，不代表真实工件测量精度。

本机 RGB-D 版本已通过 20 项 CPU 测试，并在 Isaac Sim 6 / RTX 3090（`--gpu 4`）完成闭合路径的 179 个运动目标、5,774 个执行采样，走圈后成功抬起。走圈相对视觉路径的 RMS 跟踪误差约 0.53 mm，最大约 0.84 mm；相对估计板面的间隙为 9.18–9.76 mm。验证结果在 `outputs/runs/rgbd_validation/`。开放 U 形测试场景也完成了 147 个运动目标；走线最大偏差约 0.77 mm，并在终点抬到估计板面上方约 99.1 mm，结果在 `outputs/runs/open_execution/`。

机械臂切换版本通过 22 项 CPU 测试，其中 RM65 的 USD 运动学与厂商 URDF 在三个关节姿态下独立对照（该项需要本地厂商源文件及 USD Python 绑定）。RM65 已在同一 GPU 完成 128 个运动目标、4,192 个执行采样，沿线 RMS 跟踪误差约 0.96 mm、最大约 1.49 mm，随后成功抬起；结果见 `outputs/runs/rm65_validation/`。UR5e 回归完成 179 个目标，沿线 RMS 约 0.53 mm，结果见 `outputs/runs/ur5e_switch_validation/`。这些误差均相对于视觉参考轨迹。

若图片没有钢板，检查机身相机视野、机械臂遮挡及裁剪范围。若默认资产加载失败，检查 Isaac Sim 资产根目录或使用 `--robot-usd`。若报 `No device could be created`、`NVML_ERROR_DRIVER_NOT_LOADED`，需要在能访问 NVIDIA 驱动和 GPU 的环境运行；无界面模式同样需要 GPU 渲染 RGB-D。离线生成器不需要 GPU。

代码面向 Isaac Sim 6，使用 `RtxCamera` 和 `CameraSensor` 的 `rgb`、`distance_to_image_plane` 获取同一渲染产品的 RGB-D。相机接口分辨率顺序为 `(高, 宽)`；标定文件使用 `(宽, 高)`。仿真传感器当前要求中心主点、零畸变、米制深度；离线算法支持标定文件中的 OpenCV 畸变和深度比例。场景/控制参数在 `config.py`，感知阈值默认值在 `vision/rgbd.py`，相机标定在 `calibration/body_rgbd.json`。

Isaac Sim 首次运行会提示接受 NVIDIA Omniverse 许可协议；请阅读提示并自行决定是否接受。许可协议是运行 Isaac Sim 的前提。

参考：[Isaac Sim 6 UR5e 资产](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html)、[Isaac Sim 6 Python 运行方式](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)、[Isaac Sim 6 相机 API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html)。
