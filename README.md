# QuadWeld：第一轮 Isaac Sim 6 场景

这一轮只创建固定 UR5e 机械臂、钢板、黑色矩形轮廓和俯视 RGB 相机，并保存场景及原始相机图片。没有视觉检测、坐标反投影、轨迹规划或机械臂运动代码。场景和参数分别放在 `scene/` 与 `config.py`，方便以后替换机械臂或接入视觉模块。

## 场景参数

所有位置和尺寸使用米；世界坐标系为 Z 轴向上。

| 对象 | 参数 |
| --- | --- |
| UR5e 固定底座 | 世界位置 `(0.45, -0.62, 0.45)`；下方有 `0.26 × 0.26 × 0.45` 支座；官方资产 `/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` |
| 钢板 | 中心 `(0.45, 0, 0.50)`；长 × 宽 × 厚 `0.80 × 0.60 × 0.01`；上表面 `z=0.505` |
| 黑色矩形 | 位于钢板中央；外尺寸 `0.30 × 0.20`；线宽 `0.008`；线条高出钢板 `0.0001` |
| RGB 相机 | 世界位置 `(0.45, 0, 1.75)`；正对钢板；`640 × 480`；焦距 `0.024`；水平光圈 `0.020955` |
| 预留 TCP 偏移 | `0.01`，本轮只存配置，不创建或控制 TCP |

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

无界面创建、拍照并退出：

```bash
conda activate quadweld
cd /data/huangliqi/QuadWeld
python main.py --headless
```

若使用 Isaac Sim WebRTC 客户端，可先运行上面的无界面命令更新 USD，再以 Isaac Sim 6 的完整串流体验打开它：

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

成功后会生成 `outputs/quadweld_scene.usda` 和 `outputs/overhead_rgb.png`。前者仍引用 UR5e USD 资产；复制场景文件到别的机器时，该资产也必须可访问。打开 USD 后，应能看到机械臂、架在支座上的浅灰钢板，以及表面中央的黑色矩形。图片应完整包含钢板和矩形。脚本有界面运行时会保持窗口打开；关闭窗口即可退出。

若图片没有钢板，先检查相机 prim `/World/Sensors/OverheadRGB` 的位置、旋转和裁剪范围，以及 `config.py` 中的钢板位置。若默认资产加载失败，检查 Isaac Sim 资产根目录或使用 `--robot-usd`。若报 `No device could be created`、`NVML_ERROR_DRIVER_NOT_LOADED`，需要在能访问 NVIDIA 驱动和 GPU 的环境运行；无界面模式同样需要 GPU 才能渲染 RGB 图像。

代码面向 Isaac Sim 6，使用 `isaacsim.sensors.experimental.rtx.RtxCamera` 和 `CameraSensor` 获取 RGB，使用 `SimulationManager` 管理仿真，避免 6.0 已弃用的 `isaacsim.sensors.camera` 与 `isaacsim.core.api`。相机运行接口的分辨率顺序为 `(高, 宽)`；配置仍写作 `(宽, 高)`。参数均在 `config.py`；以后切换机械臂时改资产路径和机器人坐标系参数即可。

Isaac Sim 首次运行会提示接受 NVIDIA Omniverse 许可协议；请阅读提示并自行决定是否接受。许可协议是运行 Isaac Sim 的前提。

参考：[Isaac Sim 6 UR5e 资产](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html)、[Isaac Sim 6 Python 运行方式](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)、[Isaac Sim 6 相机 API](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html)。
