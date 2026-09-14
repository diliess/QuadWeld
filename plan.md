可以。你现在最适合让 AI 先帮你生成一个**最小可运行版本的 Isaac Sim 场景**：机械臂 + 钢板 + 简单轮廓 + 固定相机，暂时不接四足、不做真实焊接、不先上 ROS 2。

下面这个 Prompt 可以直接丢给 Codex、Claude Code、Cursor 或其他编程 AI。

你是一名熟悉 NVIDIA Isaac Sim、机器人运动学、OpenCV、ROS 2 和 MoveIt 2 的机器人仿真工程师。

我正在开发一个“四足机器人搭载机械臂进行船体钢板焊接”的项目。

最终目标是：

四足机器人携带 6 自由度机械臂和焊枪，在船体钢板表面识别焊缝，并使机械臂末端沿焊缝轨迹运动。

但是目前不要实现完整系统。

我现在只希望搭建一个最小可运行的基础仿真场景，用于验证：

“相机观察钢板上的几何轮廓 → 提取轮廓 → 转换成机械臂坐标系中的路径点 → 机械臂末端沿轮廓完整运动一圈。”

请帮助我从零搭建这个 Isaac Sim 仿真项目。

## 一、当前阶段目标

第一阶段只包含：

1. 一台固定底座的 6 自由度机械臂
2. 一块平面钢板
3. 钢板表面一个明显的二维几何轮廓
4. 一台固定 RGB 相机
5. 一个简化的机械臂末端 TCP
6. 使用视觉获取轮廓
7. 将图像轮廓转换到机器人世界坐标系
8. 生成 Cartesian waypoints
9. 控制机械臂末端沿轮廓完整走一圈

暂时不要加入：

* 四足机器人
* 真实焊接过程
* 熔池
* 焊接热力学
* 焊枪电弧
* 强化学习
* 神经网络
* 复杂 3D 焊缝
* 实时视觉伺服
* RGB-D 深度相机
* 动态钢板

第一阶段只做一个尽可能简单、稳定、容易调试的 MVP。

---

## 二、仿真平台

使用：

NVIDIA Isaac Sim

尽量采用 Isaac Sim 官方推荐的 Python API 和 Articulation API。

请避免使用已经废弃的旧 API。

如果不同 Isaac Sim 版本 API 有差异，请指出可能需要修改的位置，而不是假设某个旧版本一定可用。

---

## 三、机械臂

第一版优先使用 Isaac Sim 中已有、容易加载和控制的机械臂。

优先考虑：

* UR5 / UR5e
* Franka Panda

如果其中某一个在 Isaac Sim 中更加容易实现 Cartesian end-effector control，请优先选择它。

暂时不要使用我最终准备采购的真实机械臂。

后续我可能会替换成：

* RealMan RM65
* EliteRobot CS66
* FAIRINO FR3-WMS
* DynaArm

所以请设计程序结构时，不要让视觉和轨迹生成代码与具体机械臂强耦合。

将机器人相关参数单独配置，例如：

robot_base_frame
end_effector_frame
joint_names
robot_usd_path
TCP offset

---

## 四、场景设计

创建一个简单场景。

推荐坐标约定：

World：

Z 轴向上。

机械臂：

固定安装在世界坐标系中。

钢板：

放置在机械臂前方工作空间内。

例如可以先设置为水平钢板：

z = 0.5 m

钢板尺寸例如：

800 mm × 600 mm

厚度：

10 mm

钢板颜色：

浅灰色。

---

## 五、轮廓目标

第一版在钢板表面创建一个非常明显的矩形轮廓。

例如：

矩形尺寸：

300 mm × 200 mm

轮廓可以采用：

* 黑色线条
  或者
* 稍微凸起的细线
  或者
* 独立的深色几何体

目标是让 RGB Camera 非常容易识别。

不要一开始追求真实焊缝视觉效果。

第一阶段要求：

OpenCV 能稳定检测这个矩形轮廓。

后续再支持：

* 圆
* 椭圆
* 多边形
* 不规则曲线
* 实际焊缝

---

## 六、相机

在钢板上方安装一台固定 RGB Camera。

第一阶段采用近似俯视。

例如：

Camera
↓
Steel Plate

要求钢板和矩形轮廓完整出现在图像中。

请合理设置：

camera position
camera orientation
focal length
resolution

例如：

640 × 480

并解释为什么这样设置。

相机需要输出 RGB 图像供 OpenCV 使用。

---

## 七、视觉处理

第一阶段使用最简单可靠的方法。

推荐流程：

RGB image
→ grayscale
→ threshold 或 Canny
→ findContours
→ 筛选目标 contour
→ contour resampling
→ 得到有序轮廓点

使用 Python + OpenCV。

请避免深度学习。

需要完成：

detect_contour(image)

返回类似：

[(u1,v1),
(u2,v2),
...
(un,vn)]

注意：

轮廓点必须按照连续路径排序，机械臂需要沿同一个方向连续运动，不能随机跳点。

对轮廓进行降采样。

例如：

每隔 5～15 mm 左右生成一个 waypoint。

不要直接使用几千个像素点作为机械臂路径。

---

## 八、2D 像素 → 3D 世界坐标

第一阶段钢板是已知平面，因此暂时不要使用深度相机。

请实现：

pixel_to_world(u, v)

将图像像素坐标：

(u,v)

转换成钢板平面上的：

(x,y,z)

由于钢板是固定平面，可以优先使用：

Homography

或者：

camera intrinsic + ray-plane intersection

请推荐其中一种更适合当前 MVP 的方法，并说明理由。

要求最终获得：

P_world =
[x, y, z]

其中 z 为钢板表面高度。

---

## 九、TCP 焊接偏移

机械臂末端不要直接接触钢板。

假设末端 TCP 位于钢板表面上方：

10 mm

即：

TCP_offset = 0.01 m

路径应该变成：

[x,
y,
plate_z + 0.01]

暂时保持 TCP 姿态固定。

例如：

工具 Z 轴始终垂直于钢板。

暂时不用根据轨迹切线改变焊枪姿态。

---

## 十、轨迹生成

实现：

generate_waypoints(contour_world_points)

输出：

Pose 1
Pose 2
Pose 3
...
Pose N

并确保最后：

Pose N → Pose 1

完成整个闭合轮廓。

路径应该：

* 连续
* 平滑
* 没有明显跳跃
* 不超过机械臂工作空间
* 尽量避免奇异位形

---

## 十一、机械臂运动

第一版本优先采用 Isaac Sim 内部 IK / Cartesian control。

如果实现简单，可以使用：

Articulation
+
IK Solver

控制机械臂末端按照 waypoints 依次运动。

暂时不要求 ROS 2。

后续我会加入：

ROS 2
+
MoveIt 2

所以请将控制接口封装，例如：

move_to_pose(target_pose)

以及：

follow_cartesian_path(waypoints)

这样以后可以替换控制后端，而不改变视觉代码。

---

## 十二、推荐程序结构

请尽量按照模块化方式组织代码，例如：

project/
│
├── main.py
│
├── scene/
│   └── create_scene.py
│
├── robot/
│   ├── robot_controller.py
│   └── ik_controller.py
│
├── vision/
│   ├── camera.py
│   ├── contour_detector.py
│   └── pixel_to_world.py
│
├── planning/
│   ├── waypoint_generator.py
│   └── path_smoothing.py
│
└── config/
└── config.py

如果对于第一版本来说过度复杂，可以适当简化，但请保持：

场景
视觉
坐标转换
轨迹规划
机器人控制

这几个逻辑彼此分离。

---

## 十三、程序运行流程

最终 main.py 的逻辑应该接近：

1. 启动 Isaac Sim
2. 创建 World
3. 加载机械臂
4. 创建钢板
5. 创建矩形轮廓
6. 创建 RGB Camera
7. Reset simulation
8. 获取 Camera Image
9. OpenCV 检测 contour
10. 对 contour 降采样
11. pixel → plate/world coordinates
12. 加 TCP offset
13. 生成 Pose waypoints
14. 机械臂移动到第一个 waypoint 上方
15. 下降到工作高度
16. 沿 contour waypoint 连续运动
17. 完整走一圈
18. 返回安全位置

建议加入：

Approach Pose

例如：

先运动到：

TCP_z = plate_z + 100 mm

然后下降到：

TCP_z = plate_z + 10 mm

完成路径后再抬起到：

plate_z + 100 mm

这样更接近未来真实焊接流程。

---

## 十四、可视化调试

请加入尽可能多但不过度复杂的调试可视化。

例如：

在 Isaac Sim 中显示：

* 检测得到的轮廓点
* 转换后的 world waypoint
* TCP 轨迹
* 第一个 waypoint
* 路径方向

如果方便，可以使用小球或 debug draw：

● ● ● ● ●
●       ●
●       ●
● ● ● ● ●

让我能够肉眼确认：

“视觉得到的轨迹”和“机械臂实际走的轨迹”是否一致。

OpenCV 侧也显示：

原始图像
+
检测后的 contour overlay

---

## 十五、坐标系调试

这是非常重要的部分。

请明确区分：

World Frame

Robot Base Frame

Camera Frame

Image Pixel Frame

TCP Frame

Plate Frame

并告诉我每一步采用什么变换。

例如：

Pixel

→ Camera Ray

→ Plate Plane

→ World

→ Robot Base

→ TCP Pose

不要把不同坐标系混在一起。

请在代码中明确变量命名，例如：

point_pixel

point_camera

point_world

point_robot_base

tcp_pose

---

## 十六、第一阶段验收标准

程序运行以后，我希望看到：

一个机械臂

↓

前方有一块钢板

↓

钢板上有一个黑色矩形

↓

顶部 RGB Camera 获取图像

↓

程序自动识别矩形轮廓

↓

Isaac Sim 中显示转换后的路径点

↓

机械臂末端移动到矩形附近

↓

末端距离钢板约 10 mm

↓

机械臂沿矩形：

左上
→
右上
→
右下
→
左下
→
左上

完整走一圈

↓

末端抬起

↓

任务结束

整个过程不需要人工输入矩形的四个角坐标。

路径必须来自 Camera Image。

---

## 十七、输出要求

请不要一次只告诉我理论。

我要实际搭建项目。

请按照下面顺序回答。

第一部分：

给出整体技术方案，并说明为什么这么设计。

第二部分：

给出 Isaac Sim 场景的具体参数：

* Robot position
* Plate position
* Plate size
* Camera position
* Camera orientation
* Camera resolution
* Rectangle size
* TCP offset

全部使用 SI 单位。

第三部分：

给出完整项目目录。

第四部分：

给出每一个 Python 文件的完整可运行代码。

不要只给伪代码。

第五部分：

告诉我如何运行。

例如：

使用 Isaac Sim Python 环境运行：

./python.sh main.py

或者对应当前 Isaac Sim 推荐的运行方式。

第六部分：

给出调试方法。

包括：

* 相机看不到钢板怎么办
* OpenCV 没找到 contour 怎么办
* pixel-to-world 位置不对怎么办
* 机械臂 IK 无解怎么办
* TCP 姿态反了怎么办
* 路径方向反了怎么办
* 机械臂运动抖动怎么办

第七部分：

指出哪些地方以后可以替换为：

RealMan RM65

以及以后如何扩展为：

RGB-D
→
3D 焊缝
→
MoveIt 2
→
MoveIt Servo
→
机械臂安装到四足机器人

---

## 十八、编码原则

请遵循：

1. 优先保证 MVP 可以运行
2. 不要过度设计
3. 不要一开始加入 ROS 2
4. 不要一开始加入 MoveIt 2
5. 不要一开始加入四足机器人
6. 不要使用深度学习
7. 不要假设手工已知矩形角点
8. 轨迹必须由 Camera Image 得到
9. 所有距离统一使用米
10. 所有角度明确使用 degree 或 radian
11. 避免使用 Isaac Sim 已废弃 API
12. 对容易受 Isaac Sim 版本影响的 API 明确标注
13. 所有关键参数集中放到 config
14. 给代码增加必要注释，但不要堆大量无意义注释

如果某个设计存在多种方案，请不要把所有方案都实现。

直接选择最适合 MVP、最稳定、调试成本最低的一种。

最终目标不是做漂亮 Demo，而是搭出后续“真实焊缝视觉跟踪 + 四足机器人机械臂焊接”的正确软件基础。
