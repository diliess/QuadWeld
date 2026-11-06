# B2 背负 E10L-Pro 焊接臂

`B2-E10L/B2-E10L.usda` 是一个浮动底座 articulation：宇树 B2 机器狗（12 个腿部关节）的躯干背部正中固定一块转接板，
其上安装华言 Elfin E10L-Pro 六轴臂和 TRM351 焊枪，共 18 个转动关节。`manifest.json` 记录来源版本、关节名、
安装位姿、TCP 和全部假设值。主程序配置为 `--robot b2_e10l`，见 [项目 README](../../../README.md#b2-背负-e10-l-焊枪)。
移动或分发时保留整个 `B2-E10L/` 目录（约 17 MB）。

| 项目 | 约定 |
| --- | --- |
| 根节点 | B2 `base_link`，即 articulation root；无世界固定关节 |
| 机械臂命名 | 所有连杆、关节加 `e10l_` 前缀，如 `e10l_base_link`、`e10l_joint1`…`e10l_joint6` |
| 转接板 | 260×240×20 mm 铝板（假设），下表面位于躯干坐标 z=0.09 m，略高于躯干外壳 0.088 m |
| 机械臂基座 | 躯干坐标 `(0, 0, 0.11)`，与 B2 朝向一致；B2 质心约在 x=+0.02 m |
| TCP | `e10l_Link6` 坐标 `(0.0017, 0.0009, 0.5136)` m，即焊丝尖端；+Z 沿直喷嘴，与法兰轴夹角约 60° |
| 碰撞 | B2 为原始基本体；机械臂与焊枪为网格凸包；关闭自碰撞 |
| 总质量 | 124.5 kg：B2 74.6 kg，机械臂 45 kg，焊枪 1.5 kg，转接板 3.4 kg |

B2 原始 URDF 中 6 个零质量传感器坐标系（`lidar_link`、`imu_link` 等）已删除，避免近零质量连杆影响求解；
激光雷达的碰撞圆柱并入躯干。腿部几何、惯量、限位和力矩上限保持原样。

## 假设值

以下数据未随模型提供，替换为实测或厂家数据后重新构建：

- **机械臂质量**：导出的 CAD 惯量合计 29.4 kg，与厂家规格 45 kg 不符，各连杆质量和惯量按同一比例放大到 45 kg，质心位置不变。
- **焊枪**：质量假设 1.5 kg，惯量由网格表面分布估算。网格不封闭，不能按体积计算质量。
- **关节力矩上限**：取随机姿态下最大静态重力矩的 1.5 倍，并以同级 10 kg 协作臂（UR10e）的额定值 330/330/150/56/56/56 N·m 为下限。
  J1 轴竖直，重力矩为零，只能使用这个下限。
- **关节速度**：按厂家规格，J1–J2 100°/s，J3–J4 150°/s，J5–J6 180°/s。
- **TCP**：取网格中沿喷嘴轴最远的焊丝尖端，未经实物标定。

机械臂 45 kg 加焊枪与转接板约 50 kg，超过 B2 行走时 40 kg 的额定负载，仅在站立允许范围内。
当前仿真只做站立焊接，未接入行走。

## 重新构建

源文件不入库，放在 `assets/vendor/`（已加入 `.gitignore`）。在项目根目录准备：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git assets/vendor/unitree_ros
git -C assets/vendor/unitree_ros sparse-checkout set robots/b2_description
mkdir -p assets/vendor/e10l && cp /path/to/带焊接枪E10-L机械臂模型.zip assets/vendor/e10l/
cd assets/vendor/e10l && unzip -O GBK 带焊接枪E10-L机械臂模型.zip \
  && unzip 带焊接枪E10-L机械臂模型/E10L_pro_with_welding_torch_urdf.zip && cd -
```

本次使用 `unitree_ros` 提交 `15ebe53f4cd77e7d745ae83b9d7ff90e685e8998`；压缩包的 SHA-256 记录在 `manifest.json`。
然后合成 URDF 并转换（转换需要 Isaac Sim 6 和 GPU，约 1 分钟）：

```bash
python scripts/build_b2_e10l.py              # URDF → assets/vendor/b2_e10l/，USD 和 manifest → 本目录
python scripts/build_b2_e10l.py --urdf-only  # 只合成 URDF，并更新 manifest.json
```

安装位置、转接板尺寸和假设质量都在 `scripts/build_b2_e10l.py` 开头。修改 TCP 后需同步 `config.py` 中的
`flange_translation` 和 `flange_quaternion_wxyz`；`tests/test_b2_e10l.py` 会校验二者与 `manifest.json` 一致，
并核对 USD 中的机械臂正运动学与合成 URDF 一致。
