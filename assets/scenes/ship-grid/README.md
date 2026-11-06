# 船板焊接格架环境

`ship_grid.usda` 是可独立引用的环境资产，来自 RoboSimStudio 中当前实时构建的船板格架。
只包含船板、T 形肋条、周围显示地面和材质；不包含 B2、Z1、安装板、相机、灯光、控制器或焊缝坐标。
使用它不需要安装 RoboSimStudio。请整体复制本目录，保留 `textures/` 的相对位置。

| 项目 | 约定 |
| --- | --- |
| 格架 | 3×3 格，单格立板内表面净距 2×2 m |
| 连续底板 | 6.45×6.45 m，厚 10 mm |
| 加强肋 | 足板宽 180 mm；立板高 220 mm、厚 10 mm；总高 230 mm |
| 坐标 | 米制，Z 向上；原点为中央格中心的底板上表面 |
| 高度 | 底板 z=-0.01…0；足板 z=0…0.01；立板 z=0.01…0.23 m |
| 根节点 | `/ShipGrid`，已设为 default prim |
| 碰撞 | 33 个静态解析箱体：1 底板、16 足板、16 立板；各自独立，不封堵格内净空 |
| 物理材质 | 静/动摩擦均为 0.65，恢复系数为 0；未添加刚体或关节 |
| 显示 | 49 个格架网格、1 个场地网格，USD Preview Surface 金属材质及两张 PNG 贴图 |

所有碰撞箱体都在各部件的 `Collision` 节点上，隐藏显示但启用碰撞；`Visual` 节点只负责外观。
相邻格共用加强肋。16 个亮色切割顶面和周围制作场地地面仅用于显示。
本资产不包含原实时场景的无限地平面；格架外如果也要承重，请使用目标仿真已有的地面，
其上表面可放在 z=-0.01 m，与钢底板下表面对齐。

在已初始化的 Isaac Sim 场景中，加载现有机器狗和机械臂后引用环境：

```python
from pathlib import Path
from scene.environment import add_environment

asset = Path("assets/scenes/ship-grid/ship_grid.usda").resolve()
add_environment(stage, asset)  # /World/Environment；中央格地面上表面 z=0
```

函数要求目标 stage 也是米制、Z 向上。`at=(x, y, z)` 可整体平移资产，默认保持源场景坐标。
机器人配置和机身/机械臂安装关系由目标项目管理。中央格立板内表面约为 x、y=±1 m；
按脚底接触 z=0 设置四足的站立根高度，不能把躯干根节点直接放在 z=0。

也可完全不使用本项目的辅助函数，直接用 USD reference：

```python
stage.DefinePrim("/World/Environment", "Xform").GetReferences().AddReference(str(asset))
```

QuadWeld 中的副本有一处外观修正：导出时贴图又乘了一次形体颜色（`UsdUVTexture inputs:scale`），
立板反照率只剩约 0.04，渲染接近纯黑。RoboSimStudio 的 `clear_texture_tints` 说明，视觉匹配渲染端直接显示贴图。
因此三个带贴图材质的 `inputs:scale` 已改为 `(1, 1, 1, 1)`，几何与碰撞未改。
重新导出前需在 `export_environment.py` 中去掉该缩放，否则导出后会恢复原状。

`manifest.json` 保存源文件哈希、布局参数及各部件实际位置/尺寸，供资产核对使用，
没有接入焊缝识别或轨迹规划。导出读取实时编译模型的几何，保留了约 1e-7 m 的浮点误差。

当前结构为固定支承工件，没有熔池、焊接热过程、坡口、真实焊道和变形。
纹理使用新的 UV 映射，渲染器不同也会影响观感。已验证 USD 几何、静态碰撞、
物理材质绑定、相对贴图路径和搬迁后的组合加载。QuadWeld 中 B2 的 Isaac Sim 6
运行记录及定位误差见[项目 README](../../../README.md#船板格架角焊b2-前置-rgb-相机--auto_weld)。

在 RoboSimStudio 的源 twin 中重新导出：

```bash
./ctl eval - --wait <<'PY'
from export_environment import export_environment
export_environment(sim, grid)
PY
```

输出在 twin 的 `assets/export/ship-grid/`。复制整个目录到目标项目的
`assets/scenes/ship-grid/`；这一步不需要导入或复制机器人文件。
