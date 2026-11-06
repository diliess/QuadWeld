# 场景资产

每个子目录是一个可整体复制的场景包，包含一份 USD 和一份 `manifest.json`；外部导出的包另附说明和贴图。

| 场景包 | 用途 | 来源 | 根节点 | 原点 | 默认几何 |
| --- | --- | --- | --- | --- | --- |
| [plate-grid/](plate-grid/) | 工件，放在机械臂前的支架上 | `scene/workpieces.py` 生成 | `/Workpiece` | 拼板上表面中心 | 2×2 方板；单板 0.20 m，厚 0.006 m，间隙 0.003 m；总边长 0.403 m |
| [open-box/](open-box/) | 工件，放在机械臂前的支架上 | `scene/workpieces.py` 生成 | `/Workpiece` | 盒内底面中心 | 外边长 0.40 m，板厚 0.006 m，侧壁高 0.12 m |
| [ship-grid/](ship-grid/README.md) | 环境，替换示例地面和工件 | RoboSimStudio 导出 | `/ShipGrid` | 中央格中心的底板上表面 | 3×3 格，单格净距 2×2 m，底板 6.45×6.45 m |

## 共同约定

- 米制、Z 向上，已设 default prim；根节点下只有 `Looks`（材质）和 `Parts`（部件）。
- 局部 Z=0 是承载面的上表面，板厚位于 Z=0 以下。
- 每个部件单独启用静态碰撞；不含刚体、关节、机器人、相机、灯光或焊缝标注线。
- 建模尺寸只用于搭建场景，不传给感知或轨迹生成。
- `manifest.json` 都有 `asset`、`default_prim`、`units`、`up_axis`、`origin`、`robots_included`、`source`。
  生成的工件另有 `workpiece` 字段，记录完整的 `WorkpieceConfig` 参数；船板格架另有布局参数和各部件的实际位置与尺寸。

两种加载方式都经过 `scene/assets.py` 的 `reference_asset`，统一检查单位、轴向和 default prim：
`add_workpiece` 把工件引用到 `/World/Workpiece/Assembly/Model`，并在场景中另建支架；
`add_environment` 把环境引用到 `/World/Environment`。不使用项目代码时，也可直接添加 USD reference。

## 生成的工件

底板占据 Z=-0.006 到 0。每块板单独启用静态碰撞，未给整个盒子生成凸包；材质为无贴图的金属 USD Preview Surface。
完整机器人场景的预览方式和自定义参数见 [项目 README](../../README.md#拼板与开口盒子)。
`main.py` 每次按命令行参数在输出目录生成 `workpiece.usda`，不会覆盖这里的默认资产。

修改 `scene/workpieces.py` 后，在项目根目录重建（只用 USD Python 绑定，不需要 GPU 渲染）：

```bash
uv run --locked --extra usd python - <<'PY'
from config import WorkpieceConfig
from scene.workpieces import export_workpiece_bundle
for kind in ("plate-grid", "open-box"):
    export_workpiece_bundle(f"assets/scenes/{kind}", WorkpieceConfig(kind=kind))
PY
```

`tests/test_scene_assets.py` 按各包 `manifest.json` 中的参数重新生成并逐字节比较；资产与代码不一致时该测试失败。

船板格架的导入、坐标约定和重新导出方法见 [ship-grid/README.md](ship-grid/README.md)。
