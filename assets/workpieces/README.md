# 参数化工件资产

这两份 USD 由项目的 `scene/workpieces.py` 生成，没有外部网格或贴图依赖，可直接在 Isaac Sim 中打开，或作为 reference 加入场景。

| 资产 | 默认几何 |
| --- | --- |
| [plate-grid.usda](plate-grid.usda) | 2×2 方板；单板 0.20 m，厚 0.006 m，间隙 0.003 m；总边长 0.403 m |
| [open-box.usda](open-box.usda) | 外边长 0.40 m，板厚 0.006 m，侧壁高 0.12 m；底板与四面侧壁独立建模 |

统一约定：米制、Z 向上、默认根节点 `/Workpiece`。局部 Z=0 为拼板上表面或盒内底面，底板占据 Z=-0.006 到 0。每块板单独启用静态碰撞，未给整个盒子生成凸包，也未添加动态刚体。资产内不含机器人、支架、灯光或焊缝标注线。

在项目根目录、`quadweld` 环境中可以重建默认资产（此步骤只用 USD Python 绑定，不需要启动 GPU 渲染）：

```bash
python - <<'PY'
from config import WorkpieceConfig
from scene.workpieces import export_workpiece_asset
for kind in ("plate-grid", "open-box"):
    export_workpiece_asset(f"assets/workpieces/{kind}.usda", WorkpieceConfig(kind=kind))
PY
```

预览完整机器人场景及自定义参数的方式见 [项目 README](../../README.md#拼板与开口盒子)。`main.py` 每次按配置生成输出目录中的 `workpiece.usda`，不会覆盖本目录里的默认资产。
