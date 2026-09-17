# 本地机器人资产

目录用途：`vendor/` 保存厂商提供的 URDF 和网格源文件，`robots/` 保存转换后可复用的 USD 资产。仿真图像、轨迹和执行报告仍保存在项目根目录的 `outputs/`。

RM65-6F 已完成 USD 转换，主入口为 `robots/rm65/RM65-6F/RM65-6F.usda`。其 `payloads/` 包含网格、材质及物理配置，移动或分发时需保留整个 `RM65-6F/` 目录。重新转换时，将 `URDFImporterConfig.usd_path` 设置为项目中的 `assets/robots/rm65/` 绝对路径。

RM65-6F 已从睿尔曼官方 Gitee 仓库下载到 `vendor/rm_models/`，使用稀疏检出保留 `RM65/urdf/RM65-6F` 目录及仓库根文件。

- 来源：https://gitee.com/RealManRobot/rm_models.git
- 下载版本：`bdb12ca3db532cb677ae33fa94795ac9c1b01f98`
- URDF：`vendor/rm_models/RM65/urdf/RM65-6F/urdf/RM65-6F.urdf`
- 网格：`vendor/rm_models/RM65/urdf/RM65-6F/meshes/`
- 导入时 ROS package 映射：`RM65-6F` → 上述 `RM65-6F` 根目录。

第三方仓库及网格已加入项目 `.gitignore`。在项目根目录重新下载：

```bash
git clone --depth 1 --filter=blob:none --sparse https://gitee.com/RealManRobot/rm_models.git assets/vendor/rm_models
git -C assets/vendor/rm_models sparse-checkout set RM65/urdf/RM65-6F
```

以上命令下载当时的默认分支版本；下载版本号用于记录本次来源。已检查 URDF 和七个网格文件，转换后的九个 USD 文件迁移到 `robots/` 后也已通过依赖解析检查。

已接入主程序，可用 `python main.py --robot rm65 --headless` 执行，或使用 `--webrtc --gpu 4` 串流观看。`--robot ur5e` 切回默认机器人。机器人配置包括关节名称、工具安装、home 姿态、驱动参数和相机标定，见项目根目录 `config.py` 与 `calibration/rm65_body_rgbd.json`。RM65 使用 `link_6` 原点和沿 +Z 的 120 mm 简化工具，实际安装需重新标定。
