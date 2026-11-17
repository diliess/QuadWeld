# outputs 保留与清理记录

2026-11-17 已按用户确认执行前三条清理建议，保留整个 archive。此前评估原文保存在 [清理前评估](../outputs/validation/20261117_01__maintenance__outputs__cleanup/retention_before_cleanup.md)。

本次删除 10 个目录组、1317 个文件，原文件内容约 1.03 GiB；规范命名 18 个保留批次。目录元数据及新核验结果会使磁盘净占用与该数略有差异。

## 已删除

| 原目录 | 文件数 | 原文件大小 MiB |
|---|---:|---:|
| `outputs/benchmarks/bead_refactor_20261117` | 64 | 12.64 |
| `outputs/benchmarks/project_cleanup` | 62 | 12.64 |
| `outputs/benchmarks/bead_sweep_20261116_01` | 3 | 0.03 |
| `outputs/bead_toe_demo` | 7 | 1.03 |
| `outputs/stereo_seam_demo` | 3 | 0.41 |
| `outputs/setup` | 2 | 0.10 |
| `outputs/benchmarks/bead_toes_20261113_01` | 121 | 47.21 |
| `outputs/hj90` | 926 | 832.53 |
| `outputs/runs/a04_20261109_191201_351388` | 2 | 0.01 |
| `outputs/runs/a04_20261109_191213_153744` | 127 | 146.14 |

仅删除评估中前三条所列的目录；archive、五组核心双目实验、场景导入/标定验证、预选依据及旧根线报告均保留。删除前的文件名、大小与 SHA256 见 [删除清单](../outputs/validation/20261117_01__maintenance__outputs__cleanup/deleted_files.json)。

## 命名与新旧路径

格式：`日期_批次__方法__工位与侧面__目的`。新自动运行用台北日期时间和微秒保证唯一性：

```text
outputs/runs/<YYYYMMDD_HHMMSS_ffffff>__stereo__A25_front__sweep/
outputs/runs/<YYYYMMDD_HHMMSS_ffffff>__stereo__A26_front__replay/
outputs/validation/<日期时间>__active_stereo__<姿态或试件>__<模式>_scan/
outputs/imports/<日期时间>__scene__industrial_welding__backup/
```

方法/目的使用固定词，工位沿用项目大小写与 front/back/both。仅历史场景报告没有可信日期，使用 undated；没有补造时间。正式运行的状态写 JSON，不以 final、latest、cleanup 等模糊名称反复改名。

| 旧目录 | 新目录 |
|---|---|
| `outputs/benchmarks/staged_regions_20261117_01` | `outputs/benchmarks/20261117_01__stereo__multi8__staged` |
| `outputs/benchmarks/bead_sweep_20261116_02` | `outputs/benchmarks/20261116_02__stereo__A25-A26_front__sweep` |
| `outputs/benchmarks/bead_toes_20261113_02` | `outputs/benchmarks/20261113_02__stereo__A25-A26_both__view_compare` |
| `outputs/benchmarks/candidate_regions_20261117_01` | `outputs/benchmarks/20261117_01__stereo__multi6__root_negative` |
| `outputs/benchmarks/candidate_back_beads_20261117_01` | `outputs/benchmarks/20261117_01__stereo__A25-A26_back__roll_compare` |
| `outputs/benchmarks/stereo_plane_full_20261111_01` | `outputs/benchmarks/20261111_01__stereo__multi3__root_full` |
| `outputs/benchmarks/stereo_plane_close_20261111_01` | `outputs/benchmarks/20261111_01__stereo__multi3__root_close` |
| `outputs/benchmarks/stereo_plane_stations_20261111_01` | `outputs/benchmarks/20261111_01__stereo__multi3__root_views` |
| `outputs/benchmarks/t_station_full_frame_20261110_01` | `outputs/benchmarks/20261110_01__stereo__multi3__framing` |
| `outputs/preselection/stereo_candidates_20261117` | `outputs/preselection/20261117_01__stereo__multi6__preselection` |
| `outputs/imports/industrial-welding_20261112_174825_211229` | `outputs/imports/20261112_174825_211229__scene__industrial_welding__backup` |
| `outputs/imports/scene_sync_20261112` | `outputs/imports/20261112_01__scene__industrial_welding__sync` |
| `outputs/reviews/scene_sync_20261112` | `outputs/reviews/20261112_01__scene__industrial_welding__sync_review` |
| `outputs/validation/hj90_vendor_rgb_20261112_01` | `outputs/validation/20261112_01__rgb_calibration__HJ90__vendor_capture` |
| `outputs/validation/hj90_vendor_rgb_20261112_02_native` | `outputs/validation/20261112_02__rgb_calibration__HJ90__native_capture` |
| `outputs/runs/industrial_welding_context` | `outputs/runs/undated_01__scene__industrial_welding__context_preview` |
| `outputs/runs/industrial_welding_hj90` | `outputs/runs/undated_01__scene__industrial_welding__sensor_preview` |
| `outputs/runs/ship_grid_import` | `outputs/runs/undated_01__scene__ship_grid__import_check` |

## 保留与校验规则

- 图像、标定、SAM 掩码、轨迹、原实验 JSON、source_snapshot 均保留原内容。
- archive 整组保持原路径和原内容，原归档 SHA256 清单继续有效。
- 活动 Markdown、复现命令和脚本默认输入已改为新路径；历史绝对路径由 [映射](../outputs/path_mapping.json) 解析。
- 最新八区域实验的 overview、close、coarse-retry、bead-retry、evaluation 和报告全部保留；重试目录包含最终结果所需输入。
- 新八区域实验不是 main.py 的标准 sweep 回放数据；主流程回放仍使用保留的 A25/A26 正面 sweep 基准。

[维护与验证结果](../outputs/validation/20261117_01__maintenance__outputs__cleanup/README.md)；[当前输出导航](../outputs/README.md)。

## 最终验证

63 项单元测试通过；改名后 A25/A26 实际回放均为 ok，融合 JSON 与原基准完全相同。归档 8,752 个文件、迁移保留的 1,855 个文件和源码快照均已核验，活动报告本地链接有效。当前 outputs 约 3.1 GiB。
