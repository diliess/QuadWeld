# 本地资产

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| [scenes/](scenes/README.md) | 场景包：拼板、开口盒子两种工件，以及船板格架环境 | 共同约定、来源与重建方法 |
| [robots/rm65/](robots/rm65/README.md) | 转换后可复用的 RM65-6F 机器人 USD | 转换、来源与接入方式 |
| [robots/b2_e10l/](robots/b2_e10l/README.md) | B2 机器狗背负 E10L-Pro 焊接臂的整机 USD（浮动底座） | 安装尺寸、假设值与构建方法 |
| `vendor/` | 厂商提供的 URDF 和网格源文件，已加入 `.gitignore` | 获取方法见各机器人说明 |
| [weld_images/](weld_images/README.md) | SAM3 分割实验的 12 张原图及点提示、来源表 | 来源与标注格式 |

仿真图像、轨迹和执行报告不放在这里，仍保存在项目根目录的 `outputs/`。
