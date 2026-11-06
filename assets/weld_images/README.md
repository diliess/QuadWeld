# 焊缝图片

本目录集中保存 SAM3 分割实验的 12 张原图，上传图和网上样本直接放在同一层。
`sources.csv` 记录来源、可见焊接状态和原文件路径；`point_prompts.csv` 保存点提示，
坐标使用原图像素，`label=1` 为正点、`label=0` 为负点。重复的已焊对照图合并为
`welded_front_01.png`。结果集中在 [outputs/offline/sam3/](../../outputs/offline/sam3/)，
实验说明见 [项目 README](../../README.md#sam3-分割实验)。
