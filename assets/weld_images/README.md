# 焊缝 RGB 样本

本目录保存 12 张 RGB 原图，`sources.csv` 记录来源和可见焊接状态。
这些记录用于样本追溯，不作为算法输入；程序不读取文件名中的焊接状态。

从项目根目录运行：

```bash
python scripts/auto_weld.py assets/weld_images
```

程序自动生成 DA3 深度、SAM 板件与焊道掩码，并导出三维接缝和焊趾候选。
结果默认位于 `outputs/offline/depth_anything_3/`。
