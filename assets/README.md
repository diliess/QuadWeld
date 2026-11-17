# 场景与历史输入样本

[`scenes/industrial-welding/`](scenes/industrial-welding/README.md) 保存工业焊接厂房、机器人、
HJ-90 及拍摄姿态，可直接在 Isaac Sim 中打开完整场景 USD。当前双目流程从该场景采集同步左右目 RGB。

[`weld_images/`](weld_images) 保存历史单张 RGB 样本及来源，供追溯和人工查看。
当前识别流程使用双目图像与标定，不能从这些单张图片恢复标定双目测量结果。

SAM3 源码和权重部署在 `thirdparty/`，说明见 [模型部署](../thirdparty/README.md)。
