# 第三方代码

第三方项目保留独立 Git 仓库；主项目通过 `.gitignore` 排除其源码目录，仅记录来源和检出的版本。

## SAM 3 / SAM 3.1

- 本地目录：`thirdparty/sam3/`
- 官方仓库：[facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- SSH 地址：`git@github.com:facebookresearch/sam3.git`
- 克隆日期：2026-09-18
- 分支：`main`
- 克隆时提交：`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`
- 模型发布说明：[SAM 3.1](sam3/RELEASE_SAM3p1.md)
- 上游许可证：[LICENSE](sam3/LICENSE)

使用 SSH 浅克隆获取当时最新版本的全部仓库文件，未安装 Python 依赖或下载模型权重。

从项目根目录重新克隆最新 `main`：

```bash
git clone --depth 1 git@github.com:facebookresearch/sam3.git thirdparty/sam3
```

如需恢复到本次记录的提交，在克隆后执行：

```bash
git -C thirdparty/sam3 fetch --depth 1 origin 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
git -C thirdparty/sam3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
```

安装及权重获取方式见上游 [README](sam3/README.md)。
