# PR #18211: [CI] Fix torchaudio/torchvision CUDA version mismatch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#18211](https://github.com/sgl-project/sglang/pull/18211) |
| 作者 | alisonshao |
| 合并日期 | 2026-02-12 |
| labels | high priority, dependencies |
| 热度 | 👍 0 · 💬 49 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
torch 与 torchaudio/torchvision 的 CUDA 版本不一致(cu128 vs cu129)导致循环导入报错,CI 安装脚本检测到不匹配后从匹配索引强制重装对齐。

## 问题现象
CI 中加载需要 torchaudio 的多模态 processor 时报错:`AttributeError: partially initialized module 'torchaudio' has no attribute 'lib' (most likely due to a circular import)`。作者给出了 CI 失败示例:https://github.com/sgl-project/sglang/actions/runs/21652299684/job/62419467816。受影响面为依赖 torchaudio 的 VLM 测试(如 TestMiniCPMo26Server),即 CUDA CI 依赖安装环境。

## 根因
正文披露:已有的 `[tool.uv.sources]` 配置只对 `torch` 指定了 source override,没有对 `torchaudio` 和 `torchvision` 做同样处理,导致安装来源不一致——`torch` 从 PyPI 解析(cu128),而 `torchaudio`/`torchvision` 从 extra-index-url 解析(cu129)。CUDA 版本不匹配使 torchaudio 的 C extension 加载失败,进而表现为上述循环导入错误。具体到版本:PyPI 的 torch 2.9.1 自带 cu128,而 pytorch.org/cu129 的 torchaudio 用的是 cu129。正文还说明:不能把 torch 换成 cu129,否则会破坏 sgl_kernel 的 ABI,因此只能反向把 torchaudio/torchvision 对齐到 torch 的 CUDA 版本。

## 修复方案
改动在 `scripts/ci/cuda/ci_install_dependency.sh`(diff 中可见部分;正文 Summary 还提到在 `pyproject.toml` 的 `[tool.uv.sources]` 中加入 torchaudio/torchvision,该文件改动未包含在 pack 的 diff 摘录中)。安装完 python extras 后,用 `torch.version.cuda` 检测实际 CUDA 版本并与 `CU_VERSION` 比较,不一致时按 pyproject 安装的版本号(去掉 `+cuXYZ` 后缀)从匹配的索引强制重装:

```bash
TORCH_CUDA_VER=$(python3 -c "import torch; v=torch.version.cuda; parts=v.split('.'); print(f'cu{parts[0]}{parts[1]}')")
if [ "${TORCH_CUDA_VER}" != "${CU_VERSION}" ]; then
    $PIP_CMD install "torchaudio==${TORCHAUDIO_VER}" "torchvision==${TORCHVISION_VER}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VER}" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
fi
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 CUDA CI 的依赖安装脚本路径;`--force-reinstall --no-deps` 重装 torchaudio/torchvision 时跳过依赖解析,理论上可能与 uv/pip 已装依赖产生版本漂移;检测逻辑依赖 `torch.version.cuda` 可解析,异常格式会导致比较失败路径行为变化。

## 关联
正文未列出 issue/PR 链接(仅给出 CI failure run 链接作为复现示例)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
