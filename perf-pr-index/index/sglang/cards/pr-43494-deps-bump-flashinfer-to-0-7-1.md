# PR #43494: [Deps] Bump FlashInfer to 0.7.1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-10-12 |
| 分类 | Attention / Kernel(classify.py 给出,tier: major) |
| Label | documentation, dependencies, run-ci, diffusion, run-ci-extra, release-highlight, bypass-fail-fast, parallel-stages, max-concurrency |
| Diff | diffstat 缺失(pack 未提供 +X/-X 统计);diff 显示 6 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/43494 |

## 一句话结论

将 FlashInfer 全家桶(python/cubin/JIT-cache)从 0.7.0.post1 升级到 0.7.1,让 SM120 SubBlock 稀疏注意力不再需要源码安装。

## 动机

FlashInfer 0.7.1 发布,包含 SM120 所需的 flashinfer-ai/flashinfer#4691(`bsa_attn_sm120_blk64_sage_fwd` 与 `quantize_sage_qkv_sm120`):此前默认 pin 0.7.0.post1 不含这两个 API,SGLang 文档要求用户额外从源码 revision 安装 FlashInfer,升级后即可移除该步骤。同时修复版本 bump 脚本因引用已删除文件而验证失败的问题。

## 优化点

依赖对齐加工具链维护:四处 pin 同步从 0.7.0.post1 改为 0.7.1(`docker/Dockerfile`、`docker/Dockerfile.cu134`、`pyproject.toml` 的 `flashinfer_python[cu13]==0.7.1`、`engine.py` 运行时断言);`scripts/release/bump_flashinfer_version.py` 的 FILES_TO_UPDATE 加入 `docker/Dockerfile.cu134`(上次靠手改)、移除 `python/sglang/srt/utils/common.py`(其版本化 docstring 示例在 #40709 中删除,曾导致脚本验证失败),Dockerfile 匹配逻辑改为前缀匹配以覆盖 cu134 变体。SubBlock sparse attention README 删除 SM120 Sage 源码安装步骤。0.7.1 升级公告点名的 API(`cute_dsl_fused_moe_bf16`、`CuteDslBf16MoEWrapper`、`moe_a2a_dispatch`、fp8/fp4 paged mqa logits、被移除的 PrimTS decode 入口)SGLang 均未使用,依赖下限(nvidia-cutlass-dsl[cu13]==4.8.0、apache-tvm-ffi==0.1.11、cuda-tile==1.6.0rc5)已由现有 pin 满足。

## 关键改动

```text
docker/Dockerfile 与 Dockerfile.cu134: FLASHINFER_VERSION 0.7.0.post1 -> 0.7.1
python/pyproject.toml: flashinfer_python[cu13]==0.7.1
python/sglang/srt/entrypoints/engine.py: flashinfer_python 版本断言改为 0.7.1
scripts/release/bump_flashinfer_version.py: pin 文件清单修正(cu134 加入、common.py 移除)
python/sglang/multimodal_gen/.../subblock_sparse/README.md: 删除 SM120 源码安装步骤
```

```python
FILES_TO_UPDATE = [
    Path("python/pyproject.toml"),
    Path("docker/Dockerfile"),
    Path("docker/Dockerfile.cu134"),
    Path("python/sglang/srt/entrypoints/engine.py"),
]
```

## 性能声明

PR 正文未提供 benchmark 或性能对比数据,未注明测试条件。

## 验证与回归风险

作者给出的验证:`bump_flashinfer_version.py 0.7.1` 更新并验证全部 4 个 pin 文件;`uv pip compile` 解析到 `flashinfer-python==0.7.1`;cubin/jit-cache 0.7.1 已在 flashinfer.ai/whl 发布(cu129/cu130、x86_64/aarch64);0.7.1 wheel 含 subblock_sparse_attn.py 所需的两个符号。GPU E2E 未本地运行,依赖 CI(CI 快照 Base/Extra 通过、AMD ROCm 10 失败)。风险:运行时断言强制 0.7.1,旧环境启动失败需重装;`nvidia-cudnn-frontend>=1.30.0` 经传递依赖解析到 1.31.0,间接引入版本浮动。

## 关联

同区域 FlashInfer bump 链:#36954(0.6.18)→ 本 PR(0.7.1)→ #44591(0.7.2);0.7.1 pin 为 MegaMoE NVFP4 W4A16/SiTU 支持(PR #44036)的依赖前提(含 FlashInfer #5019)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
