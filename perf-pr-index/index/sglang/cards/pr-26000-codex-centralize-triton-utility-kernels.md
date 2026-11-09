# PR #26000: [codex] Centralize Triton utility kernels

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-06-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);25 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/26000 |

## 一句话结论

纯代码搬迁:把内嵌在 attention/投机解码/mem-cache 大文件中的 Triton 工具 kernel 集中到领域专属 triton_ops 模块,不改 kernel 行为。

## 动机

多个非 diffusion 的 Triton kernel 内嵌在大型 backend/helper 文件里(如 `flashattention_backend.py` 内嵌 FA metadata、SWA spec page table kernel,`triton_backend.py` 内嵌 KV split kernel),多 backend 间存在重复实现,维护与复用困难。PR 以纯 relocation cleanup 方式集中收拢,不改变 kernel 行为。

## 优化点

按领域搬迁:(1) FlashInfer/FlashMLA KV index、KV split、FlashAttention metadata、cache、padding、RoPE/cache、position kernel → `attention/triton_ops/*` 与 `model_executor/triton_ops/*`;(2) logits/elementwise softcap → `layers/triton_ops/softcap.py`(保留 in-place 与 out-of-place 两种 wrapper 语义);(3) 投机解码 cache-location、EAGLE、multi-layer EAGLE kernel → `speculative/triton_ops/*`;(4) mem-cache MLA buffer、req-to-token/last-loc、allocator、KV-copy → `mem_cache/triton_ops/*`。旧模块保留 legacy import 以兼容既有引用,多 backend 共享的 kernel 做去重。从保留 diff 可见:`flashattention_backend.py` 移出 `_prepare_swa_spec_page_table_kernel`、`_fused_metadata_kernel_general/_ps1_no_swa`(`normal_decode_set_metadata`),`triton_backend.py` 移出 `get_num_kv_splits_triton`;新建 `attention/triton_ops/kv_indices.py`(create_flashinfer/create_flashmla kv indices)、`metadata.py`、`cache_ops.py`(concat_and_cast_mha_k、reshape_and_cache_flash)。

## 关键改动

```text
python/sglang/srt/layers/attention/triton_ops/metadata.py: 新文件,fused decode metadata / SWA spec page table / KV splits kernel
python/sglang/srt/layers/attention/triton_ops/kv_indices.py: 新文件,FlashInfer/FlashMLA KV indices kernel
python/sglang/srt/layers/attention/triton_ops/cache_ops.py: 新文件,MHA K concat+cast、paged KV cache 写入 kernel
python/sglang/srt/layers/attention/flashattention_backend.py: 移除内嵌 kernel,改 import triton_ops
python/sglang/srt/layers/attention/triton_backend.py: 移除 get_num_kv_splits_triton 等内嵌实现
```

## 性能声明

本 PR 为纯搬迁,未提供新 benchmark,PR 未注明测试条件。被搬迁代码自身 docstring 保留历史声明(非本 PR 产生):fused metadata kernel 将 4-5 个串行 CUDA kernel 合并为 1-2 个,"Achieves ~5.2x speedup on H200 hardware for typical decode workloads"。

## 验证与回归风险

验证为静态检查:ruff check、compileall、`git diff --check`、新 Triton 模块 smoke import;作者注明其 macOS 环境的 openai/transformers 版本偏旧,legacy 路径完整 import 未能在本地全量验证;CI Extra 通过,Base 未执行(no_entry_sign)。风险:触面广(25+ 文件未在 pack diff 中保留),legacy import shim 若有遗漏会破坏下游引用;"kernel 行为不变"依赖 review 保证,无逐 kernel 数值回归。

## 关联

被搬迁 kernel 的相关优化:#21104(FA3 scheduler metadata 预计算)、#14781(NSA backend metadata 优化);同区域元数据类 kernel 后续演进见 #36568(跳过冗余 scheduler metadata 生成)。
