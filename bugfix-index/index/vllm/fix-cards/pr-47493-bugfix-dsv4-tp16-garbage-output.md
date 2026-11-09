# PR #47493: [Bugfix] DSV4 TP16 garbage output

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47493](https://github.com/vllm-project/vllm/pull/47493) |
| 作者 | majunze2001 |
| 合并日期 | 2026-07-08 |
| labels | bug, ready, v1, nvidia |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
#44577 packed KV + FlashInfer sparse-MLA 按 flat token id 读地址导致 DSV4 TP16 输出乱码(gsm8k ~0%);索引重映射 + 512B 页对齐修复,恢复 ~0.95。

## 问题现象
DeepSeek-V4 在 TP16(GB300/SM100,4 节点,fp8 KV cache,`FLASHINFER_MLA_SPARSE_DSV4` 后端,spec-decode off)下输出乱码,gsm8k 约 0%(issue #47783)。修复前 HEAD gsm8k 约 0.0;修复后 flexible-extract 0.9492 / strict-match 0.9500(5-shot),与 #44577 之前基线 ~0.9484 在 stderr 内一致。

## 根因
PR 正文披露:#44577 把一个 block 的 KV 组件(indexer + compressor + SWA)打包进一个连续的 per-block 分配,使每层的 per-block stride 等于整个 block 大小;而 FlashInfer sparse-MLA decode 按 **flat global token id** 读池(`addr = base + slot * token_stride`),忽略张量的 block stride,token 跨过 block 边界后就读到错误地址,造成 attention 损坏(gsm8k ~95% → ~0%)。FlashMLA 不受影响(它遵循运行时 block stride)。

## 修复方案
保留 packed 布局(为 KV-transfer/offload 保持 per-block KV 连续性),只修读侧,共 5 个模型本地文件、+61/-6:
1. 索引重映射:`models/deepseek_v4/common/ops/cache_utils.py` 新增 Triton device 函数 `_remap_flashinfer_index`,把全局槽位 `block*block_size + off` 重写为 `block*block_span + off`(`block_span = page_stride / token_stride`),在既有 index-build kernel 的 4 处槽位写入点应用,无额外 kernel launch;`models/deepseek_v4/nvidia/flashinfer_sparse.py` 新增 `_packed_block_span(pool)`(= `stride(0)//stride(-2)`,unpacked 布局下 == block_size,故为 no-op;不对齐则 raise NotImplementedError)。
2. 页对齐:`attention.py`、`compressor.py`、`sparse_swa.py` 中 `MLAAttentionSpec.alignment` 改为 `576 if fp8_ds_mla else 512`,使 packed block stride 是 512B 压缩 KV token stride 的整数倍。
两者均为 workaround,直到 flashinfer-ai/flashinfer#3856 给 sparse-MLA kernel 增加真正的 block-stride 参数。

```diff
+@triton.jit
+def _remap_flashinfer_index(values, block_size, block_span):
+    is_valid = values >= 0
+    safe_values = tl.where(is_valid, values, 0)
+    values = (safe_values // block_size) * block_span
+    values += safe_values % block_size
+    return tl.where(is_valid, values, -1)
```

## 影响与回归风险
- 影响窗口:自 #44577 引入 packed KV 起,使用 `FLASHINFER_MLA_SPARSE_DSV4` 后端的 DeepSeek-V4 均受影响。
- 回归面:仅 DSv4/FlashInfer sparse 相关的模型本地文件(不动 `kv_cache_utils.py`、`backend.py`、`kv_cache_interface.py`);unpacked KV 布局下重映射为 no-op;512B 对齐使 KV cache 容量 22.70M tokens(比未 padding 布局低约 0.03%);packed KV 要求 per-block stride 必须是 per-token stride 的整数倍,否则抛 NotImplementedError。`pytest tests/v1/core/test_contiguous_kv_packing.py` 8 passed。

## 关联
- Issue #47783(gsm8k ~0% 报告)
- PR #44577(packed KV 回归引入者)
- PR #47895(重映射方案来源;本 PR 是改用 per-page alignment 字段的更低侵入版本)
- PR #47805(另一方案:为 FlashInfer 放弃 packed KV,目标不同)
- [flashinfer-ai/flashinfer#3856](https://github.com/flashinfer-ai/flashinfer/issues/3856)(上游长期修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
