# PR #15601: Fix BatchMLAPagedAttentionWrapper query/qo_inptr mismatch for EAGLE

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15601](https://github.com/sgl-project/sglang/pull/15601) |
| 作者 | wenscarl |
| 合并日期 | 2026-02-27 |
| labels | blackwell, bug, run-ci |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
TRTLLM MLA 后端实例跨 forward 复用,target-verify/draft-extend 时残留上一次 extend 的 forward_prefill_metadata,qo_indptr 与 query 不匹配触发非法内存访问;修复为切换模式时将其置 None。

## 问题现象
当 `BatchMLAPagedAttentionWrapper` 的 query batch 维度与 `qo_indptr[-1]` 不匹配时(同 FlashInfer issue #2236 描述),CUDA kernel 报错崩溃:BatchMLA kernel 内 `Check failed: (status == cudaSuccess) is false: Failed to run MLA, error: an illegal memory access was encountered`。复现配置为 DeepSeek-R1-0528-FP4-v2 + `--attention-backend trtllm_mla` + `--speculative-algorithm EAGLE`(trtllm_mla 后端)的组合,即 EAGLE speculative decoding 走 TRTLLM MLA decode 路径时。flashinfer 后端下同基准 Accuracy 0.961 / Invalid 0.000,trtllm_mla 修复后 Accuracy 0.959 / Invalid 0.000。

## 根因
正文披露:forward_metadata(包括 `qo_indptr`)没有在 TRTLLM MLA 后端中被更新,执行时使用了不一致的元数据。由于后端实例跨多个 forward pass 持久存在,之前一次普通 extend 调用设置的 `forward_prefill_metadata` 可能仍残留;在 target-verify / draft-extend 等 decode 型模式下使用时,query 与 `qo_indptr` 不匹配,触发非法内存访问。

## 修复方案
修改 `python/sglang/srt/layers/attention/trtllm_mla_backend.py` 的 `init_forward_metadata`:当 forward 模式为 target-verify 或 draft-extend(含 v2)时,把残留的 `self.forward_prefill_metadata` 置为 `None`,避免上一轮 extend 的元数据被沿用。关键 diff:

```python
+            # This is necessary because the backend instance persists across forward passes,
+            # and forward_prefill_metadata from a previous regular extend call could still be set.
+            if (
+                forward_batch.forward_mode.is_target_verify()
+                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
+            ):
+                self.forward_prefill_metadata = None
```

该分支位于新建 `TRTLLMMLADecodeMetadata` 的 decode 元数据初始化路径中。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅影响 `trtllm_mla` 后端的 `init_forward_metadata`:在 target-verify / draft-extend(含 v2)模式下每次都会把 `forward_prefill_metadata` 清空;后续代码若仍期望在该模式下读取 prefill 元数据,会走 None 分支重建。普通 extend / decode 模式行为不变。

## 关联
- [flashinfer-ai/flashinfer#2236](https://github.com/flashinfer-ai/flashinfer/issues/2236)(query 与 qo_indptr[-1] 不匹配导致非法内存访问)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
