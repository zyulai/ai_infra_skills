# PR #67741: [Perf][Qwen4Exp] QSA top-k: default to FlashInfer gvr_2 on SM100

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gau-nernst |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel(tier: major) |
| Label | ci/build, qwen |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 .buildkite/test_areas/models_basic.yaml、tests/models/qwen4_exp/test_qsa_reference.py、vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py、vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py、vllm/models/qwen4_exp/nvidia/qsa_indexer.py |
| 链接 | https://github.com/vllm-project/vllm/pull/67741 |

## 一句话结论

Qwen4Exp QSA 索引器的 top-k 在 SM100 家族上默认改用 FlashInfer `gvr_2`(decode 与 prefill),GB300 kernel 时间长上下文 decode 提速 1.3–2.3x、256+ token 的 prefill chunk 提速 1.1–2.5x,128k 输入端到端输出吞吐 +2.4%。

## 动机

QSA top-k 原先用 cooperative/persistent 链(`torch.ops._C.cooperative_topk` 等);FlashInfer 的 `gvr_2` varlen kernel 在长上下文大 batch 时明显更快。decode 在 CUDA graph 中运行,dispatch 只看行数看不到上下文长度,因此不做行数阈值:`auto` 一律优先 `gvr_2`(短上下文下 `gvr_2` 每次调用贵 1–2 µs,但长上下文省 4–7 µs)。

## 优化点

1. `use_gvr2_topk`:SM100 家族 + block_topk ∈ (512, 1024, 2048) + logits 宽度为 4 的倍数 + FlashInfer gvr_2 可用时启用;`sparse_indexer_topk_backend` 支持 `auto` / `flashinfer_gvr2`(强制,不满足则 raise)/ `cooperative`(恢复旧链)/ `persistent`(decode 强制 persistent)。
2. decode 用 `gvr_2` varlen kernel(按精确 logits 宽度编译,QSA decode 宽度固定);prefill 用 windowed kernel(按 2 的幂宽度 bucket 编译,适配变化的 chunk 宽度),配全零 `row_starts`。
3. 启动 warmup:`warmup_gvr2_topk` 预编译 serving 能命中的所有形状(decode 按精确宽度,prefill 按 pow2 bucket);`init_gvr2` 在模型初始化时分配 prefill 零 `row_starts` buffer,serving 只做 slice。
4. `max_seq_len=width` 避免 host sync;`-1` 填充短于 block_topk 的 token。

## 关键改动

```python
# vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
+def use_gvr2_topk(topk_backend: str, block_topk: int, width: int) -> bool:
+    supported = (
+        current_platform.is_device_capability_family(100)
+        and block_topk in (512, 1024, 2048)
+        and width % 4 == 0
+        and has_flashinfer_topk_varlen_gvr2()
+    )
+    ...
```

```python
# _topk 分发
+    if use_gvr2_topk(topk_backend, block_topk, logits.shape[1]):
+        _gvr2_topk(logits, visible_blocks, block_topk, block_indices, prefill)
+        return
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- Kernel(GB300, `_topk` only, CUPTI, cold L2, decode in CUDA graphs;k=512, `max_model_len` 256k;median of 12 processes):decode long context(每请求 128k–256k tokens):rows 1: 8.0 µs → 6.2 µs (1.30×),16: 12.9 → 7.7 (1.68×),64: 17.4 → 10.1 (1.72×),256: 41.8 → 21.4 (1.96×),1024: 127.8 → 56.7 (2.25×);decode short context(4k–32k):rows 1: 0.77×,16: 0.75×,64: 0.70×,256: 1.07×,1024: 29.0 → 17.0 (1.70×);prefill long(131k chunk):128: 13.1 → 10.2 (1.28×),1024: 75.5 → 40.5 (1.86×),16384: 1031.4 → 404.9 (2.54×);prefill short(32k chunk):128: 0.81×,1024: 1.75×,16384: 2.49×。
- End to end(Qwen3.8-Flash-Next-FP8, 4×GB300 node, TEP4, fp8 KV + indexer cache, no MTP):128k 输入 c1/c4/c16 输出吞吐 +1.7% / +2.3% / **+2.4%**(P50 TTFT 1606→1558 / 2422→2181 / 2430→2361 ms);8k 输入 c1–c64 为 −0.7% / −0.4% / −0.6% / −0.2%。
- Accuracy(MTP3, temp 0):GSM8K (5-shot, n=2638) 96.89% → 97.01%;LongBench v2 (403 prompts within 262k) 37.2% → 38.2%。

## 验证与回归风险

- `pytest tests/models/qwen4_exp -k "not amd"`:260 passed, 1 skipped(GB300)。
- `test_qsa_prefill_selection_correctness` 新增 1,030-token chunk 与 140k context(`large-chunk-wide`),覆盖 `gvr_2` 的 streaming prefill engine;CI 的 B200 step(`qwen4-exp-kernels-b200`)现在也依赖 `ops/qsa_indexer.py`。
- 回归风险/限制:`gvr_2` 延迟依赖 score 分布(kernel 数字用随机 q/K);warmup 使启动最多增加约 24 s;仅在 GB300 上测量,B200 只有 CI 正确性覆盖;`flashinfer_gvr2` 强制模式在不满足条件时初始化即报错。

## 关联

相似 dispatch 逻辑参考 DeepSeek indexer;`has_flashinfer_topk_varlen_gvr2` 来自 `vllm.utils.flashinfer`;与 #65706(GLM SM107 低延迟表)同属 2026-11 的 kernel 后端选择类工作(pack 未提供更多同域 PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
