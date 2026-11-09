# PR #19825: [Core] Add Flashinfer TRTLLM Backend for Flashinfer decode path (SM100).

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pavanimajety |
| 合并 | 2025-07-11 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 2 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/19825 |

## 一句话结论

SM100 decode 启用 TRTLLM 后端，Llama-3.3-70B-FP8 V1 吞吐 5.20→5.58 req/s。

## 动机

PR 为 Paged GQA 在 `kv-cache-dtype="auto"` 下新增 decode kernels（FA3 风格的 Q=FP8/KV=FP8 支持留待后续 PR）。正文在 checklist 中引用关联 issue #18153，并把「对比集成前后性能、lm_eval 精度、kernel 级精度与基准」列为测试计划。该 PR 由 @wenscarl 共同署名；文档更新部分说明新增的环境变量用于在 flashinfer 的 `BatchDecodePagedKVCacheWrapper` 与 `trtllm_batch_decode_with_kv_cache` API 之间切换。

## 优化点

在 flashinfer decode 路径引入 TRTLLM 后端，用 `trtllm_batch_decode_with_kv_cache` API 与 flashinfer `BatchDecodePagedKVCacheWrapper` 之间切换。新增环境变量 `VLLM_USE_TRTLLM_DECODE_ATTENTION` 控制；`use_trtllm_attention` 在 env 未设时按 SM100a 支持、`attn_head_size==128`、`batch_size<=256`、`max_seq_len<131072`、`kv_cache_dtype=="auto"` 自动检测启用。`arg_utils.py` 中当启用该 env 时对 kv-cache-dtype 的 FP8 能力检查放行；`platforms/cuda.py` 提供 SM100a 能力判断。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: 新增 use_trtllm_attention 判定与 TRTLLM decode 调用
vllm/envs.py: 新增 VLLM_USE_TRTLLM_DECODE_ATTENTION 环境变量
vllm/engine/arg_utils.py: 启用该后端时放行 kv-cache-dtype 检查
vllm/platforms/cuda.py: SM100a 能力检测
benchmarks/kernels/benchmark_trtllm_attention.py、tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py: 基准与测试
```

```diff
+    "VLLM_USE_TRTLLM_DECODE_ATTENTION":
+    lambda: os.getenv("VLLM_USE_TRTLLM_DECODE_ATTENTION", None),
...
+            use_trtllm = (FlashInferBackend.cached_sm100a_supported
+                          and batch_size <= 256 and max_seq_len < 131072
+                          and kv_cache_dtype == "auto")
```

## 性能声明

Llama 3.3 70B FP8，`vllm bench throughput --model nvidia/Llama-3.3-70B-Instruct-FP8 --input-len 100 --output-len 1000 --num-prompts 500 --quantization modelopt`。Before PR：V0 5.21 req/s、5731.91 total tok/s、5212.07 output tok/s；V1 5.20 req/s、5717.03 total tok/s、5198.54 output tok/s。After PR：V0 5.55 req/s、6104.40 total tok/s、5550.78 output tok/s；V1 5.58 req/s、6139.94 total tok/s、5583.09 output tok/s。测试条件：模型 Llama-3.3-70B-Instruct-FP8、精度 FP8（modelopt）、workload input 100/output 1000/500 prompts；硬件为标题标注的 SM100（正文提及 Blackwell），PR 未注明具体 GPU 型号。

## 验证与回归风险

`tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py`：24 passed, 1 warning in 72.65s。测试计划含 baseline/集成后性能对比、lm_eval 精度、kernel 级精度与基准测试，正文未给出 lm_eval 数值。回归风险：当前仅覆盖 `kv-cache-dtype="auto"` 的 decode 路径，FP8 KV 支持留待后续 PR；自动启用被 SM100a、head_size 128、batch<=256、seq<131072 等条件约束，不满足时回落到 flashinfer wrapper。

## 关联

PR 正文未给出明确的前序/后续 PR 编号；仅在 checklist 引用关联 issue #18153，并提到后续 PR 会加入 Q=FP8/KV=FP8 支持（未给编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
