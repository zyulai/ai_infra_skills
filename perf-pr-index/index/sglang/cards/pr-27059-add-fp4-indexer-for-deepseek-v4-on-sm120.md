# PR #27059: Add FP4 Indexer for DeepSeek V4 on SM120

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @liz-badada |
| 合并 | 2026-07-24 |
| 分类 | Quantization(major) |
| Label | deepseek, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/27059 |

## 一句话结论

把 DeepSeek V4 的 FP4 Indexer 从 SM100 扩展到 SM120,消除 FP8 indexer 约 31 秒的 E2E 开销。

## 动机

本 PR 是 #26209 的后续:FP4 Indexer 原本只支持 SM100,SM120(Blackwell 桌面级)上 DeepSeek V4 的 DSA indexer 仍走 Torch FP8 路径。在较新的软件栈下,SM120 主注意力路径已大幅提速,FP8 indexer 约 31 秒的固定开销成为 E2E 的主导瓶颈,因此需要把 DeepGEMM FP4 indexer(依赖 sgl-project/DeepGEMM#56 / #55)带到 SM120。

## 优化点

机制上分三层。第一,放宽服务端校验:`server_args.py` 把 `--enable-deepseek-v4-fp4-indexer` 的硬件约束从"仅 SM100"改为"SM100 或 SM120"。第二,修正 metadata 规划:SM120 FP4 kernel 的 split_kv 调度为 128,而通用 JIT metadata planner 编码为 256,因此给 `PagedIndexerMetadata` 新增 `force_deep_gemm_metadata` 开关,在该开关生效时绕过 Torch fallback 和 JIT indexer metadata 分支,强制走 DeepGEMM 的 `get_paged_mqa_logits_metadata`;`copy_` 的等价性校验字段也加入该开关,避免 CUDA graph 复用时元数据错配。第三,backend 在 `init_forward_metadata_indexer` 中按 `enable_deepseek_v4_fp4_indexer and _is_sm120` 触发上述强制路径。

## 关键改动

```text
python/sglang/srt/server_args.py: FP4 indexer 硬件校验放宽到 SM100 或 SM120
python/sglang/srt/layers/attention/dsv4/metadata.py: 新增 force_deep_gemm_metadata 字段,
  强制走 DeepGEMM paged mqa logits metadata
python/sglang/srt/layers/attention/deepseek_v4_backend.py: SM120 FP4 时传入
  force_deep_gemm_metadata=True
test/registered/kernels/benchmark/attention/bench_dsv4_fp4_indexer.py: benchmark
  支持 SM120(暂标 disabled until DeepGEMM is ready)
test/registered/unit/layers/test_dsv4_nonpaged_indexer.py: 新增 metadata 单测
```

## 性能声明

两组结果均在 4x SM120、TP4、DeepSeek-V4-Flash、8 prompts(8192 输入/256 输出)、并发 8、seed 3、chunked prefill 4096、Marlin MoE、禁用 radix cache 下测得,均为同栈内 FP8 vs FP4 indexer 对比,跨表绝对值不可比:

- 最新验证(FlashInfer SM120 注意力,SGLang `245a47444c`):Request throughput 0.155 -> 0.389 req/s(+150.83%),Mean E2E 51.447 s -> 20.533 s(-60.09%),Mean TTFT -27.43%,Mean TPOT 150.981 ms -> 43.678 ms(-71.07%)。
- 原始测量(Triton SM120 注意力,SGLang `5453e93c50`):Input throughput +7.45%,Mean E2E 440.97 s -> 410.49 s(-6.91%),Mean TPOT -13.19%。

PR 明确指出:FP8 indexer 在两组栈中都固定增加约 31 秒,百分比差异来自 FP4 侧周边路径变快后 FP8 开销占比放大,不应把跨栈加速全部归因于 FlashInfer。

## 验证与回归风险

新增单测 `TestDSV4PagedIndexerMetadata`(验证 SM120 FP4 强制 DeepGEMM metadata、FP8 torch fallback 保持 None);kernel benchmark 已注册 CI 但标注"Temporarily disabled until DeepGEMM is ready"。精度:DeepSeek V4 Flash GSM8K,FP8 indexer 95.5% vs FP4 indexer 93.0%。风险集中在 `PagedIndexerMetadata` 元数据规划:`force_deep_gemm_metadata` 改变了 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`/JIT metadata 分支的优先级,`copy_` 的 check_eq_fields 变化可能影响 CUDA graph 重放复用逻辑;依赖外部 DeepGEMM PR #55,DeepGEMM 未就绪时 SM120 路径不可用。

## 关联

前序:#26209(SM100 FP4 Indexer)。同区域:PR #26788(DSA indexer 优化)、PR #24584 / #24930 / #26722(DeepSeek V4 系列集成)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
