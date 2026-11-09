# PR #39752: add warning when FP8 KV cache misses prefill query quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qiching |
| 合并 | 2026-04-14 |
| 分类 | Quantization \| tier: major |
| Label | ready, deepseek, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/39752 |

## 一句话结论

为 FP8 KV cache 补启动警告提示开启 prefill FP8；GB300/128K profiling 快 1.28x。

## 动机

正文背景：按 PR #31195（@pavanimajety），FP8 prefill query quantization 因在短序列上回退性能（ISL=1024 约 -20%）而被有意用 `use_prefill_query_quantization=true` 门控；但在长上下文（ISL >= 4K）下 FP8 FMHA kernel 明显更快（128K 下 up to 1.4x）。当前用户启用 `--kv-cache-dtype fp8` 时会静默得到 BF16 prefill attention，无从得知存在更快路径。本 PR 用于消除这一信息缺口，Resolves #39751。

## 优化点

在 `determine_prefill_query_data_type()` 中加入 `warning_once` 日志：当 FP8 KV cache 激活、backend 支持 FP8 prefill、而用户未启用该选项时打印警告，警告中给出确切的 `--attention-config '{"use_prefill_query_quantization": true}'` 用法。正文强调 "no default behavior is changed"，即本 PR 只加提示、不改变任何默认计算路径。

## 关键改动

```text
vllm/model_executor/layers/attention/mla_attention.py: 在 determine_prefill_query_data_type() 中加 warning_once，提示开启 FP8 prefill query quantization 的 --attention-config 参数
```

## 性能声明

本 PR 仅新增日志、不改变默认行为，PR 未给出本 PR 自身的定量性能提升；正文所列数字是"用户开启 flag 后可得收益 / 回归"的背景对比。Profiling on GB300（DeepSeek-R1-FP4，DP=4，ISL=128K）：`fmhaSm103aKernel_QkvBfloat16`（default）287 ms、`fmhaSm103aKernel_QkvE4m3`（with flag）225 ms，Speedup 1.28x。Short-Seq e2e benchmark（GB300，DeepSeek-R1-0528-FP4，DP=4，`max-model-len=34816`，fp8 KV cache）：ISL=1024 BF16 15,803 tok/s、FP8 12,562 tok/s（79.5%），Mean TTFT 24.8 s → 38.2 s，FP8 仍回退约 20%；ISL=4096 BF16 80,861 tok/s、FP8 83,799 tok/s（103.6%），Mean TTFT 5.25 s → 5.16 s，FP8 约 3.6% faster。

## 验证与回归风险

Test Plan 为三种启动场景的手工验证：仅 `--kv-cache-dtype fp8` 时警告出现（含给定 WARNING 文本）；加上 flag 时无警告且出现 `INFO: FP8 prefill attention enabled: query data type is FP8`；不启用 FP8 KV cache 时无警告。PR 未注明引入或依赖的自动化测试。默认行为未变，正文未给出其他模型、硬件或精度的回归评估。

## 关联

Resolves #39751；背景引用 PR #31195（@pavanimajety）中 FP8 prefill query quantization 的门控依据。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
