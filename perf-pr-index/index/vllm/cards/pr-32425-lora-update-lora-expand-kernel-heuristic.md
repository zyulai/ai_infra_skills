# PR #32425: [LoRA] Update LoRA expand kernel heuristic

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-01-16 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 未标注 dropped 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/32425 |

## 一句话结论

调小 LoRA expand 的 `block_n` 增加 SM 并发，内核约快 30%，e2e 输出吞吐 +1%。

## 动机

PR 正文说明，profiling 时发现 `qkv_proj` 的 expand kernel 比 shrink kernel 更耗时。原因是像 `QKVParallelLinearWithLoRA` 这类层 `num_slices > 1`，每个 slice 的 N（output_size）更小；因此可以调小 `block_n`，以启动更多 block、提高 SM 并发度。这是一个针对 LoRA expand 内核启发式配置的调优，正文明确其目的是更新 heuristic，未涉及其他路径。

## 优化点

改动集中在 LoRA Triton 内核配置的默认 `block_n` 取值：原先固定 128，改为 `max(64, next_power_of_2(128 // num_slices))`。当 `num_slices > 1` 时新值更小（下限 64），从而让每个 slice 的较小输出维度匹配更小的 tile、启动更多 block 提升 SM 并发；`block_m=64`、`block_k=16`、`num_warps=4`、`num_ctas=1` 保持不变。正文提到"With tuned config, it can achieve the 2x more performance gain"，但本次落地的是默认启发式调整，非逐项 tuned config。shrink kernel 不在本次改动范围。

## 关键改动

```text
vllm/lora/ops/triton_ops/utils.py: get_lora_op_configs 的 default 配置把 block_n 由 128 改为按 num_slices 缩放
```

```python
     else:
         default = {
             "block_m": 64,
-            "block_n": 128,
+            "block_n": max(64, next_power_of_2(128 // num_slices)),
             "block_k": 16,
             "num_warps": 4,
             "num_ctas": 1,
```

## 性能声明

PR 正文声明：profiling 显示 `_lora_expand_kernel` 提升约 30%（"improves about 30%"，并称 tuned config 可达 2x），Main 约 14µs、PR 约 10µs；e2e output token throughput improves 1%。基准条件（正文 Benchmarking 段）：`vllm serve openai/gpt-oss-20b`，`--tensor-parallel-size 1`、`--max-num-seqs 16`、`--compilation-config '{"compile_sizes": [1, 2, 4, 8, 16]}'`、`--enable-lora`、`--max-loras 1`、`--lora-modules lora1=VaarunC/gpt-oss-20b-lora`、`--max-lora-rank 16`、`--no-enable-prefix-caching`；`vllm bench serve` 用 sharegpt 数据集、`--sharegpt-output-len 800`、`--max-concurrency 16`、`--num-prompts 1000`、`--num-warmups 20`、`--ignore-eos`。主/PR 对比（1000 requests、800000 generated tokens）：

| 指标 | Main | PR |
| --- | --- | --- |
| Benchmark duration (s) | 438.70 | 434.72 |
| Request throughput (req/s) | 2.28 | 2.30 |
| Output token throughput (tok/s) | 1823.57 | 1840.27 |
| Peak output token throughput (tok/s) | 2016.00 | 2032.00 |
| Total token throughput (tok/s) | 2340.53 | 2361.97 |
| Mean TTFT (ms) | 91.53 | 108.03 |
| Mean TPOT (ms) | 8.61 | 8.51 |
| Mean ITL (ms) | 8.61 | 8.51 |

硬件型号与精度 PR 未注明测试条件。

## 验证与回归风险

Test Plan 为 `pytest -s -v tests/lora/test_punica_ops.py`，Test Result 写"Unit tests passed."；正文另附 Main/PR 两帧 profiling 截图（14µs / 10µs）与完整 serving benchmark 输出。可能影响的路径：所有走 `get_lora_op_configs` 默认配置的 LoRA expand 内核，尤其是 `QKVParallelLinearWithLoRA` 等 `num_slices > 1` 的层；`block_n` 下限 64 保证不会过小。对比数据中 Mean TTFT 由 91.53ms 升到 108.03ms，而 TPOT/ITL 略降，属需关注项；正文所述 tuned config 的 2x 收益未在 e2e 基准中体现。其余回归信息 PR 未注明。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
