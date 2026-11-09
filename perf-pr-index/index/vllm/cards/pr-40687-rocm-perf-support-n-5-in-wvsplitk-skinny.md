# PR #40687: [ROCm][Perf] Support N=5 in wvSplitK skinny GEMM kernels for speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-05-28 |
| 分类 | Speculative Decoding（tier: major） |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾未见 "N file(s) dropped from diff" 标注，diff 段含 2 个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/40687 |

## 一句话结论

ROCm wvSplitK 支持 N=5，使 eagle3（4 spec tokens）验证走快速 HIP kernel，Strix Halo / Qwen3-8B 下 TPOT 快 12.3%、decode 吞吐快 14.2%。

## 动机

正文：使用 speculative decoding（如 eagle3 且 `num_speculative_tokens=4`）时，target model 的 verification pass 的 batch size 为 5（1 个原始 + 4 个 speculative tokens）。而 skinny GEMM 的 wvSplitK kernel 只支持 `N<=4`，导致 verification step 回落到 `torch.nn.functional.linear`（hipBLAS）。

## 优化点

在 `csrc/rocm/skinny_gemms.cu` 的 wvSplitK dispatch switch 中新增 `case 5`，按 `use_wave32` 选择 `WVSPLIT_TILE_CFG(32, 16, sYT, 5)` 或 `WVSPLIT_TILE_CFG(64, 16, sYT, 5)`；同时把 `vllm/model_executor/layers/utils.py` 中 `rocm_unquantized_gemm_impl` 的 skinny 分支阈值从 `m > 8 and 0 < n <= 4` 放宽到 `m > 8 and 0 < n <= 5`，使 speculative verification 命中快速 HIP kernel 路径而非 hipBLAS。diff 仅新增 `case 5` 与放宽该阈值，其余分支未改动。

skinny 分支的完整上下文见 `vllm/model_executor/layers/utils.py`：先执行 `x_view = x.reshape(-1, x.size(-1))`，在 `use_skinny` 且 `m > 8` 时取 `cu_count = num_compute_units()` 并调用 `ops.wvSplitK(weight, x_view, cu_count, bias)`，最后把输出 reshape 回 `(*x.shape[:-1], weight.shape[0])`。`csrc/rocm/skinny_gemms.cu` 的 switch 除新增的 `case 5` 外仍保留 `default:` 分支，对不支持的 N 抛出 "Unsupported N value: ..." 运行时错误（`wvSplitK` 的改动位于 `@@ -1277,6 +1277,12 @@`），即本次仅把 N=5 纳入快速路径，其余 N 值与回退行为不变。正文注明测试环境为 "Benchmarked on AMD Strix Halo (gfx1151, LPDDR5X-8000 128 GB)"，Model 为 `Qwen/Qwen3-8B (bf16, unquantized)` 配 eagle3 speculative decoding，input-len=128、output-len=128、num-prompts=5；perf claims 段另标注 12.3% faster 与 14.2% faster 两项，复现命令还带 `--max-model-len 4096 --gpu-memory-utilization 0.62 --max-num-seqs 1 --no-enable-prefix-caching` 与 `--num-warmups 5 --percentile-metrics ttft,tpot,e2el` 等参数。diff 中 `csrc/rocm/skinny_gemms.cu` 的 `case 5` 位于 `case 4` 之后、`default:` 之前，`WVSPLIT_TILE_CFG` 随 `use_wave32` 在 `(32, 16, sYT, 5)` 与 `(64, 16, sYT, 5)` 间切换；`vllm/model_executor/layers/utils.py` 的阈值改动位于 `@@ -178,7 +178,7 @@` 单行。正文首句给出触发条件为 eagle3 且 `num_speculative_tokens=4` 时 verification pass 的 batch size 为 5。

## 关键改动

```text
csrc/rocm/skinny_gemms.cu: wvSplitK 的 N 分派 switch 新增 case 5（wave32/64 两种 tile 配置）
vllm/model_executor/layers/utils.py: rocm_unquantized_gemm_impl 的 skinny GEMM 阈值由 N<=4 放宽到 N<=5
```

```diff
+      case 5:
+        if (use_wave32)
+          WVSPLIT_TILE_CFG(32, 16, sYT, 5)
+        else
+          WVSPLIT_TILE_CFG(64, 16, sYT, 5)
+        break;
```

```diff
-        if m > 8 and 0 < n <= 4:
+        if m > 8 and 0 < n <= 5:
```

## 性能声明

测试条件（正文）：AMD Strix Halo（gfx1151，LPDDR5X-8000 128 GB）；Model `Qwen/Qwen3-8B`（bf16，unquantized）配 eagle3 speculative decoding，input-len=128、output-len=128、num-prompts=5。结果：`Median TPOT: 43.02 ms → 37.73 ms (12.3% faster)`；`Decode throughput: 23.2 tok/s → 26.5 tok/s (14.2% faster)`。Reproduce 命令（照抄）：`vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.62 --max-num-seqs 1 --no-enable-prefix-caching --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":4}'`；`vllm bench serve --model Qwen/Qwen3-8B --num-warmups 5 --random-input-len 128 --output-len 128 --num-prompts 5 --max-concurrency 1 --temperature 0 --percentile-metrics ttft,tpot,e2el`。

## 验证与回归风险

PR 正文给出上述 Reproduce 的 serve/bench 命令与 benchmark 数字，但未注明新增/依赖的测试或单元测试。可能影响路径：ROCm 上 `rocm_unquantized_gemm_impl` 的 skinny GEMM 分支（现在 `m>8` 且 `N=5` 也会走 wvSplitK）以及 wvSplitK 对 N=5 的 tile 配置；PR 未注明其他回归风险。

## 关联

无（PR 正文与 diff 未给出其他 PR 编号或前序/后续线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
