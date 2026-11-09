# PR #16488: [Kernel] Add tuned FusedMoE kernel config for Llama4 Scout, TP=8 on H100 

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shinoss |
| 合并 | 2025-04-11 |
| 分类 | MoE \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/16488 |

## 一句话结论

Llama4 Scout TP=8/H100 新增 tuned FusedMoE 配置，5k/1k 负载下 TTFT 最多提升 17.6%。

## 动机

PR 未注明改前的具体瓶颈。正文仅说明这份配置主要由 benchmark_moe.py 脚本生成，其中 bsz=16 需要进一步单独调优；并给出了 `vllm serve` 与 `benchmark_serving.py` 的复现命令，用于为 meta-llama/Llama-4-Scout-17B-16E-Instruct 在 TP=8 的 H100 上补齐调优后的 FusedMoE kernel 配置。

## 优化点

改动是纯配置新增：新增一份 FusedMoE 调优配置 `E=16,N=1024,device_name=NVIDIA_H100.json`，针对 H100 上 E=16、N=1024 的 MoE 形状，按 batch size 的 18 个档位（1、2、4、8、16、24、32、48、64、96、128、256、512、1024、1536、2048、3072、4096）逐档给出 BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K、GROUP_SIZE_M、num_warps、num_stages 的取值。小 batch（如 1、2、4）多用 BLOCK_SIZE_M=16、BLOCK_SIZE_K=64、num_warps=4，大 batch（如 2048、3072、4096）改用 BLOCK_SIZE_M=128、BLOCK_SIZE_N=256、num_warps=8。PR 未改动 kernel 实现代码，也未说明选取这些参数的具体依据。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=NVIDIA_H100.json: 新增 H100 / E=16,N=1024 的 tuned FusedMoE 配置，按 batch size 索引 tile 大小、num_warps 与 num_stages
```

配置形态（摘自 diff）：

```text
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
```

## 性能声明

正文结论为 "up to 17.6% improvement in TTFT, 7.7% improvement in TPOT, and 9.1% gain in output throughput"，测试条件写为 "Results tested on 5k input len and 1k output len"：模型 meta-llama/Llama-4-Scout-17B-16E-Instruct，硬件 H100，TP=8，`VLLM_USE_V1=1`，serve 参数为 `--max-num-seqs 64 --no-enable-prefix-caching --max_num_batched_tokens=81920 --max-model-len=262144`；benchmark_serving 使用 random dataset、`--num-prompts 500 --random-input-len 5000 --random-output-len 1000`，并按 `--max-concurrency` 扫并发。精度 PR 未注明。

## 验证与回归风险

PR 未注明引入或依赖的测试。改动为新增一份按 device_name / E / N 匹配的配置文件，正文未说明对其他模型、硬件或精度的回归评估。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
