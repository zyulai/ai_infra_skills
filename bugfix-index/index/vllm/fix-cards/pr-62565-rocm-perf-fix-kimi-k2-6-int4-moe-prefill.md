# PR #62565: [ROCm][Perf] Fix Kimi-K2.6 int4 MoE prefill regression on Triton 3.8 (gfx950 tuned config)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62565](https://github.com/vllm-project/vllm/pull/62565) |
| 作者 | afriedri |
| 合并日期 | 2026-11-03 |
| labels | rocm, ready, kimi |
| 热度 | 👍 0 · 💬 6 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
Triton 3.8 对 tuned config 的 256×256×32/2-stage tile 生成劣化代码,MI350/MI355 上 Kimi-K2.6 int4 MoE c32 TTFT 涨 ~150%;调 config 为 BLOCK_N 128 + 1 stage。

## 问题现象
Kimi-K2.6(int4 W4A16,TP8)在 MI350/MI355 上 prefill 性能在 v0.29 到 v0.31 之间回归:c32 并发 TTFT 上涨约 150%。影响面:只有 Kimi-K2 系列 int4 checkpoint 在 TP8、MI350/MI355 上会加载这些 config 文件,且只有约 6k token 以上的 batch 使用该条目。

## 根因
正文披露:起因是 Triton 3.7.1 → 3.8.0 的 codegen 回归,发生在 `fused_moe_kernel_gptq_awq` 的一个 tile 上——即 tuned config `"8192"` 条目(约 6k token 以上的每个 batch)所用的 BLOCK 256×256×32、8 warps、2 stages、`matrix_instr_nonkdim=32`。在 Triton 3.8 上该 tile 慢 3–8 倍;同一 tile 改为 1 stage 或 BLOCK_N 128 则正常。

## 修复方案
改四个内容相同的 fused_moe tuned config 文件(`vllm/model_executor/layers/fused_moe/configs/` 下 `E=384,N=256,dtype=int4_w4a16.json`,device_name 分别为 AMD_Instinct_MI350X、AMD_Instinct_MI350_OAM、AMD_Instinct_MI355X、AMD_Instinct_MI355_OAM):仅把 `"8192"` 条目从 BLOCK_N 256、num_stages 2 改为 BLOCK_N 128、num_stages 1,其余参数(BLOCK_M 256、BLOCK_K 32、8 warps、matrix_instr_nonkdim 32)不变;无任何代码改动。

```diff
     "8192": {
         "BLOCK_SIZE_M": 256,
-        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 32,
...
-        "num_stages": 2,
+        "num_stages": 1,
```

测试(Kimi-K2.6 TP8 shapes,Triton 3.8.0,w13+w2 在 M=8192/16384 求和):原配置 44180µs → 本 PR 配置 6129µs。Serving:c32 TTFT 1981→829ms(-58%),tok/s +6.2%;c1 不到达 `"8192"` 条目,其 TPOT 差异为重启造成的 run-to-run 噪声。正文注明分析、config 搜索与起草使用了 AI 辅助(Claude),所有改动均经人工评审。

## 影响与回归风险
- 影响窗口:v0.29 与 v0.31 之间(Triton 3.8 环境下);正文以版本区间表述。
- 回归面:仅上述四个 int4_w4a16 config JSON 的 `"8192"` 条目;只影响 MI350/MI355 + Kimi-K2 系列 TP8 + ~6k token 以上 batch 的 MoE kernel tile 选择;c1 等小 batch 不受影响。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
