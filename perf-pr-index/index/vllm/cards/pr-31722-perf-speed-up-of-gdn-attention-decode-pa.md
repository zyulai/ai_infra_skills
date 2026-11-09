# PR #31722: [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vadiklyutiy |
| 合并 | 2026-01-06 |
| 分类 | Attention / Kernel |
| Label | performance, ready, qwen |
| Diff | diffstat 缺失(pack 未提供);diff 可用,仅涉及 `vllm/model_executor/layers/fla/ops/fused_recurrent.py` 一行 |
| 链接 | https://github.com/vllm-project/vllm/pull/31722 |

## 一句话结论

把 GDN(gated delta rule)decode 内核的 V 维分块 BV 从 8 提到 32,Qwen3-Next 端到端 decode 吞吐提升 16.8%(23891 → 27904 tok/s)。

## 动机

Qwen3-Next 的 GDN attention decode 走 `fused_recurrent_gated_delta_rule_fwd` Triton 内核。该内核沿 V 维按 `BV` 分块累加,原先 `BV` 被硬性压到 `min(next_power_of_2(V), 8)`,即使 head_dim 较大也只以 8 为块宽,导致一个 kernel 内要串行处理大量小块、launch 内并行度利用不足。作者针对 Qwen3-Next 的实际形状放大分块以提速 decode。

## 优化点

一行改动:在 `fused_recurrent_gated_delta_rule_fwd` 中把 `BV = min(triton.next_power_of_2(V), 8)` 改为 `min(triton.next_power_of_2(V), 32)`。V 维每块处理元素数从最多 8 提到最多 32,NV 分块数相应减少约 4 倍,单次 kernel 内每 program 处理更多 V 列,减少总计算迭代与访存事务数;K 维分块 `BK` 不变(仍要求 `NK == 1`)。内核算法(gated delta rule 递推)本身未变,属于纯 tiling 参数调优。

## 关键改动

```text
vllm/model_executor/layers/fla/ops/fused_recurrent.py: 放大 fused_recurrent 内核 V 维 block size
```

```python
-    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
+    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
     NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
     assert NK == 1, "NK > 1 is not supported yet"
```

## 性能声明

内核微基准(`fused_recurrent_gated_delta_rule_fwd`,Qwen3-Next 形状):

- H200:TP=4 下 batch 1024 从 345.12 us 降到 215.12 us,batch 32 从 19.87 us 降到 13.60 us;TP=2 下 batch 1024 从 672.48 us 降到 420.22 us。
- B200:TP=4 下 batch 1024 从 244.60 us 降到 154.94 us;TP=2 下 batch 1024 从 473.53 us 降到 305.21 us。

端到端 decode:Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,TP2、`--enable-expert-parallel`、`--async-scheduling`、`--no-enable-prefix-caching`,random input 32 / output 1000、并发 1024、1024 prompts:`VLLM_USE_FLASHINFER_MOE_FP8=1`,Output token throughput 23891 → 27904 tok/s,**Speedup 16.8%**。

## 验证与回归风险

PR 提供两套硬件(H200/B200)x 两种 TP(2/4)x 多个 batch size 的内核级 Before/After 表与端到端基准命令;未声明新增单测。风险面:BV 增大改变所有走该内核的模型的寄存器/共享内存占用,GDN 系模型(Qwen3-Next 等)均受影响;若某些形状下 V 很小,BV 上限提高无副作用(仍取 min)。PR 未报告其他 GDN 模型或更小 batch 的端到端回归数据。

## 关联

前序:#24526(Qwen3-Next 模型支持)。同批无其他 GDN/线性注意力卡。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
