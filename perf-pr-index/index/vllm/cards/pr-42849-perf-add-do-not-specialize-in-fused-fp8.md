# PR #42849: [Perf] Add do_not_specialize in fused FP8 RoPE kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-05-18 |
| 分类 | Quantization |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(pack 未提供);diff 可用,仅涉及 `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` 一行 |
| 链接 | https://github.com/vllm-project/vllm/pull/42849 |

## 一句话结论

给融合 FP8 RoPE+quant 内核加 `do_not_specialize`,消除推理期 num_tokens 反复触发 Triton 重编译,H200 上 e2e TTFT 降约 9%。

## 动机

Triton 默认会对整型参数做特化(如可被 16 整除的值会被编译进二进制)。`_fused_inv_rope_fp8_quant_per_head` 内核的 `num_tokens` 参数随每个 batch 变化,推理中每遇到一个新的 num_tokens 值,就可能触发一次内核重编译。冷启动/清空 Triton cache 后的推理期间,这些重编译开销直接打在 TTFT 上。作者将其列为 `do_not_specialize` 来阻止这种不必要的重编译。

## 优化点

单行改动:把 `@triton.jit` 换成 `@triton.jit(do_not_specialize=["num_tokens"])`。num_tokens 不再参与 Triton 的编译期特化,一套编译产物即可服务任意 token 数,消除推理过程中因 batch 大小变化带来的重复 JIT 编译与 host 端查找开销。内核计算逻辑(逆 RoPE + FP8 量化融合)本身不变。

## 关键改动

```text
vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py: 禁止对 num_tokens 做 Triton 特化
```

```python
-@triton.jit
+@triton.jit(do_not_specialize=["num_tokens"])
 def _fused_inv_rope_fp8_quant_per_head(
     o_ptr,
     positions_ptr,
```

## 性能声明

H200,e2e serving:DeepSeek-V4-Flash,TP8、`--max-num-seqs 8`、`--kv-cache-dtype fp8`、`--block-size 256`、`--no-enable-prefix-caching`,ShareGPT 数据集 output-len 300、480 prompts、max-concurrency 8,且基准前 `rm -rf ~/.triton/cache`:

- Main:Mean TTFT 370.14 ms,Output throughput 681.27 tok/s,Mean TPOT 10.54 ms
- PR:Mean TTFT 338.37 ms,Output throughput 687.99 tok/s,Mean TPOT 10.53 ms

即 Mean TTFT 改善约 9%(PR 原文:"e2e TTFT improves 9% on H200"),TPOT 基本持平。

## 验证与回归风险

验证方式为上文 serving 基准(含清空 Triton cache 的冷启动场景);PR 未声明新增单测。风险面:改动位于 DeepSeek V4 专属算子目录,只影响走该 fused FP8 RoPE 内核的模型路径;`do_not_specialize` 可能牺牲少量针对特定 num_tokens 的编译期优化空间,但基准显示 TPOT 无回退。其他使用 `num_tokens` 整型参数的 Triton 内核若存在同类问题,可复用此模式(PR 未展开)。

## 关联

DeepSeek V4 内核同批其他优化:#51967(DSV4 global top-k 索引内核 constexpr 化)、#56254(DSA:DeepGEMM sparse MQA logits 接入 V4.1 indexer)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
