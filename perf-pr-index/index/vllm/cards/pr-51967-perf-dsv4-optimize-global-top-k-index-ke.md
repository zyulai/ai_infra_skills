# PR #51967: [Perf][DSV4] Optimize global top-k index kernel with compile-time constants

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chaunceyjiang |
| 合并 | 2026-08-16 |
| 分类 | Attention / Kernel |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 `vllm/models/deepseek_v4/common/ops/cache_utils.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/51967 |

## 一句话结论

把 DSV4 global top-k 索引 Triton kernel 的 stride/topk/block_size 改为编译期常量,内核延迟降约 15.1%(~7.50us → ~6.37us)。

## 动机

PR 正文未展开动机,Purpose 仅一句:"Optimize global top-k index kernel with compile-time constants"。从 diff 看,DeepSeek V4 的 global top-k 索引计算 kernel 此前把 `topk`、`block_size`、各 tensor stride 等在服务期间不变的形状参数全部作为运行时参数传入。这些值每次 launch 都要走参数传递,且阻碍 Triton 编译期对循环边界和地址计算的特化,是内核层面的可优化点。

## 优化点

改动是纯 kernel 签名特化:将 `_compute_global_topk_indices_and_lens_kernel` 中的 `global_topk_indices_stride`、`topk_indices_stride`、`topk`、`block_table_stride`、`block_size` 五个参数标注为 `tl.constexpr`。对固定的服务配置,这些值均不变(topk 由模型配置决定、block_size 由 KV cache 配置决定、stride 由张量形状决定),Triton 会将其编译为立即数,从而做常量折叠、消除运行时除法/寻址中的变量,并收紧生成代码。kernel 内部计算逻辑本身未改动。

## 关键改动

```text
vllm/models/deepseek_v4/common/ops/cache_utils.py: global top-k 索引内核参数 constexpr 化
```

```python
-    global_topk_indices_stride,
+    global_topk_indices_stride: tl.constexpr,
     topk_lens_ptr,
     topk_indices_ptr,
-    topk_indices_stride,
-    topk,
+    topk_indices_stride: tl.constexpr,
+    topk: tl.constexpr,
     token_to_req_indices_ptr,
     block_table_ptr,
-    block_table_stride,
-    block_size,
+    block_table_stride: tl.constexpr,
+    block_size: tl.constexpr,
```

## 性能声明

Kernel 微基准(pack 中的脚本:num_tokens=512、topk=512、block_size=256、TRITON_BLOCK_SIZE=1024,`triton.testing.do_bench` warmup=100/rep=500):

| Kernel 版本 | Latency | 相对变化 |
| --- | ---: | ---: |
| 原实现 | ~7.50 us | Baseline |
| 编译期常量优化 | ~6.37 us | **15.1% faster** |

端到端 serving:DeepSeek-V4-Flash-0731,TP8、`--kv-cache-dtype fp8`、block-size 256、开启 EP、`--no-enable-prefix-caching`,random 数据集 input 1024 / output 64、128 prompts、request-rate inf:吞吐 634.97 → 638.12 tokens/s(**+0.50%**),Mean TPOT 103.33 → 102.32 ms(**-0.98%**)。

## 验证与回归风险

PR 给出可复现的内核微基准脚本与 serving 基准命令,但 Test Plan 一节为空,未声明新增单测。风险面:改动仅位于 DSV4 的 cache_utils global top-k 路径,不影响其他模型;constexpr 特化意味着每个不同的 (stride, topk, block_size) 组合都会触发一次 Triton 重编译,配置或 batch 形状频繁变化时可能增加首次编译开销;端到端收益较小(+0.50%),主要价值在内核延迟本身。

## 关联

DeepSeek V4 家族同批其他优化:#56254(DSA:DeepGEMM sparse MQA logits 接入 DeepSeek V4.1 indexer)、#64101(DSv4.1 ROCm MI355X TP8 mono decode)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
