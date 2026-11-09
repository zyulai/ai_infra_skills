# PR #18161: [Kernel] [V1] Fix performance regression for triton unified attention

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18161](https://github.com/vllm-project/vllm/pull/18161) |
| 作者 | tdoublep |
| 合并日期 | 2025-05-15 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
Triton 3.3 不再把 kv cache 末维 stride 判为常量致 unified attention 回归 ~40%;两形参改 tl.constexpr 恢复性能。

## 问题现象
从 triton 3.2 迁移到 3.3 后,`triton_unified_attention` kernel 出现约 40% 的严重性能回归。serve benchmark(985 请求,3 轮):main 中位 TPOT 49.74-49.79ms、请求吞吐 37.62-40.74 req/s、输出吞吐 4785-5181 tok/s;本 PR 后中位 TPOT 36.04-38.20ms、吞吐 53.64-57.50 req/s、输出吞吐 6823-7313 tok/s。

## 根因
正文披露:回归来自 triton-lang/triton#5512,该变更重做了 Triton 判定 kernel 参数是否为常量的方式。此前 Triton(正确)检测出 `stride_k_cache_3` 与 `stride_v_cache_3` 是常量,并借此生成更快的 kernel;新逻辑不知为何不再如此,需要显式告诉编译器把这两个 stride 当常量解释。

## 修复方案
1. `vllm/attention/ops/triton_unified_attention.py`:`stride_k_cache_3`/`stride_v_cache_3` 形参从 `tl.int64` 改为 `tl.constexpr`;
2. `vllm/v1/attention/backends/triton_attn.py`:因 flash_attn.py 的 assert(要求 builder 类型),而 TritonBackend 有意直接复用 `FlashAttentionMetadata`/`FlashAttentionMetadataBuilder`,新增 `TritonAttentionMetadataBuilder(FlashAttentionMetadataBuilder)`(`aot_schedule=False`)并让 `get_builder_cls()` 返回它,避免复制一套 TritonAttentionMetadata:

```diff
-    stride_k_cache_3: tl.int64,  # int
+    stride_k_cache_3: tl.constexpr,  # int
...
-    stride_v_cache_3: tl.int64,  # int
+    stride_v_cache_3: tl.constexpr,  # int
```

## 影响与回归风险
- 影响窗口:Triton 3.3 起(正文:从 triton 3.2 迁到 3.3 观察到回归)。
- 回归面:unified attention kernel 签名变化——两个 stride 改为 constexpr,参与编译期特化(不同 stride 值会产生不同编译变体);`TritonAttentionBackend.get_builder_cls()` 返回新的 builder 子类,metadata/builder 复用 FlashAttention 实现的既有路径不变。

## 关联
- triton-lang/triton#5512(引入回归的变更)
- vllm flash_attn.py#L287 的 assert(第二处改动的动机)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
