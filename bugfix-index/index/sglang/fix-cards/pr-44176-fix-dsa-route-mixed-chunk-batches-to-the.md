# PR #44176: [Fix] dsa: Route mixed-chunk batches to the RAGGED top-k transform for bf16-only sparse prefill

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#44176](https://github.com/sgl-project/sglang/pull/44176) |
| 作者 | yunyang1999 |
| 合并日期 | 2026-11-10 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 27 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
FP8 KV cache 下 mixed-chunk 批(MIXED)未被路由到 RAGGED top-k 变换,DSA 后端首批即崩;让 MIXED 与 EXTEND 同等路由修复。

## 问题现象
FP8 KV cache 且 `--enable-mixed-chunk` 时,DSA 后端在第一个 mixed batch 上崩溃,prefill backend 为 `flashmla_sparse`、`flashmla_sparse_q8` 或 `triton_sparse_mla`。两种症状:prefill CUDA graph(breakable,CUDA 默认)下报 `RuntimeError: RAGGED topk_transform requires topk_indices_offset; expected extend-without-speculative metadata.`;把 MIXED 直接交给 `forward_extend` 的场景(context parallel、NPU)报 `RuntimeError: kv must have dtype torch::kBFloat16`(flashmla_sparse / flashmla_sparse_q8)或 `TypeError: sparse_mla_prefill needs a bf16 KV cache, got torch.float8_e4m3fn`(triton_sparse_mla)。无 prefill graph 的 eager 路径不失败(eager_runner 把批重写成 EXTEND)。

端到端复现:SM90、GLM-5.2-FP8 真实权重、TP8、`--kv-cache-dtype fp8_e4m3 --chunked-prefill-size 4096 --enable-mixed-chunk`、默认 breakable prefill CUDA graph;8 条短 prompt 先开始 decode,0.5 秒后到达 12000-token 长 prompt,三个 chunk 各与 8 条 decode 行合并;main 上服务器在第一个 mixed batch 即死。

## 根因
正文披露:根因是一个路由谓词——`get_topk_transform_method` 只对 `ForwardMode.EXTEND` 返回 `TopkTransformMethod.RAGGED`,而 mixed-chunk 批携带 `ForwardMode.MIXED`。栈的其余部分已把这种批当作 extend 形状批处理(decode 行是带完整 prefix 的单 token extend)。prefill CUDA graph runner 用真实 MIXED 批构建 attention metadata(为 PAGED 准备、无 `topk_indices_offset`),而图内 indexer 请求 RAGGED 变换,于是报错。该缺口由 @ch-wan 在 #32779 评审中指出,是与 `flashmla_sparse` 共有的 pre-existing 问题。

## 修复方案
`get_topk_transform_method` 对这三个 prefill 实现把 `ForwardMode.MIXED` 与 `ForwardMode.EXTEND` 同等路由(FP8 pool,或 #41790 之后 flashmla_sparse_q8 的任意 pool),使后端从 MIXED 批构建的 metadata 与 EXTEND 图重放操作、eager runner 的期望一致。`python/sglang/srt/layers/attention/dsa_backend.py`:

```diff
-            and forward_mode == ForwardMode.EXTEND
+            # MIXED matches EXTEND: the prefill graph runner builds metadata from
+            # MIXED but replays EXTEND graphs.
+            and forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
```

其余不变:`init_forward_metadata` 的 extend 分支本就经 `is_extend()` 为 MIXED 构建 `page_table_1_flattened` 与 `topk_indices_offset`,indexer 本就为 MIXED 跑 ragged top-k。同时删除 q8 launch validator 对 `--enable-mixed-chunk` 的拒绝(参数 `mixed_chunk_enabled` 及对应测试用例移除),q8 的 `forward_extend` 报错信息改为仅列出仍不支持的 SPLIT_PREFILL、DLLM_EXTEND。新增单测 `test_sparse_fp8_mixed_chunk_prefill_cases`(MIXED FP8 用例过全部 prefill impl)与路由表 5 行 MIXED 用例;端到端表格中 main 全部报错、本 PR 三个后端均正常运行,`flashmla_kv` 对照组保持 PAGED 不受影响;服务端 fp32 重算校验 RAGGED 行 min cosine ≥ 0.9998(q8 为其自身 FP8 精度)。

## 影响与回归风险
- 影响窗口:正文未给出明确版本/提交范围(缺口为 #32779 评审时已存在的 pre-existing 问题)。
- 回归面:RAGGED mixed batch 每层需反量化整个 ragged KV(`sum(seq_lens)` 个 token),与带 prefix 的 chunked EXTEND 已付的代价相同(含 decode 行的上下文),这是 bf16 内核跑在 FP8 pool 上的代价;FP8-native 实现(flashmla_kv、flashinfer_sparse_mla、trtllm)与 bf16 pool 两种模式下均保持 PAGED 不变;born-fp8 q 仍 EXTEND-only,MIXED 走 bf16-q hand-off(其 helper 会 cast);q8 与 mixed-chunk 的组合由启动即拒绝变为放行。注:pack 中 Speed Tests 一节在 20k 字符处截断,diff 完整。

## 关联
- [#32779](https://github.com/sgl-project/sglang/pull/32779)(评审中指出该缺口)
- [#41790](https://github.com/sgl-project/sglang/pull/41790)(flashmla_sparse_q8 支持任意 pool)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
