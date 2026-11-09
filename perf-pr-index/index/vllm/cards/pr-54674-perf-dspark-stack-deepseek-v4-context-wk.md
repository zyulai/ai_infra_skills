# PR #54674: [Perf][DSpark] Stack DeepSeek V4 context WKV projections

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @liuyao0322 |
| 合并 | 2026-09-17 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, deepseek, DSv4, dflash |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/54674 |

## 一句话结论

把 DSpark 各 draft 层 context WKV 投影堆成单次 merged GEMM,DeepSeek-V4-Flash-0731 2xH20 TP2 e2e 中位吞吐 +3%~4%。

## 动机

DeepSeek V4 的 DSpark 此前为每个 draft layer 单独运行该层 fused `wq_a|wkv` 投影来得到 context KV,并丢弃 Q 部分——即每层一次投影且计算了 context-KV 路径并不消费的输出。PR 把这些投影合并为对共享 target hidden states 的一次计算,省掉重复 launch 与无效输出。

## 优化点

把各 draft 层的 WKV 投影堆成一个 `MergedColumnParallelLinear`(`context_wkv_proj`),对共享的 target hidden states 只运行一次,再 `view` 成 `[tokens, num_dspark_layers, head_dim]` 逐层取用。权重加载时新增 `_duplicate_context_wkv_weights`,按 `^mtp\.(\d+)\.attn\.wkv\.(.+)$` 匹配每个 draft 层的 WKV 权重与量化 scale,复制成 `context_wkv_proj.*` 并打上 `shard_id = layer_idx`(越界 layer 跳过);`precompute_and_store_context_kv` 由逐层循环改为一次 stacked projection 后按层分发。保留原有逐层 KV normalization、RoPE / 量化、cache insertion、profiling 行为与 in-place weight refresh 语义。probe 验证三个合并权重与 scale shard 在两个 rank 上逐元素等于原 draft 层 WKV 参数,旧/新投影输出 `max_abs_diff = 0.0`。

## 关键改动

```text
vllm/models/deepseek_v4/nvidia/dspark.py: 核心。新增 _CONTEXT_WKV_RE 与 _duplicate_context_wkv_weights 复制权重/scale 到 merged shard;__init__ 创建 context_wkv_proj = MergedColumnParallelLinear;precompute_and_store_context_kv 改为单次 stacked 投影
tests/models/test_dspark_mla.py: 新增 test_dsv4_context_wkv_weights_are_duplicated_by_draft_layer(校验 shard_id 与 data_ptr 共享)与 test_dsv4_context_kv_uses_one_stacked_wkv_projection(校验投影只调用一次)
```

关键代码(单次 stacked projection 取代逐层循环):

```python
        all_kv = self.context_wkv_proj(main_x).view(
            main_x.shape[0], self.num_dspark_layers, self.config.head_dim
        )
```

## 性能声明

End-to-end engine benchmark:两组独立的 2x NVIDIA H20-3e GPU 对、`DeepSeek-V4-Flash-0731`、TP=2、开启 expert parallelism、FP8 KV cache、eager execution、7-token greedy DSpark speculation;每请求 256 input tokens、恰好 128 output tokens,关闭 prefix caching 与 detokenization;AB/BA 排序,每 job 两实现各 2 次 warmup + 每 batch size 12 次 `LLM.generate()`(每实现每 batch 24 样本),speedup 为两 job 内均值比的几何平均,95% CI 为按 job/GPU 分层的 20,000 次 bootstrap。Output tok/s(baseline → stacked):Batch 1 36.07 → 39.01(配对 1.082x,95% CI 0.883x-1.334x,中位 1.037x);Batch 8 121.48 → 126.67(1.043x,1.002x-1.085x,中位 1.035x);Batch 16 233.62 → 237.97(1.019x,0.979x-1.060x,中位 1.031x)。正文保守结论:约 3%-4% median end-to-end throughput improvement,Batch 8 配对均值吞吐统计为正;Batch 8 在两 GPU 对复现 +5.0% / +3.6%,Batch 1 与 16 的 CI 跨 parity。Stacked projection probe 显示新旧投影输出 `max_abs_diff=0.0`,单投影 speedup 在 context tokens 1/8/32/128/512/2048、两 rank 上为 2.93x-38.03x。正文注明该 runtime 非逐位确定、e2e 数字含投机轨迹,不能当作孤立的 GEMM 收益;计时区含 prefill、调度、DSpark draft/verify、KV-cache 与 decode,不含模型加载、HTTP、tokenization/detokenization。

## 验证与回归风险

`.venv/bin/python -m pytest tests/models/test_dspark_mla.py -v`:12 passed;`pre-commit run --files ...`(ruff、mypy、SPDX、forbidden-import 全通过);2xH20 TP2/EP 全引擎 smoke test 加载模型、校验两 rank 权重与 scale、完成 probe 与两次 33-token 生成;gsm8k correctness(`tests/evals/gsm8k/test_gsm8k_correctness.py`,1319 题 5-shot):测得 0.9409,期望 0.9200、容差 0.0800,1 passed。风险面:改动 checkpoint 权重加载(shard 复制与 shard_id 映射)与 DeepSeek V4 DSpark 的 context-KV 计算路径;正文注明该 runtime 非逐位确定,重复 greedy 请求 output token digest 会变化,故 e2e 数字含投机轨迹;投影 probe 单独给出 `max_abs_diff=0.0` 以隔离 GEMM 收益。

## 关联

PR 正文明确:#53577 也改 `deepseek_v4/nvidia/dspark.py`,但加的是 PP prefill/context-KV materialization 与 padded graph-batch safety,仍保留 per-layer fused WKV projection,与本 PR 功能独立(若先合入需 textual rebase);#51620、#51684 修其他 DFlash-family 模型的量化 context-KV 投影正确性,不改 DeepSeek V4 DSpark 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
