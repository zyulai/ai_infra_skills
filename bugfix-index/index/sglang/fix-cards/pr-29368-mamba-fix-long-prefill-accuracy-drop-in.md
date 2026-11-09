# PR #29368: [Mamba] Fix long-prefill accuracy drop in radix prefix-cache state restore

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29368](https://github.com/sgl-project/sglang/pull/29368) |
| 作者 | adityakamat24 |
| 合并日期 | 2026-07-06 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
Mamba2Metadata 把 initial-state 恢复与 mamba_track_mask 做 AND,prefix-cache 短后缀请求被误判为无初始状态而从空白 Mamba 状态跑,长 few-shot prefill 精度崩塌;修复为删掉该 gate。

## 问题现象
Nemotron-3-Ultra 的 GSM8K 精度在长 few-shot prefill 下崩塌:8-shot 正常、20-shot 损坏,NVFP4 与 BF16 都复现(closes #29330)。用 `benchmark/gsm8k/bench_sglang.py --platinum --parallel 128` 贪心评测,Nemotron-3-Ultra-550B-A55B-NVFP4(TP4, 4x B200):8-shot 修复前 0.943 → 修复后 0.981;20-shot 修复前 0.617 → 修复后 0.980。用同架构的 Nemotron-Nano-9B-v2(BF16, 1x H100)二分定位:72-shot 0.680 → 0.945。由于根因在共享 Mamba 元数据,任何用 radix prefix cache 服务的 hybrid-mamba 模型都受影响。

## 根因
正文披露:`Mamba2Metadata` 用 tracking mask 对初始状态恢复做门控——`has_initial_states = context_lens_tensor > 0` 再与 `mamba_track_mask` 做 AND。`has_initial_states` 决定 prefill 是否从其 cache slot 里已有的状态作为 conv/SSM scan 的起点;`mamba_track_mask` 只标记会把状态快照进 prefix cache 的步骤(`extend_range.length >= mamba_cache_chunk_size`),两者本就独立。在 `--parallel` 共享 few-shot 前缀时,多数请求是 prefix-cache 命中、只 prefill 很短的唯一后缀,其 track mask 为 False,AND 之后 `has_initial_states` 被清零,后缀从空白 Mamba 状态开始跑;缓存的前缀越长丢失的状态越多,所以 8-shot 勉强通过而 20-shot 失败。该 gate 是 #24955 引入的,0.5.13 发布版没有。

## 修复方案
修改 `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` 的 `prepare_mixed`:删除 `mamba_track_mask` 门控,恢复条件只依赖 `context_lens > 0`,与 #24955 之前的行为一致。`mamba.py` 中的快照/tracking 路径仍使用 `mamba_track_mask`,保持不变。关键 diff(即被删除的代码):

```python
-        mamba_track_mask = getattr(forward_batch, "mamba_track_mask", None)
-        if mamba_track_mask is not None:
-            has_initial_states = (
-                has_initial_states & mamba_track_mask[: has_initial_states.shape[0]]
-            )
```

## 影响与回归风险
- 影响窗口:gate 由 #24955 引入,0.5.13 发布版不含该 gate(即受影响范围为 #24955 之后到本修复之前的版本)。
- 回归面:所有 hybrid-mamba 模型走 radix prefix cache 的 `prepare_mixed` 初始状态恢复路径:prefix-cache 命中的请求现在总能恢复缓存状态(即使该状态未按 chunk 快照),不再从空白状态跑;而 #24955 中引入 gate 想约束的场景(状态未被 track 的恢复)会重新出现,是否需要按 track mask 区分待后续验证。单元测试:mamba 层 934 passed,attention/mamba/gdn/hybrid_linear + mem_cache + model_executor 1501 passed,`test_fp4_kv_cache_quant_method.py` 的 3 个失败为 main 上已有、与本改动无关。无速度影响(仅删除一次布尔 AND)。

## 关联
- [#29330](https://github.com/sgl-project/sglang/issues/29330)(GSM8K 精度崩塌 issue)
- [#24955](https://github.com/sgl-project/sglang/pull/24955)(引入 mamba_track_mask gate 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
