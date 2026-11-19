# PR #68129: [Perf][Step5] Keep PrimTS SWA verify batches on the decode plan

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JaredforReal |
| 合并 | 2026-11-19 |
| 分类 | Speculative Decoding(tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/68129 |

## 一句话结论

PrimTS SWA attention 对投机 verify 批此前只在 qlen <= 4 时走 paged decode plan,超过则退回慢得多的 prefill plan;本 PR 把阈值放宽到 `max(4, num_speculative_tokens + 1)`,使 4x GB300 上 MTP `num_speculative_tokens=6` 的输出吞吐在并发 64 提升 8.2%、并发 16 提升 3.6%,输出不变。

## 动机

`Step5PrimsTSSWA` 只对 `qlen <= 4` 的 uniform 批选 decode plan,而 PrimTS routing 与 Step5 prefill planning 把最多 `num_speculative_tokens + 1` 的 uniform 批当作 decode。于是 FP8 KV cache 且 draft tokens > 3 时,每个 SWA 层的 verify 步都落到 prefill plan。CUDA-graph 微基准显示 uniform 批下 decode plan 在 1-64 query tokens 比 prefill plan 快 1.6-22x,128 tokens 处持平;输出在 5 与 8 query tokens 时与 FP32 参考一致。

## 优化点

- `Step5PrimsTSSWA.__init__` 增加 `num_speculative_tokens=0`,设 `self.max_decode_query_len = max(_MIN_SWA_DECODE_QUERY_LEN, num_speculative_tokens + 1)`(`_MIN_SWA_DECODE_QUERY_LEN = 4`),`run()` 中 `decode = n == requests * qlen and qlen <= self.max_decode_query_len`。
- 非投机 uniform 批(至多 4 tokens)保持今天的 plan,行为不变;阈值保持有界而非无限,因为 decode plan 按"确切请求数 + query 长度"建 key,无界会爆 plan 数量。
- BF16 KV 不受影响(routing 已把那些 verify 批送往 native path);构造点 `sparse_attention.py` 把 `self.num_speculative_tokens` 传入。
- 依赖说明:Step5 MTP 要求 3 个 MTP 模块的倍数,6 是第一个能到达此路径的设置;#66951 之前该设置因多模块 MTP re-prefill 可能产生负 position 而在每个 backend 上启动失败。

## 关键改动

```python
# vllm/models/step5/nvidia/prims_ts.py
# Uniform batches up to this query length use the paged decode plan.
_MIN_SWA_DECODE_QUERY_LEN = 4

class Step5PrimsTSSWA:
    """Reuse dense PrimTS plans and apply Step5's existing separate gate."""

    def __init__(self, model_len, window, num_speculative_tokens=0):
        self.model_len, self.window = model_len, window
        # The decode plan is several times faster on uniform batches; keep
        # speculative verify batches of every draft length on it.
        self.max_decode_query_len = max(
            _MIN_SWA_DECODE_QUERY_LEN, num_speculative_tokens + 1
        )
        self.plans = {}
```

```diff
-        decode = n == requests * qlen and qlen <= 4
+        decode = n == requests * qlen and qlen <= self.max_decode_query_len
```

新测试用五 token verify 批对 FP32 参考,分别 eager 与 CUDA graph replay,并断言 plan 类型为 `BatchDecodePagedTSWrapper`。

## 性能声明

- 4x GB300,Step5 TP4 + EP、PrimTS attention、FP8 KV、MTP `num_speculative_tokens=6`、`--block-size 64`、random 1024 in / 512 out、steady-state(两臂均含 #66951):并发 64 吞吐 1,155 → 1,250 tok/s(1.08x,+8.2%);并发 16 474 → 491 tok/s(1.04x);并发 1 62 → 61 tok/s(1.00x,不变)。
- GSM8K(1319 题):0.942 → 0.933,在 run-to-run noise 内;draft acceptance rate 4.97% vs 4.96%。
- 微基准(uniform 批):decode plan 相对 prefill plan 快 1.6-22x(1-64 query tokens),128 tokens 处相等。

## 验证与回归风险

- 新测试 `test_prims_ts_swa_verify_beyond_four_tokens_uses_decode_plan`:五 token verify 批对照 FP32 参考(eager 与 CUDA graph replay),用旧阈值会失败、用本 PR 通过;同时断言 `swa.plans.values()` 类型为 `BatchDecodePagedTSWrapper`。
- `pytest tests/models/step5/test_prims_ts_attention.py tests/kernels/test_fused_qknorm_rope_cache.py` 在 GB300(FlashInfer 0.7.4)36 tests 通过。
- 回归风险:CI 只在 H200 与 MI355 跑 `tests/models/step5`,那里这些 PrimTS 测试因需要 SM100/SM103/SM107 而 skip,故 GB300 路径无 CI 覆盖;本 PR 为 #62550 的 follow-up。

## 关联

- Follow-up to #62550(已合并);依赖 #66951(修复多模块 MTP re-prefill 负 position)才能以 `num_speculative_tokens=6` 启动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
