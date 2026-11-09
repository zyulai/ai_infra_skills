# PR #27482: [Bug] Fix out-of-range token id crashing tp=1 `VocabParallelEmbedding`

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27482](https://github.com/sgl-project/sglang/pull/27482) |
| 作者 | hnyls2002 |
| 合并日期 | 2026-06-07 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | crash |

## 一句话结论
测试生成越界 token id 32000,tp=1 时 embedding 不做掩码导致 GPU 端 assert/coredump;修 off-by-one 并加 OOB 探针。

## 问题现象
CI 中的 flaky 崩溃:在 `TestEagleLlama2Retract.test_radix_attention` 中以 GPU coredump 形式出现。触发条件(正文披露):`gen_radix_tree` 用 `random.randint(0, 32000)`(闭区间)构造请求,可能生成 token id `32000`,对 32000 词表的模型(合法范围 `0..31999`)越界;仅在单卡(tp=1)出现,生成器未设种子,约 1.2% 的运行会命中。越界 id 最终到达 `F.embedding`,触发 `vectorized_gather_kernel` 的 device-side assert。

## 根因
正文披露两点:1) `gen_radix_tree` 的 off-by-one——`random.randint(0, 32000)` 为闭区间,会发出合法范围之外的 id 32000;2) 在 tp=1 时 `VocabParallelEmbedding.forward` 不对 input ids 做掩码(mask),越界 id 直达 `F.embedding` 引发 device-side assert(正文原话:「At `tp=1`, `VocabParallelEmbedding.forward` does not mask input ids」)。

## 修复方案
1. `python/sglang/test/kits/radix_cache_server_kit.py` 的 `gen_radix_tree`:token id 改为 `random.randint(0, 31999)`,修复 off-by-one(diff 注释注明「randint is inclusive; vocab_size-1 = 31999」);`test/registered/unit/mem_cache/test_radix_cache_unit.py` 中 `random.randint(1, vocab_size)` 同步改为 `random.randint(1, vocab_size - 1)`。
2. `python/sglang/srt/layers/vocab_parallel_embedding.py` 的 `forward`:对输入 ids 加 `maybe_detect_oob` 门控,让越界或负数(未掩码 sentinel)id 以带定位信息的 assert 暴露,而不是静默 OOB / coredump:
```diff
     def forward(self, input_):
+        maybe_detect_oob(
+            input_, 0, self.num_embeddings, "VocabParallelEmbedding input id"
+        )
         if self.tp_size > 1:
```
3. `python/sglang/srt/utils/async_probe.py` 的 `maybe_detect_oob`:把下界与上界拆成两条 `torch._assert_async` 分别断言,使报错信息能指明是低界失败(负数/sentinel)还是高界失败(越界);该检查为异步 OOB 检查,不做 GPU-CPU 同步,错误在下一个 sync 点浮现,且仅在 `SGLANG_ENABLE_ASYNC_ASSERT` 开启时生效。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:`VocabParallelEmbedding.forward` 是所有模型 embedding 的公共路径,新增的 `maybe_detect_oob` 在 `SGLANG_ENABLE_ASYNC_ASSERT` 未开启时直接 return,正常运行零开销;开启时会引入额外的 min/max 归约。tp>1 路径仍走原有掩码逻辑不变。测试侧两处 randint 边界收紧只影响测试数据生成。

## 关联
正文未列出(正文仅附 CI 运行链接:Run #27084365414、Run #27084365398)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
