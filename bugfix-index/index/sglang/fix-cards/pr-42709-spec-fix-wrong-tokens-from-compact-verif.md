# PR #42709: [Spec] Fix wrong tokens from compact-verify CUDA graphs when a token tier is captured more than once

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#42709](https://github.com/sgl-project/sglang/pull/42709) |
| 作者 | ormandj |
| 合并日期 | 2026-10-09 |
| labels | (none) |
| 热度 | 👍 0 · 💬 3 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
compact ragged verify 下同一 token tier 被多次捕获时只刷新最后一次的 layout,更早的 graph 用捕获期合成偏移算错 token;改为每个 tier 只建一个共享 layout。

## 问题现象
开启 `SGLANG_RAGGED_VERIFY_MODE=compact` 且存在多个 attention graph 变体(如 DSv4.1 的 `candidate_unfiltered` / `candidate_filtered`)或多个 PD-mux stream 时,投机验证输出错误 token。A/B 实测(DeepSeek-V4.1-Flash,TP4,4×RTX PRO 6000 Blackwell SM120,DSpark gamma 5,#40567 已应用):修复前逐 token logprob 差最大达 14.4 nats,temperature 1 下 32 条流中 9 条循环到长度上限;修复后最大差 0.79–0.86 nats,无流循环或触顶。eager verify 在两个版本中均正确。

## 根因
正文披露:同一 token tier 被多次捕获时,每次 `_capture_ragged_verify_layout` 都新建 `RaggedVerifyLayout` 并覆盖 `_captured_ragged_layouts[num_tokens]`;replay 时 `_stage_ragged_verify_layout` 只把实时 `verify_lens` / `qo_indptr` 拷进最后一次捕获注册的 layout,更早捕获的 graph 仍读各自捕获时的 layout——其中是合成捕获偏移,replay 不再刷新。读到旧 layout 的 graph 在实时 layout 与捕获期不同时即用错误偏移计算。当前 main 上 PD-mux + TRT-LLM MLA 后端受影响(其 ragged verify 在图内用捕获的 `verify_lens`/`qo_indptr` pad/unpad);DSv4.1 target verify 的缺陷是潜伏的(其后端在图外拷贝实时 layout),但 #40567 的 ragged Engram hashing 与 `token_req_indices` 在 DSv4.1 verify 图内读 `qo_indptr`,使 `candidate_unfiltered` 图把 token 映射到错误的请求。

## 修复方案
`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` `_capture_ragged_verify_layout`:先查 `self._captured_ragged_layouts.get(num_tokens)`,已存在则直接返回同一对象——每个 token tier 只创建一次 layout,该 tier 的每个 graph 变体都捕获 replay 会刷新的同一组张量。新增 CPU 单测 `TestRaggedVerifyLayoutPerTier`(对 tier 24 用两个 variant 标签各调用一次,stage 实时 layout 后断言两个 layout 的 `qo_indptr_device`/`verify_lens` 与之一致;main 上第一个 layout 保留合成偏移,9 个 `qo_indptr` 项中 7 个不同)。

```diff
+        layout = self._captured_ragged_layouts.get(num_tokens)
+        if layout is not None:
+            return layout
         from sglang.srt.speculative.ragged_verify import (
             RaggedVerifyLayout,
             build_capture_verify_lens,
```

## 影响与回归风险
- 影响窗口:正文未说明(与 compact ragged verify、多 graph 变体/PD-mux 并存的场景相关;#40567 在图内读 `qo_indptr` 后暴露面扩大)。
- 回归面:decode CUDA graph 的 ragged verify layout 捕获路径;capture 阶段改为复用每 tier 单一 layout,replay staging 代码未变;`SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE` 强制路径与 `ragged_verify_mode` 关闭时返回 None 的行为不变。未做性能基准,正文预期无影响。

## 关联
- https://github.com/sgl-project/sglang/pull/40567 (使其暴露的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
