# PR #31519: [Fix] Update PR #25015 revert for fused topk=1 draft postprocess

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31519](https://github.com/sgl-project/sglang/pull/31519) |
| 作者 | kpham-sgl |
| 合并日期 | 2026-07-17 |
| labels | (none) |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
#30947 重构使 #25015 canary 回滚补丁文本失配、server 起不来,更新 revert yaml 匹配新代码。

## 问题现象
`main` 分支上 `test/registered/kv_canary/test_self_e2e_pr_25015.py` 持续失败。该 canary 回归测试在 server 启动时通过 `pr_fix_toggle.py`(`SGLANG_DEBUG_REVERT_PR=25015`)对 `EagleDraftWorker.draft_forward` 做源码级 patch 以「回滚」历史 position 修复,再断言 KV canary 触发 `verify_position` 违规。回滚依赖精确的源码文本匹配,匹配失败时 `source_patcher._find_match` 抛出 `PatchApplicationError`,server 无法启动,测试直接报错:

```
[source_patcher] patching ...EagleDraftWorker.draft_forward
PatchApplicationError: match text not found in source:
hidden_states = logits_output.hidden_states
forward_batch.positions.add_(1)
```

## 根因
正文披露:元凶是 #30947([1/3] [EAGLE] perf: Fuse topk=1 draft postprocess),但并非 perf 改动本身的功能回归——#30947 重构了 `draft_forward`,把循环尾部单独的 `forward_batch.positions.add_(1)` 融进了 topk=1 CUDA 链路新增的 `draft_topk1_postprocess` Triton kernel 内部执行,导致 #25015 回滚补丁的第二个 edit 所匹配的文本(`hidden_states = logits_output.hidden_states` + `forward_batch.positions.add_(1)`)在源码中不复存在,补丁应用失败。

## 修复方案
更新 `python/sglang/srt/debug_utils/pr_fix_toggle.py` 中的 `_PR_REVERT_YAML_25015`:
- 保留第一个 edit(在 forward 之前注入一个时机错误的 `positions.add_(1)`)。
- 用新 edit 替换已失效的第二个 edit:在 `draft_topk1_postprocess(...)` 调用之后追加 `forward_batch.positions.sub_(1)`。kernel 内部已正确完成 post-forward 前进且无法从 `draft_forward` 源码中删除,因此再减回去就只剩提前的那次前进,忠实复现 #25015 之前的 off-by-one bug,让 canary 触发 `verify_position`。
- 另按新源码移除循环内其余位置的 `forward_batch.positions.add_(1)`(跟随 `draft_probs_list.append(probs)`、`topk_p = torch.ones_like(...)`、`fast_topk(...)` 三处匹配)。
- `capture_one_shape` 的 edit 未动(该文件未受 #30947 影响)。

验证:B200 GPU 上 `2 passed, 1 skipped, 15 warnings in 126.76s`,`TestEaglePositionsMatchWithFix`(修复在位→无违规)与 `TestEaglePositionsMisalignRegression`(回滚 #25015→canary 触发 `verify_position`)均 PASSED。

## 影响与回归风险
- 影响窗口:正文未说明(自 #30947 合入后、本修复合并前,canary 测试在 main 上持续失败)。
- 回归面:仅改动调试用 revert yaml(`pr_fix_toggle.py`),只影响依赖 `SGLANG_DEBUG_REVERT_PR=25015` 的 canary 回归测试,不影响正常推理路径。正文同时警示:这类「回滚历史修复」的 canary 测试与被 patch 函数的精确源码文本强耦合,后续对 `draft_forward` 的重构(如已合并的 #30948、开放的 #30949)仍可能再次悄悄打破它们。

## 关联
- [#25015](https://github.com/sgl-project/sglang/pull/25015)(被 canary 回滚的历史 position 修复)
- [#30947](https://github.com/sgl-project/sglang/pull/30947)(触发失配的 topk=1 draft postprocess 融合)
- [#30948](https://github.com/sgl-project/sglang/pull/30948)(已合并的后续重构)
- [#30949](https://github.com/sgl-project/sglang/pull/30949)(开放的后续重构)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
