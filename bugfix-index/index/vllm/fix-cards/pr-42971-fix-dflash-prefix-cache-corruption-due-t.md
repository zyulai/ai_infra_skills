# PR #42971: Fix DFlash prefix cache corruption due to missing lookahead block

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42971](https://github.com/vllm-project/vllm/pull/42971) |
| 作者 | shreyas269 |
| 合并日期 | 2026-06-02 |
| labels | bug, dflash, ready, speculative-decoding, v1 |
| 热度 | 👍 2 · 💬 8 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
DFlash 首次 prefill 未分配 lookahead 块,drafter 越界写入共享前缀块造成 KV 损坏、并发下 MGL 持续劣化;调度器修复落 #43733,本 PR 修 drafter 输入适配检查。

## 问题现象
高并发 + prefix caching 下使用 DFlash 时,本运行正常的请求出现 MGL(median generation length)持续劣化:新到达的前缀命中请求的首个 DFlash drafter pass 把 KV 写进共享前缀块,污染其他请求的 KV cache。作者在同样负载下对比 sglang 无此劣化,确认是 vLLM 特有 bug。无 prefix caching 时 MGL 影响可忽略——损坏块只属于当前请求,drafter 坏一步后即恢复。

## 根因
正文完整披露:target prefill 与首个 drafter pass 在同一调度步执行,DFlash drafter 会以 `num_spec_tokens + 1` 个 query token(1 个 bonus token + N 个 mask token)在 prefill 之后的位置发查询;DFlash draft attention 是双向的,drafter 经 block table 查表在「未来位置」写 KV,若 block table 未覆盖这些位置就会写错块。而调度器为新请求分配块时有:

```python
effective_lookahead_tokens = (
    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
)
```

该守卫本是 P/D 分离场景的边界保护,但 `allocate_slots` 在新请求首次调度时 `num_computed_tokens` 仍为 0(它在分配成功之后才被更新),于是 prefill 时零 lookahead 块被分配;正确块不存在时 drafter 把 KV 写进最后一个 prefill 块——在 prefix caching 下该块被多请求共享,静默覆写有效 prefill KV。

## 修复方案
调度器侧修复(为 DFlash 在首次 prefill 预留 `num_spec_tokens + 1` 个 lookahead 槽、收窄 `num_computed_tokens == 0` 守卫)已单独落在 #43733,本 PR rebase 其上、不再触碰 scheduler。本 PR 改 `vllm/v1/worker/gpu_model_runner.py`:新增 `_input_fits_in_drafter`,把原先 `max_seq_len + num_spec_tokens` 的适配判断改为 `num_drafter_query_tokens = self.num_spec_tokens + (1 if spec_config.use_dflash() else 0)`,避免 DFlash 序列被错误放行进入 drafter(其实际查询 `num_spec_tokens + 1` 个位置)。新增 `tests/v1/spec_decode/test_dflash_lookahead.py` 三个单测(prefill 预留 lookahead 块、首 prefill 查询窗口落在已分配块内、drafter 窗口为 bonus token 预留),全部通过。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围;影响启用 DFlash + prefix caching + 并发的 v1 部署。
- 回归面:DFlash 的 drafter 输入适配判断(仅 use_dflash 时多留 1 个 bonus token 位置,其他 drafter 逻辑不变);非 DFlash 的投机方法不受影响。调度器改动在 #43733,需一并合入才完整。

## 关联
#43733(调度器侧修复,本 PR rebase 之上)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
