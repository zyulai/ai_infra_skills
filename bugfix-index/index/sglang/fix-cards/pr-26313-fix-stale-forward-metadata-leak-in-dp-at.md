# PR #26313: Fix stale forward_metadata leak in DP attn unpadded idle batch

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26313](https://github.com/sgl-project/sglang/pull/26313) |
| 作者 | ispobock |
| 合并日期 | 2026-05-25 |
| labels | run-ci, run-ci-extra |
| 热度 | 👍 0 · 💬 13 |
| 分类 | Other |
| bug 类型 | resource |

## 一句话结论
DP-attention 下无本地工作的 rank 走 forward_idle 且 unpadded(batch_size==0)时不重建 metadata,残留上一轮的 DSV4RawDecodeMetadata 被无条件 upgrade,重读已 free 的 KV 槽位造成 use-after-free;idle 分支显式置 None 修复。

## 问题现象
PR #26157 在 SWA `full→swa` mapping_stack 上加的 debug canary 暴露了 DSv4 disaggregation + DP-attention 下的竞态:canary 观察到 `mapping_stack[full_slot] == (producer=0, canary=SENTINEL)`,即 full-slot 在 `free_swa` 写入 sentinel 之后仍被 gather。逐项排查(EAGLE、PD、write_through_selective 逐个关闭)后确认:当且仅当开启 DP-attention 时复现,其余三者既非必要也非充分。该链路正是 disagg + DP-attention 跑 gsm8k 精度回归的来源。生产环境中被释放槽位一旦重新分配给其他请求,就是一次静默的跨 rank use-after-free。

## 根因
正文披露完整链条:DP-attention 要求每个 rank 都进入 `forward` 以完成 MLP all-gather,无本地工作的 rank 执行 `ModelRunner.forward_idle`;在 unpadded 情形(`batch_size == 0`,MLP-sync 不为该 rank padding)下,`forward_idle` 完全跳过 `init_forward_metadata`,`attn_backend.forward_metadata` 于是残留上一次 forward 的旧 `DSV4RawDecodeMetadata`(如 warmup-A 最后一次 decode 的 `req_pool_indices=[1]`、`seq_lens=[13]`、`out_cache_loc=[268]`)。随后 `DeepseekV4Model.forward` 在层前无条件调用 `get_attn_backend()._maybe_upgrade_forward_metadata()`(deepseek_v4.py:1270),该调用位于任何 per-layer 模块之外,`self_attn.forward` 的 `x.shape[0] == 0` 早退保护不了它。于是 `make_forward_metadata_from_raw_decode(rpi=[1], seq_lens=[13])` 会:1) 重读 `req_to_token[1, :13]`(free 时不清理,仍是 full-slots `[256..268]`);2) 对这些槽位调 `translate_loc_from_full_to_swa(...)`;3) 而 warmup-A 的 `free_swa` 已在 `mapping_stack[256..268]` 写入 `(producer=0, canary=SENTINEL)`,canary 触发。

## 修复方案
在 `python/sglang/srt/model_executor/model_runner.py` `forward_idle` 的 unpadded IDLE 分支显式丢弃残留 metadata:

```python
if forward_batch.batch_size > 0:
    self.attn_backend.init_forward_metadata(forward_batch)
else:
    self.attn_backend.forward_metadata = None
```

`_maybe_upgrade_forward_metadata` 对 `None` 本就是 no-op(两个 isinstance 分支都不命中),无需后端侧再加守卫;padded IDLE 路径(batch_size > 0,如需要真实 batch_size 视图的 DSA Indexer)不变。对非 DSv4 后端同样安全:`AttentionBackend.forward` 会在 idle 模式下先于访问 `forward_metadata` 早返(base_attn_backend.py:101),None 不会被解引用。

验证:DeepSeek-V4-Flash-FP8,disagg + DP-attention + EAGLE,8×H200,gsm8k(benchmark/gsm8k/bench_sglang.py),main 加 canary 精度 0.010、延迟 142.31s;本 PR(同样插桩,canary 不再触发)精度 0.975、延迟 14.12s;5/5 复现迭代中 main 必触发、本分支零触发。

## 影响与回归风险
- 影响窗口:正文未说明;从机制看,凡 DSv4/raw-decode metadata upgrade 与 DP-attention idle 路径并存的版本均受影响(由 diff 推断)。
- 回归面:改动仅 `forward_idle` 一个 else 分支。unpadded idle batch 下 `forward_metadata` 由「残留旧值」变为「None」,依赖 metadata 的路径只剩 `_maybe_upgrade_forward_metadata`(对 None no-op)与后端 forward(idle 早返);若未来有后端在 idle 模式下仍读取 `forward_metadata`,将遇到 None。padded idle 与正常 forward 路径不变。

## 关联
- PR #26157(引入 canary 的调试 PR): https://github.com/sgl-project/sglang/pull/26157

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
