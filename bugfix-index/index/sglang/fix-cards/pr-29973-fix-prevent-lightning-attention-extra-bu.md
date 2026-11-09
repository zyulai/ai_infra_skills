# PR #29973: [FIX] Prevent Lightning Attention extra-buffer mamba state corruption

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#29973](https://github.com/sgl-project/sglang/pull/29973) |
| 作者 | fkuner |
| 合并日期 | 2026-07-22 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 2 · 💬 6 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
Lightning `seg_la` 在 extra_buffer 下把整段 extend 的末态写进较短前缀的缓存槽,radix cache 复用过快推进的 mamba 状态,AIME 精度随轮次崩塌;kernel 内按边界落盘修复。

## 问题现象
Lightning `seg_la` 与 mamba radix cache `extra_buffer` 同用时,多次 serving 轮次复用被污染的 mamba 状态快照,AIME 精度崩塌:同一 server 连续 10 轮 AIME(Ling-2.6-flash,TP4,`--mamba-scheduler-strategy extra_buffer` + NEXTN 投机),修复前 accuracy 从 R1 的 0.80 逐步跌到 R8-R10 的 0.03-0.10,R7 起出现 12-19 个 extraction errors;修复后 10 轮稳定在 0.80-0.87,extraction errors 全 0。单次隔离 forward 下几乎看不出问题,radix cache 条目被反复插入和复用的端到端多轮 serving 中才暴露。

## 根因
PR 正文披露:extra_buffer 模式下,缓存的 mamba 状态存放在 `req_to_token_pool` 管理的额外乒乓槽位,active 槽位继续持有运行中请求的状态;scheduler 决定 mamba 边界并经 `ForwardBatch` 的 `mamba_track_mask`/`mamba_track_indices`/`mamba_track_seqlens` 传给模型 forward。radix entry 存在 `mamba_last_track_seqlen`(最后一个完整 `mamba_cache_chunk_size` 边界),该边界可能落在当前 extend 的**中间**而非末尾。修复前,`seg_la` 不物化非对齐 extend 所需的中间递归状态,Lightning 可能把整个 extend 的最终状态复制进与较短缓存前缀关联的 extra-buffer 槽位,造成「radix 前缀 [0,P) ↔ mamba state at E(E>P)」的错位;后续 prefix cache hit 恢复较短 token/KV 前缀时带着过度推进的 mamba 状态,被污染的快照复用后精度雪崩。与 #29945 的关系:后者已在全局层面修复 deferred Mamba COW/clear 顺序(移入 `ModelRunner._forward_raw`),本 PR 基于含 #29945 的 main,只修与此独立剩下的 Lightning tracked-boundary bug。

## 修复方案
1. `python/sglang/kernels/ops/attention/linear/seg_la.py`:prefill kernel 增加 `track_lens`/`track_state_indices` 入参与 `TRACK_STATE` 编译期开关;当 extend 跨越被追踪的 mamba 边界时,在循环内到达 `track_len` 的块把该边界状态写入 extra-buffer 槽位,active 槽位的既有末态写入保持不变:

```python
should_track = (
    (track_state_offset >= 0) & (track_len > 0) & (n + b == track_len)
)
tl.store(track_s_ptrs, state.to(S.dtype.element_ty), mask=should_track)
```

2. `python/sglang/srt/layers/attention/linear/lightning_backend.py`:把共享 Mamba tracking 元数据传入 Lightning metadata;新增 `_prepare_seg_la_track_store` 为 extend 批次准备紧凑追踪元数据(最后完整 chunk 边界 + 每个被追踪请求的目标 extra-buffer 槽位);对齐边界继续复制末态,decode 追踪复用共享 base-backend 辅助;seg_la 的 extra-buffer 拷贝改用 `track_ssm_final_src/dst` 元数据,不再用原来对所有被追踪行做 active→dst 复制的通用路径。
3. `test/registered/attention/unittests/lightning/test_triton.py`:新增针对非对齐 extra-buffer 路径的 kernel 级回归测试(含真实 serving 的 int64 track-slot index)。

## 影响与回归风险
- 影响窗口:正文未说明;受影响面为 `linear_backend=seg_la` + `--mamba-scheduler-strategy extra_buffer` 的部署(Ling-2.6-flash 等线性注意力模型)。
- 回归面:`seg_la` prefill kernel 签名新增可选参数,`track_lens`/`track_state_indices` 为 None 时 `TRACK_STATE=False`,非追踪路径行为不变;target_verify 模式跳过 track store;lightning_backend 仅 seg_la 分支改用新元数据拷贝,其他线性后端保留原 `ssm_states[dst] = ssm[src]` 逻辑;decode 路径新增 `_track_mamba_state_decode` 调用。若 `h_dst` 槽位数与追踪行数不一致会显式抛 RuntimeError(新增的硬检查)。

## 关联
- #29945(全局 deferred Mamba COW/clear 顺序修复,本 PR 与其互补)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
