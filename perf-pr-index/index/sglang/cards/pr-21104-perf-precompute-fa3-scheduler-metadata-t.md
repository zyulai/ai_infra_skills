# PR #21104: perf: precompute FA3 scheduler_metadata to eliminate per-layer prepare_varlen_num_blocks

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @zminglei |
| 合并 | 2026-04-10 |
| 分类 | Overlap / Scheduler(major) |
| Label | run-ci |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 为单文件 Python 改动(flashattention_backend.py) |
| 链接 | https://github.com/sgl-project/sglang/pull/21104 |

## 一句话结论

每 batch 一次预计算 FA3 `scheduler_metadata` 并传入 `flash_attn_with_kvcache`,让 FA3 不再每层跑内部 `prepare_varlen_num_blocks` kernel,每 decode 步省 63 次 GPU kernel 调用,decode 吞吐 +1.4-2.6%。

## 动机

FA3 的 `flash_attn_with_kvcache` 在未显式提供 scheduler_metadata 时,会在每一层内部启动 `prepare_varlen_num_blocks` kernel 来计算调度信息。64 层模型每个 decode 步因此重复 64 次相同的每 batch 计算(仅 batch 相关、与层无关),纯浪费;vLLM 已经通过预计算规避了这一开销。

## 优化点

在 `flashattention_backend.py` 的 decode metadata 初始化阶段(含 CUDA graph 三条路径)按 batch 调用一次 sgl-kernel 的 `get_scheduler_metadata`(#21103 提供的新 API),结果存入 `FlashAttentionMetadata.scheduler_metadata`,decode forward 时传给 `flash_attn_with_kvcache`,FA3 内部即跳过 per-layer kernel。约束:metadata 仅对非 SWA、非 cascade attention 的单 token decode 有效(`_compute_scheduler_metadata` 里跳过 MLA,forward 里按 `is_swa_layer`/`use_cascade_attn` 判定不传);CUDA graph 路径预分配固定大小的 int32 buffer(1 + round_up(max_bs,4)×4),capture 与 replay 时写入同一 buffer 保持 graph 地址稳定。依赖 sgl-kernel 含 `get_scheduler_metadata`,否则 `_get_scheduler_metadata=None` 优雅 no-op。

## 关键改动

```text
python/sglang/srt/layers/attention/flashattention_backend.py: 唯一改动文件
  - FlashAttentionMetadata 新增 scheduler_metadata 字段
  - init_forward_metadata / init_cuda_graph_state / capture / replay 四条路径预计算
  - forward_decode 传递 scheduler_metadata=sched_meta 给 flash_attn_with_kvcache
```

关键机制片段(消费侧判定):

```python
sched_meta = None
if (metadata.scheduler_metadata is not None
        and not is_swa_layer and not use_cascade_attn):
    sched_meta = metadata.scheduler_metadata
result = flash_attn_with_kvcache(..., scheduler_metadata=sched_meta, ...)
```

## 性能声明

PR 注明条件:Qwen3-32B TP=4,4x H200,3 次运行取均值(每次 fresh server restart):

- BS=1 decode-heavy(input 500/output 8K):吞吐 131.41→134.83 tok/s(+2.6%),TPOT 7.61→7.41ms(-2.6%)。
- BS=4 同 workload:+2.3% 吞吐;BS=16:+1.9%。BS=8 balanced(2K/2K):+2.2%。BS=8 prefill-heavy(8K/500):+1.4%。TTFT 均无回归(±1% 内)。
- 固定省约 200us/步,收益与 TPOT 成反比(BS=1 TPOT 7.6ms → 2.6%;TPOT 10.1ms → 1.8%),并推算 TP=8(TPOT 约 4ms)可达约 5%。
- profile 验证:`prepare_varlen_num_blocks` 调用从 576→72 次/窗口(64 次 prefill + 8 次 get_scheduler_metadata,decode per-layer 调用全部消除)。

## 验证与回归风险

GSM8K 1319 题并行 1319:baseline 0.858 vs optimized 0.861,无回归(采样方差内)。风险:scheduler_metadata 仅在非 MLA、非 SWA、非 cascade 的 decode 路径生效,SWA 层仍走每层内联计算;CUDA graph buffer 大小按 max_bs 预分配,`_sched_meta_buf[:n]` 拷贝在 capture/replay 时执行;依赖 #21103 的 sgl-kernel 新 API,旧 sgl-kernel 下自动 no-op 不劣化;prefill 路径未动。

## 关联

前序 PR #21103(sgl-kernel 提供 `get_scheduler_metadata`,本 PR 为其 sglang 侧第 2/2 部分)。对标 vLLM 的同机制实现。其余同分类 PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
