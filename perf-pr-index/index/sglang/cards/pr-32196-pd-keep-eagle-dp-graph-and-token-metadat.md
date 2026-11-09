# PR #32196: [PD] Keep EAGLE DP graph and token metadata consistent

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @weireweire |
| 合并 | 2026-09-29 |
| 分类 | Speculative Decoding |
| Label | bug, run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/32196 |

## 一句话结论

修复 PD decode 下 EAGLE draft 的 DP 图准入与 token 元数据不一致,消除 rank hang 与 DSA/CUDA 错误。

## 动机

修复 #32182。PD decode 中 target 与 draft 可以使用不同的 MoE A2A backend,但调度器此前给 draft worker 沿用 target 侧的本地 token 计数,导致各 DP rank 对 draft padding 与 collective shape 的判断不一致。GLM DSA seed 缺失时,也可能只有活跃 rank 回退 eager 执行,rank 间状态失配,表现为请求挂起或 DSA/CUDA 报错。

## 优化点

引入 draft 专属的 DP 同步位 `can_run_dp_draft_cuda_graph`,经 `MLPSyncBatchInfo` 在 rank 间 min-reduce 投票,与 target 的 decode 图准入位解耦,避免 draft-only 状态缺失误伤 target 验证与 draft extend。新增 `draft_global_num_tokens(_for_logprob)`,按 draft 的 MoE A2A backend(`--speculative-moe-a2a-backend`)决定复用 all-gather 全局计数还是本地计数。`SpecInput`/`EagleDraftInput` 增加 `cuda_graph_compatible` 位:DSA seed 缺失时标记该输入不可回放图,并在 merge_batch 间 AND 传播;draft 图 runner 将其纳入 `can_run_graph`。DSA 侧:对 eager IDLE rank 返回 -1 占位 topk 行保证 MLP/EP collective 形状一致;TRTLLM decode 前按 metadata 真实 batch 裁剪 q/topk 行,注意力只算真实行,输出再补零恢复物理 DP 形状。

## 关键改动

```text
python/sglang/srt/managers/scheduler_components/dp_attn.py: draft 图准入投票(_spec_input_cuda_graph_compatible + min-reduce)与 draft token 计数同步
python/sglang/srt/managers/schedule_batch.py / model_executor/forward_batch_info.py: 新增 draft_global_num_tokens 等字段,draft worker 按 is_draft_worker 选用
python/sglang/srt/speculative/eagle_info.py / spec_info.py: cuda_graph_compatible 位及 merge/filter 传播
python/sglang/srt/layers/attention/dsa_backend.py: _trim/_restore_trtllm_decode_dp_padding
python/sglang/srt/layers/attention/dsa/dsa_indexer.py: eager IDLE rank 的占位 topk 结果
```

```python
def _spec_input_cuda_graph_compatible(local_batch):
    if (local_batch is None or local_batch.forward_mode.is_idle()
            or local_batch.forward_mode.is_prebuilt()):
        return True
    spec_info = local_batch.spec_info
    return spec_info is None or spec_info.cuda_graph_compatible
```

## 性能声明

PR 未给出吞吐/延迟加速数字,属正确性/一致性修复。验证结果:GLM-5.2-NVFP4 TP4/DP4 EAGLE GSM8K 得分 0.948(474/500),500/500 请求全部完成,无 DSA/CUDA 错误或 rank hang。PR 未注明修复前的对照基线数据。

## 验证与回归风险

新增单测 `test/registered/unit/layers/attention/test_dsa_backend_dp_padding.py`,覆盖 DP padding 模式选择、trim/restore 与 metadata 超出物理 batch 的拒绝路径;新增 e2e `TestGLM52DSACacheLayerSplitMixedMoE`,覆盖 target DeepEP low_latency + TP-only EAGLE draft 的 PD 拓扑(含 DP padding 压力)。改动横切 DP attention 同步、TBO filter_batch、DSA backend/indexer 与 EAGLE worker/disaggregation,影响所有 DP attention + EAGLE/standalone 及 DSA 模型路径;pack 中 CI 状态显示 AMD ROCm 10 一项失败,Base/Extra 通过。

## 关联

修复 #32182;代码注释引用 #30432 的 DP-sync bookkeeping 约定。同区域(DP attention + speculative)的其他 PR 素材未提供,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
