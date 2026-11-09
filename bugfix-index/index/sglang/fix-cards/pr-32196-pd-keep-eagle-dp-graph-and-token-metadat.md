# PR #32196: [PD] Keep EAGLE DP graph and token metadata consistent

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32196](https://github.com/sgl-project/sglang/pull/32196) |
| 作者 | weireweire |
| 合并日期 | 2026-09-29 |
| labels | bug, run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 0 · 💬 29 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
PD decode 中 target 与 draft 可用不同 MoE A2A 后端,scheduler 把 target 侧 token 数给了 draft,DP rank 间 padding/集合形状不一致,还会因缺 GLM DSA seed 只有部分 rank 回退 eager;新增 draft 专用 token 计数与跨 rank 同步的 graph 准入门控。

## 问题现象
修复 issue #32182。PD(prefill-decode 分离)decode 下,DP 各 rank 对 draft 的 padding 和集合通信形状不一致,导致错误/挂起;GLM DSA seed 缺失时只有活跃 rank 回退 eager,同样造成 rank 间行为分歧。正文未附具体报错日志。

## 根因
正文披露的根因有两点:

1. PD decode 中 target 与 draft 可以使用不同的 MoE A2A 后端,但 scheduler 一直把 target 侧的本地 token 计数(target-local token counts)交给 draft worker,导致 DP rank 之间对 draft 的 padding 与 collective 形状判断不一致。
2. GLM DSA seed 缺失时,只有缺 seed 的活跃 rank 回退 eager,其他 rank 仍走 CUDA graph,rank 间路径分叉。

## 修复方案
核心思路(正文):跨 DP rank 同步 graph 准入资格(graph eligibility),并为 draft 保留专属 token 计数,同时不改 target 的 DeepEP 路径。主要改动(可见 diff):

1. `managers/schedule_batch.py` / `forward_batch_info.py`:新增 `draft_global_num_tokens`、`draft_global_num_tokens_for_logprob`(draft 模型可用与 target 不同的 MoE A2A 后端)与 `can_run_dp_draft_cuda_graph`(draft 专属的 DP graph 门控,与 `can_run_decode_cuda_graph` 分开,避免缺 draft 状态时误禁 target verify/draft extend)。
2. `managers/scheduler_components/dp_attn.py`:`prepare_mlp_sync_batch_raw` 增加 `draft_require_mlp_tp_gather` 参数,复用同一次 all-gather 结果为 draft 填充 `draft_global_num_tokens`;新增 `_spec_input_cuda_graph_compatible`,把 spec 输入的 graph 兼容位加入 `MLPSyncBatchInfo` 并 min-reduce(`can_run_draft_cuda_graph` 取各 rank 最小值),强制全体 rank 一致 eager/graph。
3. `speculative/spec_info.py` / `eagle_info.py`:`SpecInput`/`EagleDraftInput` 增加 `cuda_graph_compatible` 位,`merge_batch` 中正确传播(空 stub 继承、非空取 AND)。
4. `speculative/eagle_disaggregation.py` / `eagle_worker_v2.py`:缺 DSA seed 且模型要求 seed(`_requires_dsa_seed_for_cuda_graph`:topk==1 且 `get_dsa_seed_metadata_dim>0`)时置 `cuda_graph_compatible=False`。
5. `eagle_draft_cuda_graph_runner.py` `can_run_graph`:额外要求 `forward_batch.can_run_dp_draft_cuda_graph`。
6. `layers/attention/dsa/dsa_indexer.py` / `dsa_backend.py`:eager IDLE rank 返回占位(-1)topk 结果并从 rank0 广播;`_forward_trtllm` decode 前按 metadata 的真实 batch 裁剪 q/topk(`_trim_trtllm_decode_dp_padding`),输出再补零恢复物理形状(`_restore_trtllm_decode_dp_padding`),保证下游 MLP/EP 集合形状一致。
7. `utils/common.py` `require_mlp_tp_gather` 支持显式传入 `moe_a2a_backend`;`forward_batch_info.py` 的 `post_forward_mlp_sync_batch` 对 idle 模式裁剪 positions/seq_lens 等;新增/更新 e2e 与单测(target DeepEP + TP-only EAGLE draft 的 PD 拓扑冒烟测试、wire 测试)。

验证:GLM-5.2-NVFP4 TP4/DP4 EAGLE GSM8K 0.948(474/500),500/500 请求完成,无 DSA/CUDA 错误或 rank 挂起。

注:diff 在 40000 字符处截断(文件粒度),截断部分未展示,本卡仅描述可见部分与正文。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 PD 分离 + DP attention + EAGLE 的 decode 路径:draft worker 的 token 计数来源、DP graph 准入投票(新增 min-reduce 位,某些场景会让全体 rank 回退 eager,牺牲性能换一致性)、DSA indexer/trtllm decode 的 padding 处理、`require_mlp_tp_gather` 的调用签名(默认行为不变)。target DeepEP 路径按正文声明未改。不使用 PD 分离或非 EAGLE 的路径受影响面较小,但 `ScheduleBatch`/`ForwardBatch` 新字段贯穿调度与执行器。

## 关联
- #32182(本 PR 修复的 issue,正文开头 "Fixes #32182")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
