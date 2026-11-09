# PR #36228: [CP V1 Deprecation 3/5] Remove generic prefill CP v1 runtime

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2026-09-07 |
| 分类 | Other (tier: major) |
| Label | deepseek, run-ci, mthreads, bypass-fastfail, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用(21 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/36228 |

## 一句话结论

CP v1 弃用计划第 3/5 步:删除 generic prefill CP v1 运行时与 DSA v1 in-seq-split 路径,统一到 CP v2 strategy,缩小维护面。

## 动机

generic(非 protected 平台)的 prefill context parallel 同时存在 v1 旧分支与 v2 strategy 路径,v1 的 in-seq-split 模式、legacy server-arg 投影与多余的 wrapper 增加维护负担且限制调度(单请求调度限制)。这是五步弃用计划的第三步,目标是在保住 HIP/NPU/MUSA protected 平台契约的前提下删除 generic CUDA 上的 v1 运行时。

## 优化点

删除为主:(1) 移除 generic prefill CP v1 分支与 DSA v1 in-seq-split 的 indexer/backend 路径(`_get_topk_ragged_with_cp` 约 160 行、`_forward_trtllm` 的 zigzag seq_lens 重排、`cp_attn_forward_extend`、`is_dsa_prefill_cp_in_seq_split`/`is_prefill_cp_in_seq_split` 等);(2) 移除废弃的单请求调度限制;(3) 在 runner/strategy 边界保留 generic 输入 sharding 与 token-ID layout 交接(`cp_shard_model_inputs` 新增 input_ids 分片与 `input_ids_global` 备份恢复);(4) 删除多余的 distributed-layout、projection-cache-write、communicator-factory wrapper,DeepSeek V4 attention/decoder 层回归基线实现(含 HIP KV-score prelaunch);(5) protected 平台判定从 HIP/NPU 扩为 HIP/NPU/MUSA,legacy 参数投影仅在 protected 平台生效,`layers/utils/cp_utils.py` 重定位为 HIP/NPU/MUSA 专用 legacy helpers;`materialize_full_kv_cp` 中 legacy DSA fallback 显式 `assert is_hip()`。review 中要求新增的架构/runner 回归测试被移除,保留既有 CP suite。

## 关键改动

```text
python/sglang/srt/layers/attention/dsa/dsa_indexer.py: 删除 _get_topk_ragged_with_cp 与 in-seq-split 分支
python/sglang/srt/layers/attention/flashattention_backend.py: forward_extend 统一走 is_cp_v2_active/cp_strategy,删 v1 helpers
python/sglang/srt/layers/cp/utils.py: enable_cp_v2 收紧(排除 MUSA)、mla_use_prefill_cp 等判定迁移进来
python/sglang/srt/arg_groups/parallel_hook.py 等: legacy CP 参数投影仅限 HIP/NPU/MUSA
```

## 性能声明

PR 未给出性能数字;本 PR 属架构清理,收益为路径统一与维护面缩小(调度限制移除的运行时影响未单独量化)。

## 验证与回归风险

验证(合并前 head `044a6086ec`/`3e40652a67` 的官方 CI,历史结果):CP strategy CPU 通过;H200 DeepSeek V3 CP + DSA prefill CP 通过(GSM8K 0.954 / 0.942);B200 DeepSeek V4 FP4 CP + GPT-OSS MXFP4 CP 通过;H100 GQA prefill CP 通过;B300 GLM-5.2 PD/MTP/CP layer-split 通过(GSM8K 0.943);MLA dispatch 与 server-args CPU 回归、lint 通过。CPU fixture 缺陷(`test_index_topk_share.py` 的 forward_mode test double 未实现 `is_context_parallel_extend()`)在 follow-up `9656d8afb4` 修复,11/11 通过。风险面:generic CUDA 不再支持 v1 语义(激活改由 CP-v2 策略选择 absorbed MLA 等);HIP/NPU/MUSA 的 legacy 契约保留但需后续独立迁移;合并时 Latest PR Test (Base)/(Extra)/AMD ROCm 7.2 为失败/跳过状态。

## 关联

Stack:#36222(CP 测试 v1→v2,已合并)、#36223(移除 v1 server-arg 兼容,已合并)、本 PR(3/5)、#36229(重命名 CP v2 API 与 round-robin 术语)、#36230(文档)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
