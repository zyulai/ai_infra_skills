# PR #11398: Beta spec-overlap for EAGLE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2025-10-12 |
| 分类 | Speculative Decoding (tier: major) |
| Label | high priority, run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(8 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/11398 |

## 一句话结论

EAGLE 投机解码引入 beta 版 draft/verify overlap:`SGLANG_ENABLE_SPEC_V2=1`(或 `--enable-beta-spec`)开启后,单请求 decode 246.80→273.43 token/s(约 +10.8%)。

## 动机

此前的 EAGLE 实现要求 `disable_overlap_schedule`:调度器与 draft/verify 严格串行,每步 verify 结束后才能准备下一批的 KV 分配、seq_lens 更新与下一个 draft 输入,overlap 调度的好处完全无法惠及投机解码。本 PR 是 EAGLE v2 的 beta 起点:把 draft 与 verify 重叠起来,draft 状态通过 future 缓冲在 forward 流内接力,并配套 CUDA-graph 感知的执行与缓存管理。

## 优化点

新增独立于主路径的 `EAGLEWorkerV2`(enable_overlap 时由 `launch_draft_worker` 选择),核心机制:(1) `FutureMap` 扩展为携带完整 draft 输入状态——`topk_p`、`topk_index`、`hidden_states`、`verified_id`、`new_seq_lens` 五个 future 缓冲(按首个 draft input 的 shape 懒初始化),`store_to_map`/`resolve_future` 在 forward 流上把上一步的 draft 状态接力给下一步,替代 CPU 侧 token id 往返;(2) `ScheduleBatch.allocate_for_eagle_v2` 在 draft 前按 `ALLOC_LEN_PER_DECODE` 增量预分配 KV(含 `assign_req_to_token_pool` Triton kernel),配 `maybe_wait_verify_done` 仅在需要正确 seq_lens 时同步 verify 完成 event;(3) 新 `ForwardMode.DRAFT_EXTEND_V2` 支持 v2 worker 的定长 logits 输出;attention 后端新增 `get_verify_buffers_to_fill_after_draft`/`update_verify_buffers_to_fill_after_draft` 接口,使 tree mask/position 缓冲可在 draft 后填入、verify 元数据免于重算(triton 后端已实现,CUDA graph 下复用 `cuda_graph_custom_mask`);(4) 输出处理侧新增 `hacky_process_eagle_overlap_result`,按 accept lens 从定长 draft-token 输出中截取已接受 token,并配套 page_size=1 的投机 token 释放逻辑。代码中留有大量 FIXME,作者明示"完全分离 beta 与主路径",后续路线图与设计文档待发。

## 关键改动

```text
python/sglang/srt/speculative/eagle_worker_v2.py: 新增 EAGLEWorkerV2(draft/verify overlap、relay、verify_done event)
python/sglang/srt/managers/overlap_utils.py: FutureMap 扩展五个 draft 状态 future 缓冲
python/sglang/srt/managers/schedule_batch.py: allocate_for_eagle_v2 增量 KV 预分配与 maybe_wait_verify_done
python/sglang/srt/layers/attention/base_attn_backend.py: verify buffer 填充接口(draft 后填 tree mask/position)
python/sglang/srt/server_args.py: --enable-beta-spec / SGLANG_ENABLE_SPEC_V2,EAGLE 不再强制关 overlap
```

```python
+        if self.speculative_algorithm == "EAGLE" and self.enable_beta_spec:
+            self.disable_overlap_schedule = False
```

## 性能声明

单请求 `python -m sglang.test.send_one`:Meta-Llama-3.1-8B-Instruct + `lmsys/sglang-EAGLE-LLaMA3-Instruct-8B` 草稿模型,fp16、triton attention、`speculative-num-steps 5`、eagle-topk 1、6 draft tokens、`--cuda-graph-bs 1..64`:

- Beta Eagle(v2):acc_length=3.01,speed=273.43 token/s。
- Main Eagle:acc_length=2.99,speed=246.80 token/s(约 +10.8%)。
- 硬件型号 PR 未注明;除该单请求测试外无端到端吞吐数据。

## 验证与回归风险

- 正文提到"Adds end-to-end GSM8K-style evaluations and includes them in the per-commit test suite"(CodeRabbit 摘要),diff 中含 e2e 评估文件(8 个文件未展示,细节不可见)。
- 回归面:仅当 `--enable-beta-spec`/`SGLANG_ENABLE_SPEC_V2` 时改变行为,默认 EAGLE 仍强制关闭 overlap;实现带大量 FIXME(不支持 mixed mode 的 prefill batch、不支持 page_size>1 的投机释放、seq_lens 严格同步等),作者定位为 beta。CI 状态未随 PR 附上(截断部分不可见)。

## 关联

- 同分类:后续 spec-overlap 系列;#46184(mamba 验证窗口优化)同属投机验证路径;DFLASH #22077 为另一投机算法。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
