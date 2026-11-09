# PR #62784: [Spec Decode][MRV2] Skip drafting in steps where no request can use the drafts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zixi-qi |
| 合并 | 2026-10-22 |
| 分类 | Speculative Decoding |
| Label | speculative-decoding, ready, mrv2, dflash, scheduler |
| Diff | diffstat 缺失(pack 标注 +None/-None);可见 diff 涉及 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/62784 |

## 一句话结论

MRV2 上无请求能用 drafts 的步骤以 K=0 提议(跳过 draft 采样/decode 或 DFlash query pass),prefill-only 吞吐最高 +7.3%。

## 动机

drafts 在 step 后提议、在请求的下一个 decode step 验证。两类请求的 drafts 注定作废:非最终 prefill chunk(scheduler 本就丢弃 chunk 后的 drafts)与必然达到 `max_tokens` 的请求。P/D prefill 实例(`max_tokens=1`)上每一步都是如此,但 speculator 仍全量执行 draft 采样与 draft decode(EAGLE 系)或 query pass(DFlash),纯浪费。延续 @njhill 的 #48244;相关 draft #39266(P/D prefill 实例跳过 EAGLE drafting)。

## 优化点

(1) Scheduler:新谓词 `_drafts_unused` 只统计请求确定会有的输出 token(假设全部 draft 被拒绝),对非最终 chunk 与 `min_num_output_tokens >= max_tokens` 两种情况返回 True;`schedule()` 聚合为 `SchedulerOutput.skip_drafts`,仅在 MRV2 且 drafter 预留 lookahead slots 时启用(`can_skip_drafts`)。(2) DP:flag 通过既有的 batch-descriptor all-reduce 新增第 7 个字段 AND 归约成 `DPSyncState.skip_drafts`,dummy batch 的 rank 投反对票,防止空闲 rank 单方面跳过。(3) Runner:`skip_drafts` 时以 `num_speculative_tokens=0` 提议,复用既有 K=0 路径——EAGLE/EAGLE3/MTP 仍跑 draft prefill pass(保持 drafter KV cache 同步,P/D decode 实例传输的 KV 依赖它)但 `skip_sample` 跳过采样与 draft decode;DFlash 只存 context K/V、跳过 query pass(DSpark/DFlash2/LiLiCorr 共用该代码)。(4) 相对 #48244:删去 per-request lookahead-slot skip(需在 FULL captured graph 内屏蔽 draft KV 写,风险大),改用单一 scheduler flag。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: _drafts_unused 谓词 + schedule() 聚合 drafts_needed
vllm/v1/core/sched/output.py: SchedulerOutput.skip_drafts 字段
vllm/v1/worker/gpu/dp_utils.py: DP all-reduce 增第 7 字段,AND 归约 skip_drafts
vllm/v1/worker/gpu/model_runner.py: skip_drafts 时 num_spec_tokens_to_schedule 置 0(K=0)
vllm/v1/worker/gpu/spec_decode/{dflash,target_dependent_ar}/speculator.py: DFlash 早退存 KV;EAGLE 系 skip_sample
```

```python
# Case 2: this step samples the request's last output token...
min_num_output_tokens = request.num_output_tokens + 1
if request.num_output_placeholders > 0:
    min_num_output_tokens += 1
return min_num_output_tokens >= request.max_tokens
```

## 性能声明

环境:Qwen3-8B,`--attention-backend FLASH_ATTN`,GB300;drafter `AngelSlim/Qwen3-8B_eagle3`(K=3)与 `z-lab/Qwen3-8B-DFlash-b16`(K=16);基线 main @ 532a7d19d2。作者声明早期版本的 +9.0%(EAGLE3)/+3.4%(DFlash)因 benchmark 进程被限 1 个 CPU 核而虚高,以下数字替换之。

- Prefill-only 吞吐(64 x 16,384-token prompts,1 output token,双 GPU 同跑后互换):`max_num_batched_tokens=512` 时 EAGLE3 33,834→34,473(+1.9%)、DFlash 32,367→34,730(**+7.3%**);2048 时 +1.4%/+2.7%。host 侧 `propose()` 每步 1.41→0.98 ms(EAGLE3)/1.05→0.92 ms(DFlash)。
- 1P1D prefill 实例(NixlConnector,`max_tokens=8`):512 时 EAGLE3 +2.1%、DFlash +3.4%;2048 时 +0.6%/+1.9%;PR 侧全部 proposal 均为 K=0。
- 等价性:`VLLM_BATCH_INVARIANT=1` 下输出 80/80 相同、spec-decode 计数器逐项一致;1P1D/DP=2 serving 等价 40/40,decode 侧接受长度逐项相同。无 batch invariance 时 DFlash 在 K=0 步以单一 eager pass 存 context K/V,舍入可差。

## 验证与回归风险

新增单测:调度器 `test_skip_drafts_only_when_no_request_can_use_them`/`test_drafts_unused_with_async_placeholders`(async placeholder 最坏情形);DP `test_gpu_dp_utils.py` 验证 AND 归约组合;`test_gpu_ubatch_slicing` 同步 6→7 字段。既有 e2e(GSM8K 0.886、dflash 接受率/正确性)main 与 PR 结果相同;unit 281 项通过(2 个 PP scheduler 失败为 main 已有)。风险面:仅覆盖 MRV2 + 预留 lookahead slots 的 drafter;multi-module MTP 与 draft-model speculator 忽略 K 仍照常 draft;N-gram 不预留 lookahead slots,flag 永不置位;MTP/DSpark/DFlash2/LiLiCorr 共用代码但未实测(全部验证用 Qwen3-8B);MRV1 不变。

## 关联

同区域(MRV2 spec decode/调度):索引内 PR #52228(MRV2 spec decode adaptive verification)、PR #53183(MRV2 全模型默认,`use_v2_model_runner` 即本 PR 的启用条件之一)、PR #36847(DFlash)、PR #52816(DFlash2)、PR #46995(DSpark)、PR #47808(DSpark confidence scheduling);前序 #48244、相关 draft #39266、#53426。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
