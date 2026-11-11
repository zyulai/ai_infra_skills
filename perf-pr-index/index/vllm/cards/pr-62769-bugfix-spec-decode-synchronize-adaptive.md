# PR #62769: [Bugfix][Spec Decode] Synchronize adaptive DSpark decisions across TP ranks

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd(co-authored with @CedricHwong) |
| 合并 | 2026-11-11 |
| 分类 | Speculative Decoding(major) |
| Label | bug, intel-gpu, speculative-decoding, ready, nvidia, mrv2, dflash |
| Diff | +84/-0,2 files(素材未提供 diffstat,按 pack diff 计数) |
| 链接 | https://github.com/vllm-project/vllm/pull/62769 |

## 一句话结论

DSpark 自适应验证改为以 TP rank 0 为权威广播 draft 预算与 capacities,消除 rank 间批布局分歧导致的 collective hang,单步开销约 7-68us(0.4-1.4%)。

## 动机

Adaptive verification(DSpark)在 host 端用 rank-local stale confidences 选每步 draft budget,在 device 端用 rank-local live confidences 选保留哪些 slot。TP rank 只要在任一处不一致(哪怕 1 个 fp32 ULP),各 rank 的 batch 布局就不同:token 总数不同,或总数相同但 request 边界不同——这会破坏随后的 collectives 并可能 hang(#54011)。本 PR 是把闲置的 #56831(CedricHwong)rebase 到 main 后落地,原提交保持不变,另加两个 review 修正 commit。

## 优化点

让 rank 0 成为唯一决策权威,两处广播:1) `get_num_tokens` 中 `draft_budget` 经 `broadcast_object`(shm MessageQueue,仅 host)广播;2) `reallocate_drafts` 在发生裁剪的步骤,构建 `cu_num_logits`/`query_start_loc` 之前由 rank 0 广播 per-request `capacities`(c10d `get_tp_group().broadcast`,TP 设备组)。两个分支条件(`draft_budget == 0` 与 `draft_budget < scheduled drafts`)只依赖广播后的 budget 与 scheduler 提供的计数,所有 TP rank 必走同一分支,collective 不会被 uneven 进入。传输层沿用 #56831 的 c10d;review 指出经设备 communicator(PyNCCL)路由每调用便宜约 2x、CUDA 上还能免去首个 trimmed step 才建 NCCL communicator,但要改所有 `GroupCoordinator.broadcast` 调用方(含 PCP `sync_drafts`)且需 XPU 验证,留作独立 PR。Scope:覆盖 TP;PCP=1 时的 DCP 也覆盖;PCP>1 时各 PCP slice 的 TP group 各有 leader,未覆盖。

```python
draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
# Rank-local confidences must not select different TP execution sizes.
draft_budget = get_tp_group().broadcast_object(draft_budget, src=0)
...
# Equal totals alone do not ensure identical request boundaries.
get_tp_group().broadcast(capacities, src=0)
```

## 关键改动

```text
vllm/v1/worker/gpu/spec_decode/adaptive_verification.py: get_num_tokens 广播 draft_budget;reallocate_drafts 在 trimmed step 广播 capacities(各 +2 行)
tests/v1/spec_decode/test_adaptive_verification.py: 新增真实多 rank 测试(gloo + 真 GroupCoordinator,置信度差 1 ULP 的 tie/boundary/64 随机步用例)
```

## 性能声明

MI355X 实测:
- 每步成本 p50(p90):budget `broadcast_object` 6.7us(7.5)@TP2、7.9us(8.2)@TP4(有 drafts 的每步,host only);capacities broadcast 仅 trimmed step,约 32+25us @TP2、33+26us @TP4(host enqueue + in-stream GPU time,1-256 reqs 不变);无重叠的最坏 trimmed step 约 64us @TP2、68us(p90 ~76)@TP4,占 5-15ms decode step 的 0.4-1.4%。PyNCCL follow-up 可降至约 25-29us/trimmed step。
- E2E TP2(Qwen3-4B + `dspark_qwen3_4b_block7`、`enable_adaptive_verification`、V2 model runner、ROCM_ATTN;600 GSM8K CoT prompts、output 512、greedy;capacities sync 用 PyNCCL 的早期修订,c10d 路径当时未跑 e2e):conc 1: 721.0±4.1 → 727.7±10 tok/s(+0.9%);conc 16: 5489±12 → 5493±15(+0.1%);conc 64: 10590±50 → 10585±19(-0.1%);TPOT p50 1.311→1.314 / 2.755→2.763 / 5.615→5.629 ms;acceptance 变化 ±0.7% 以内,24 轮 0 failed requests。
- 无修复时,单个 fp32 ULP 即可改变布局,每 1000 步 mismatch(TP2/4/8):1-16 ULP 0-2;1e-4 相对漂移 2-9;1e-2 相对漂移 65-382。
- 自然分歧在该设置未观察到:instrumented main 跑 8434 步 @TP2、7886 步 @TP4、DeepSeek-V4.1 TP4 约 478k AV 步均 0 mismatch——custom all-reduce 固定 rank 求和顺序使 confidences bitwise 相同;本修复是针对 rank-asymmetric numerics(如 #54011 的 TP8 H20 DeepSeek-V4)的廉价保险。

## 验证与回归风险

`test_adaptive_verification.py` 31 passed / 7 skipped(CUDA-only kernel 测试),mypy/ruff 干净;多 rank gloo harness 在 TP2/4/8、1 ULP 到大漂移各级均 0 mismatch;真实 rank 测试 10/10 次通过。消融验证:去掉 budget 广播则 budget tie 失败(36/64 @TP2、53/64 @TP4),去掉 capacities 广播则 boundary case 失败(4/64 @TP2、10/64 @TP4)。风险:PCP>1 未覆盖(#61994 后 NVIDIA 上 DeepSeek-V4.1 拒绝 AV+PCP,但通用 `PCPManager` 路径仍放行,属 follow-up);每步 host 广播的固定开销;DSpark CUDA kernel 测试路径。

## 关联

Supersedes #56831(原作者 CedricHwong)。前序/相关:#54011(hang 报告);#63421(Triton budget kernel)、#58466、#56888、#55939(rebase 时适配的主线改动);#61994(AV+PCP 限制);后续:PyNCCL transport 独立 PR。本批同分类:#58094。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
