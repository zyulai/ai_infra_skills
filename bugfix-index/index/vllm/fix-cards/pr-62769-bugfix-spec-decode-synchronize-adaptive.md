# PR #62769: [Bugfix][Spec Decode] Synchronize adaptive DSpark decisions across TP ranks

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#62769](https://github.com/vllm-project/vllm/pull/62769) |
| 作者 | okorzh-amd |
| 合并日期 | 2026-11-11 |
| labels | bug, dflash, intel-gpu, mrv2, nvidia, ready, speculative-decoding |
| 热度 | 👍 1 · 💬 37 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
自适应验证(DSpark)各 TP rank 用本地置信度独立决策,布局分歧可破坏并挂起集合通信;改为 rank 0 广播 budget 与 capacities 保证一致。

## 问题现象
正文描述:若 TP rank 之间对 per-step draft budget 或保留哪些 slots 判断不一致,各 rank 会执行不同的 batch layout——token 总数不同,或总数相同但请求边界不同。这会破坏其后的集合通信,甚至将其挂起(引用 #54011)。触发条件是 rank 间置信度输入存在数值差异(正文称「rank-asymmetric numerics」,如依赖 rank 顺序的归约、非确定性内核,以及 #54011 中 TP8 H20 DeepSeek-V4 的设置)。实测:无修复时,单个 fp32 ULP 的置信度差即足以改变布局;随机 64 步中 TP2 有 36 步、TP4 有 53 步发散。

## 根因
正文披露:Adaptive verification(DSpark)在 host 上用 rank 本地的 stale confidences 选 per-step draft budget,又在 device 上用 rank 本地的 live confidences 决定保留哪些 slots——两处决策都只依赖 rank 本地数据,rank 间数值一旦不一致,决策即分歧。正文还说明本 PR 是 #56831(原作者 CedricHwong)rebase 到当前 main 的携带版,原提交保持不变,其上两个提交处理 review 意见。

## 修复方案
让 TP rank 0 成为唯一权威,改 `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`:

1. `get_num_tokens` 中 `draft_budget` 用 `broadcast_object`(shm MessageQueue,仅 host)广播;
2. `reallocate_drafts` 中,发生裁剪的步在构建 `cu_num_logits`/`query_start_loc` 之前把 per-request `capacities` 从 rank 0 广播。

```python
draft_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
# Rank-local confidences must not select different TP execution sizes.
draft_budget = get_tp_group().broadcast_object(draft_budget, src=0)
...
# Equal totals alone do not ensure identical request boundaries.
get_tp_group().broadcast(capacities, src=0)
```

两处分支条件(`draft_budget == 0`、`draft_budget < scheduled drafts`)只依赖广播后的 budget 与 scheduler 计数,所有 rank 必走同一分支。另新增真实多 rank 测试(gloo + 真实 `GroupCoordinator`,TP 2/4,置信度差 1 ULP)。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本范围,不写。
- 回归面:按 diff,`get_num_tokens` 每个有 draft 的步新增一次 host 侧 `broadcast_object`(~6.7-7.9 us),`reallocate_drafts` 在裁剪步新增一次 c10d `broadcast`(~57-59 us,最坏约占 5-15 ms decode 步的 0.4-1.4%);e2e(AV 走 PyNCCL 的早期版本)无可见回归。覆盖 TP 及 PCP=1 的 DCP;PCP>1 时各 PCP 切片的 TP 组各有 leader,DeepSeek-V4.1 on NVIDIA 自 #61994 起已拒绝 AV+PCP,但通用 `PCPManager` 路径仍放行,作者注明为遗留 follow-up。transport 用 c10d;改走 PyNCCL 的优化涉及所有 `GroupCoordinator.broadcast` 调用方且需 XPU 验证,留待独立 PR。

## 关联
- [#56831](https://github.com/vllm-project/vllm/pull/56831)
- [#54011](https://github.com/vllm-project/vllm/pull/54011)
- [#61994](https://github.com/vllm-project/vllm/pull/61994)
- [#58466](https://github.com/vllm-project/vllm/pull/58466)
- [#63421](https://github.com/vllm-project/vllm/pull/63421)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
