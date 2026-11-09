# PR #39064: [Bugfix] Fix GDN FLA kernel crashes with NULL_BLOCK_ID=0 CUDA graph padding

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#39064](https://github.com/vllm-project/vllm/pull/39064) |
| 作者 | vibhavagarwal5 |
| 合并日期 | 2026-04-11 |
| labels | bug, nvidia, ready, v1, verified |
| 热度 | 👍 2 · 💬 15 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | crash |

## 一句话结论
CUDA graph 填充从 -1 改为 0 后 FLA 内核 `<0` 守卫失效,padding 读写 ssm_state[0] 致 IMA;守卫改 `<=0` 修复。

## 问题现象
GDN(Gated Delta Network)混合架构模型在 Blackwell GPU 上、CUDA graphs + TP>1 场景下崩溃,报 IMA(Illegal Memory Access)。作者在 4x Blackwell RTX PRO 6000、TP=2、Qwen/Qwen3.5-35B-A3B 上复现;`--enforce-eager`(关闭 CUDA graph)可规避,确认 CUDA graph padding 是触发条件。单独修 `fused_sigmoid_gating.py` 时 100 并发通过但 200 并发仍崩;两个文件都修后 10,000 请求(50 轮 x 200 并发,混合提示,约 2.8M tokens)零失败。

## 根因
正文披露:commit `bcc6f6744`(3 月 30 日)把 CUDA graph block table 的填充值从 `fill_(-1)`(PAD_SLOT_ID)改成了 `fill_(NULL_BLOCK_ID=0)`。FLA SSM 内核用 `state_idx < 0` 判断并跳过 padding 项,能挡住 -1 但**挡不住 0**;于是 CUDA graph 的 padding 项以 state index 0 进入内核,直接读写 `ssm_state[0]`,破坏真实序列 0 的状态,最终引发 IMA。即 padding 约定变了而内核守卫没同步。正文同时说明 `causal_conv1d.py` 和 `mamba_ssm.py` 用的是 `== null_block_id` 比较,不受影响。

## 修复方案
把两个 FLA 内核文件中共 5 处守卫比较改为 `<= 0` / `> 0`(`fused_sigmoid_gating.py` 初始 state 加载、final state 存储;`fused_recurrent.py` 多 token 初始 state 加载、final state 存储、packed decode 内核):

```diff
-            # Skip if state index is invalid (PAD_SLOT_ID = -1)
-            if state_idx < 0:
+            # Skip if state index is invalid (NULL_BLOCK_ID=0)
+            if state_idx <= 0:
                 return
```

final state 存储侧对应由 `final_state_idx >= 0` 改为 `final_state_idx > 0`,只对有效索引写回状态。`pytest tests/models/test_gdn.py -v` 通过。

## 影响与回归风险
- 影响窗口:自 commit `bcc6f6744`(2026-03-30)改填充值为 NULL_BLOCK_ID=0 起,至本修复前;GDN 模型 + CUDA graphs 组合受影响。
- 回归面:仅 Triton FLA 内核的索引有效性判断,状态索引 0 从"被当作 padding 误写"变为"被跳过"。注意语义变化:若真有合法序列使用 state index 0,其初始 state 加载与 final state 存储现在会被跳过——按正文,padding 约定即 NULL_BLOCK_ID=0,index 0 不再是合法槽位,故该假设成立;但这意味着依赖 index 0 为有效槽位的路径需留意。

## 关联
- [#39025](https://github.com/vllm-project/vllm/issues/39025)(本 PR 修复的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
