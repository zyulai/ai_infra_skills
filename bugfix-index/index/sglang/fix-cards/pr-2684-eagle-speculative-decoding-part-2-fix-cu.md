# PR #2684: Eagle speculative decoding part 2: Fix cuda graph + DP attention hanging

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2684](https://github.com/sgl-project/sglang/pull/2684) |
| 作者 | merrymercy |
| 合并日期 | 2024-12-31 |
| labels | (none) |
| 热度 | 👍 5 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
EAGLE 投机解码(承接 #2150)配套修复 cuda graph + DP attention 挂起:IDLE 纳入 cuda graph、replay token 数改为 bs*num_tokens_per_bs,并把 spec_info/spec_algorithm 贯通到 ForwardBatch。

## 问题现象
正文未描述具体现象,仅说明:这是 #2150 投机解码的配套改动,并修复 #2678 中 cuda graph + DP attention 的挂起(hanging)bug。

## 根因
正文未披露根因。diff 改动暗示的机制(由 diff 推断):(1) `ForwardMode.is_cuda_graph` 原来只含 DECODE/TARGET_VERIFY,不含 IDLE,DP attention 下 idle batch 不走 cuda graph 路径;(2) `cuda_graph_runner.replay` 用 `forward_batch.input_ids.numel()` 推 raw_num_token,在投机解码时与 `raw_bs * num_tokens_per_bs` 不符(原注释自己也写明了这一点),导致 padding/运行尺寸错误而挂起;(3) spec_info/spec_algorithm 原本没有从 ScheduleBatch 传入 ModelWorkerBatch/ForwardBatch,投机信息在 forward 侧缺失。

## 修复方案
- `forward_batch_info.py`:`is_cuda_graph` 加入 `ForwardMode.IDLE`;`ForwardBatch` 增加 `spec_info`/`spec_algorithm` 字段并在 `init_new` 从 batch 透传。
- `cuda_graph_runner.py`:`replay` 改为 `raw_num_token = raw_bs * self.num_tokens_per_bs`。
- `schedule_batch.py`:ScheduleBatch/ModelWorkerBatch 增加 `spec_info`/`spec_algorithm`,`init_new` 接收 `speculative_algorithm`,`merge_batch` 合并 spec_info,`get_model_worker_batch` 透传。
- `logits_processor.py`:`extend_return_logprob` 时若 `capture_hidden_mode.need_capture()` 也返回 last_logits(供 draft 使用)。
- `tp_worker.py`:`forward_batch_generation` 新增 `skip_sample` 参数,跳过 sample 直接返回 None。
- `server_args.py`:新增 `--speculative-algorithm/draft-model-path/num-steps/num-draft-tokens/eagle-topk` 参数(与 Double Sparsity 参数换位重排,功能不变)。

关键 diff:
```diff
-        # In normal decoding case, raw_bs == raw_num_token
-        # But in speculative decoding, raw_num_token is raw_bs * self.num_tokens_per_bs
-        raw_num_token = forward_batch.input_ids.numel()
+        raw_num_token = raw_bs * self.num_tokens_per_bs
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:cuda graph replay 的 token 数计算(普通 decode 下两者相等,行为不变;投机/DP 路径改变);IDLE 模式进入 cuda graph 影响所有 DP attention 用户;`extend_return_logprob + need_capture` 组合的返回值;CI workflow 中 MLA(TP=2)评估 timeout 由 20 分钟改为 10 分钟。

## 关联
- 配套 PR:#2150;修复挂起 bug:#2678;Co-authored-by: yukavio

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
