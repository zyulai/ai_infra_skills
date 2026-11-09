# PR #57053: [Bugfix][MRV2][Spec Decode] Honor dynamic K in autoregressive speculators, 1.2~1.3x kernel perf improvement

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57053](https://github.com/vllm-project/vllm/pull/57053) |
| 作者 | yewentao256 |
| 合并日期 | 2026-10-05 |
| labels | bug, speculative-decoding, ready, nvidia, mrv2, dflash |
| 热度 | 👍 1 · 💬 30 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
MRV2 未把调度器选的动态 K 传给自回归 speculator,导致按最大 K 多跑 draft forward 且残留陈旧 draft token;打通传递链后 benchmark 提速 1.2~1.3x。

## 问题现象
动态投机解码(dynamic spec decoding)下,调度器已选定本轮 K(如 3),但自回归 speculator(MTP/draft model 等)仍按最大 K=5 执行 draft forward——多余的 draft 步是纯浪费(正文基准脚本显示 forwards 5→3,固定 batch 下 0.149ms→0.115ms);同时产生 stale draft tokens,`propose()` 返回的 draft_tokens 中超出本轮 K 的槽位未被清掉。正文未描述用户可见的报错现象,问题以性能浪费与正确性隐患形式存在。

## 根因
正文披露:MRV2 没有把 `SchedulerOutput.num_spec_tokens_to_schedule` 传递到自回归 speculator,`propose()` 内部各处循环与早退判断使用 `self.num_speculative_steps`(配置的最大 K),而不是调度器动态选出的本轮 K。

## 修复方案
- `vllm/v1/worker/gpu/model_runner.py`:`ExecuteModelState` 增加 `num_spec_tokens_to_schedule` 字段,`execute_model()` 从 `scheduler_output` 存入,`sample_tokens()` 传给 `propose(num_speculative_tokens=num_spec_tokens)`;`num_spec_tokens < self.num_speculative_steps` 时把 `draft_tokens[:, num_spec_tokens:]` 填 -1,且 `set_draft_tokens` 只传切片后的 `[:num_spec_tokens]`。
- `spec_decode/autoregressive/speculator.py`:`propose()` 接受 `num_speculative_tokens: int | None`,为空回落 `self.num_speculative_steps`;`_multi_step_decode`/`_fused_multi_step_decode`/`_generate_draft` 等循环上界、`draft_max_seq_len`、早退条件(K<=1,K==0 填 -1)全部改用运行时 K。
- `spec_decode/autoregressive/cudagraph_utils.py`:新增 `SpeculatorBatchDescriptor`(带 `num_speculative_tokens`)与 `specialize_spec_tokens()`,动态 K 场景按 schedule 对不同 K 捕获/复用对应 CUDA graph,非命中则回退 eager。
- 各 speculator 子类(dflash、draft_model、multi_module_mtp、extract_hidden_states)与基类接口同步加参数;`warmup.py` 的 SchedulerOutput 构造补上 `num_spec_tokens_to_schedule`。

## 影响与回归风险
- 回归面:所有自回归 speculator(MTP、draft model、dflash、multi_module_mtp、extract_hidden_states)的 propose 链路、draft token 缓冲区写回与 CUDA graph 捕获/重放选择;`cudagraph_utils._init_candidates` 新增 `decode_query_len > num_speculative_tokens` 条件。非动态 K 场景运行时 K 等于最大 K,行为不变;动态 K 场景 graph 需按 K 特化,未命中时走 eager(性能而非正确性回退)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
