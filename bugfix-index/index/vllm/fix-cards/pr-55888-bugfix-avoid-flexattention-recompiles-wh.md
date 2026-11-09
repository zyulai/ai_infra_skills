# PR #55888: [Bugfix] Avoid FlexAttention recompiles when request counts change

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#55888](https://github.com/vllm-project/vllm/pull/55888) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-09-09 |
| labels | bug |
| 热度 | 👍 0 · 💬 2 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | hang |

## 一句话结论
请求数变化导致捕获张量形状变化,Dynamo 16 变体上限耗尽;改用满容量持久缓冲避免重编译。

## 问题现象
AMD CI 的 E2E Scheduling job(build 12711)失败:随着被捕获的请求维度随请求增减而变化,FlexAttention 耗尽 Dynamo 的 16-variant 编译变体上限。正文说明:nightly 12674 通过但两端源码完全一致、两个原始 workload 本地均可通过,因此未能定位引入性 PR。正文未给出用户侧报错日志原文。

## 根因
正文披露:FlexAttention 捕获的请求张量(request tensors)形状随请求完成/增减而变化,每次变化触发 Dynamo 重新编译,最终耗尽其 16 个编译变体的上限。修复思路(正文):让这些张量保持已分配容量、不随请求数改变形状。

## 修复方案
`vllm/v1/attention/backends/flex_attention.py`:
- 新增持久缓冲 `persistent_query_start_loc`(max_num_seqs+1)与 `persistent_seq_lens`(max_num_seqs),每步把活跃行拷入(`copy_to_persistent`)。
- `build()` 中传给 metadata 的 `query_start_loc`、`seq_lens`、`physical_to_logical`、`decode_offset` 全部改为持久的满容量张量,注释说明:mask 闭包只索引活跃请求,保留满容量形状可避免 batch 缩小到单请求时重编译。
- `copy_to_persistent` 调用从返回值风格改为就地更新(`rswa_prefix_lens` 同步改为持久张量引用)。

新增测试 `tests/kernels/test_flex_attention.py::test_flex_attention_request_count_changes_reuse_compiled_graph`:请求序列 (4,2,1,8,1) 下数值对齐 dense 参考,并断言 `counter.frame_count == 1`(仅编译一次),另验证 CUDA graph replay 下结果正确。

## 影响与回归风险
- 影响窗口:正文未说明(未能定位引入 PR)。
- 回归面:FlexAttention backend 的 metadata 构建路径;张量改为满容量持久缓冲后,依赖张量实际长度做语义判断的代码需确认仍按活跃行数处理;新增测试覆盖单请求 batch 与 CUDA graph replay。

## 关联
- [E2E Scheduling job build 12711](https://buildkite.com/vllm/amd-ci/builds/12711#01a08040-53ed-4d18-9de5-61feb19c4848)(正文提及的失败 CI)
- [nightly build 12674](https://buildkite.com/vllm/amd-ci/builds/12674#01a07b19-ca8d-499f-a279-e15993ff6da9)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
