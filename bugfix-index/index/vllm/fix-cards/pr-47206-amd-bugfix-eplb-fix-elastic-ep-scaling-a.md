# PR #47206: [AMD][Bugfix][EPLB] Fix elastic EP scaling accuracy on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#47206](https://github.com/vllm-project/vllm/pull/47206) |
| 作者 | okorzh-amd |
| 合并日期 | 2026-07-24 |
| labels | bug, ready, rocm |
| 热度 | 👍 2 · 💬 9 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

ROCm dummy 权重初始化清零专家拓扑缓冲致 scale-up 输出全错,另有 EPLB 冗余重排拖垮精度;buffer 改非持久化并跳过无效重排。

## 问题现象

`distributed/test_elastic_ep.py` 在 ROCm 上初始与扩容后的 checkpoint 处 GSM8K 精度崩溃:Bug 1 为 scale-up 后输出全为垃圾(accuracy=0);Bug 2 为 sync-EPLB 精度损失。修复前各 checkpoint 精度 0.19 / 0.016,修复后全部 ≥ 0.58 阈值。四个 elastic EP scaling 测试(sync/async eplb、even/uneven)全部 PASSED(共 1194.63s)。

## 根因

正文披露两个根因:

- Bug 1:弹性 EP 的新 rank 以 `load_dummy_weights=True` 创建;ROCm 上 `initialize_single_dummy_weight()` 会对 `model.state_dict()` 中所有非浮点张量执行 `.zero_()`。FusedMoE 的专家拓扑缓冲 `_expert_map`、`expert_mask`、`expert_{global_to_physical,physical_to_global,local_to_global}` 被注册为持久化整型 buffer,刚按新 EP 拓扑正确构建完就被清零,导致 all2all dispatch 输出垃圾。CUDA 从不清零整型张量,故该 bug 仅出现在 ROCm。正文并纠正了早先的猜测:映射不是在 ep_size==1 时未分片构建,而是在目标 ep_size 构建正确后遭清零。
- Bug 2:`rearrange_expert_weights_inplace()` 在提议映射几乎未改善均衡(imbalance 1.0)时仍每个 step_interval 触发一次,约 16 秒的横向 all-to-all 往返挤占引擎、拉低精度。

## 修复方案

- `vllm/model_executor/layers/fused_moe/routed_experts.py`:`update_expert_map_info` 中把上述五个 buffer 全部改为 `register_buffer(..., persistent=False)`。它们是按 rank 确定性推导的 EP 拓扑元数据、非 checkpoint 状态,不应进 `state_dict()`(与 moondream3.py 的 `_expert_map` 注册方式一致);由此 ROCm dummy 初始化无法再清零它们,`batch_transfer_weights()`(遍历 state_dict)也无法用发送端的值覆盖它们,无需再做重排后元数据重建或按名过滤。
- `vllm/distributed/eplb/eplb_state.py`:`rearrange` 中新增跳过逻辑(按 diff,ROCm、非 profile、非 scale-up(rank_mapping is None)且映射全部 ≥0 时生效):对当前与提议映射分别计算每 rank 负载不均衡度,经 all-reduce 集体决策,相对改进 <5% 则跳过 `rearrange_expert_weights_inplace` 与 `_commit_eplb_maps` 并记日志。

## 影响与回归风险

- 影响窗口:正文未给出具体版本/提交范围;Bug 1 仅 ROCm(dummy 初始化清零整型张量的行为差异)。
- 回归面:五个专家拓扑 buffer 移出 `state_dict()`,任何依赖 state_dict 包含这些 buffer 的加载/保存(如 strict 加载、外部工具遍历 state_dict)会看到键集合变化;`batch_transfer_weights` 不再传输它们,依赖隐式传输这些张量的路径需依赖修复后"无需重建"的假设;跳过重排逻辑按 diff 限定 ROCm 且仅在改进 <5% 时触发,若不均衡度估算与实际负载偏差大,可能漏掉本有益的重排;CUDA 平台行为不受跳过逻辑影响。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
