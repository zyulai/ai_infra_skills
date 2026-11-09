# PR #32963: [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#32963](https://github.com/sgl-project/sglang/pull/32963) |
| 作者 | wenscarl |
| 合并日期 | 2026-09-18 |
| labels | bug, deepseek, run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 35 |
| 分类 | MoE |
| bug 类型 | general |

## 一句话结论
hybrid EP+TP 下 allreduce fusion 跳过两个正交 post-experts 规约、融合内核只按 _MOE_EP 单组规约,输出无错但错误;合并为一次 _TP 规约修复。

## 问题现象
在 4×GB200 上以 `--tp-size 4 --ep-size 2` 启动(如 Qwen/Qwen3-30B-A3B),除非传 `--enforce-disable-flashinfer-allreduce-fusion`,补全结果为乱码(问法国首都答不出 "Paris");`--ep-size 4` 与 `--ep-size 1` 均正常。`nvidia/DeepSeek-V4-Flash-NVFP4` 上表现为 GSM8k 准确率仅 0.012,静默返回「规约不足」(under-reduced)的激活,无任何报错。复现于 `main` 分支。

## 根因
正文披露:`--tp-size 4 --ep-size 2` 时 post-experts 规约跨越两个不相交组(`moe_expert_parallel_all_reduce` 走 `_MOE_EP`,再 `moe_tensor_model_parallel_all_reduce` 走 `_MOE_TP`),二者是 TP 世界的正交分解,4 个 rank 需要两次规约都执行。当 decoder 发布 `fuse_mlp_allreduce` 后,`should_skip_post_experts_all_reduce()` 基于 `should_skip_mlp_all_reduce()` 不区分 EP/TP 路径,**两个**规约都被跳过;本应由下一层融合 residual+RMSNorm 吸收,但该内核只对单组规约——MoE fusion workspace 在 `moe_ep_size > 1` 时 rendezvous 在 `_MOE_EP`——净效果是 MoE 输出只在 4 个必需 rank 中的 2 个上规约,结果错误且无错误信息。`ep_size=1`/`ep_size=4` 下单组即覆盖全部 peer 故不受影响。该问题 main 上早已存在,与在途 FlashInfer 工作无关(#30700 评审中暴露)。

## 修复方案
- `layers/moe/utils.py` 新增合并原语:`can_merge_post_experts_all_reduce()`(`moe_ep_size>1 且 moe_tp_size>1 且 moe_dp_size==1` 时,两组正交分解等价于一次 _TP 规约)、`post_experts_all_reduce()`(可合并时发一次 `tensor_model_parallel_all_reduce`,否则保持两步)、`deferred_post_experts_all_reduce()`(融合内核无法处理该 shape 时的内联兜底)。
- 各模型(deepseek_v2、hunyuan_v3、qwen3_moe、qwen2_moe)的 post-experts 规约统一改调 `post_experts_all_reduce()`;`flashinfer_comm_fusion` 用 `resolve_fusion_group()/resolve_fusion_world_size()` 保证 workspace 建在与融合内核一致的组上(可合并时落在 _TP,ws=4);`LayerCommunicator` 仅在合并不可用(`moe_dp_size>1`)时关闭融合。正文 Modifications 段描述的是「hybrid EP+TP 直接返回 False 不融合」的方案,合并后的 diff 实现为上述「合并成一次 _TP 规约、可合并时允许融合」的最终形态。
- 新增 CPU 单测 `test_layer_communicator_fusion_gate.py`;GSM8k 对比:`--tp-size 4 --ep-size 2` 修复前 0.012 → 修复后 0.777(参考 tp4 ep4 为 0.743);DSV4-Flash-NVFP4 GSM8k 0.954;单测 3 passed。

## 影响与回归风险
- 影响窗口:正文未说明(称该 bug 在 main 上 pre-existing)。
- 回归面:所有 MoE 模型的 post-experts all-reduce 调用点(deepseek_v2/hunyuan_v3/qwen2_moe/qwen3_moe)、LayerCommunicator 融合门控、flashinfer workspace 组解析。纯 TP/纯 EP 行为不变;hybrid 且 `moe_dp_size==1` 时两次规约合并为一次(减少集合通信);`moe_dp_size>1` 保持两步且不融合(测试注明该拓扑当前无法启动,属护栏)。

## 关联
- [#30700](https://github.com/sgl-project/sglang/pull/30700)(评审中发现该问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
