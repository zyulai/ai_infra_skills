# PR #28164: [Feature] Enable TP + EP `shared_experts` overlap with router, 3.7% E2E performance improvement

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-11-06 |
| 分类 | Overlap / Scheduler（tier: major） |
| Label | ready |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 2 个文件：`vllm/model_executor/layers/fused_moe/shared_fused_moe.py`、`vllm/model_executor/layers/fused_moe/layer.py`（仅注释修正） |
| 链接 | https://github.com/vllm-project/vllm/pull/28164 |

## 一句话结论

在 TP+EP 部署下放开 shared_experts 与 router 专家计算的重叠执行，PR 声称小 batch 下 3.7% E2E、TTFT 24% 提升，并保证 shared expert 输出按需提前做 TP all-reduce。

## 动机

DeepSeek 类 MoE 的 shared_experts 计算与 routed experts 相互独立，可并行执行；此前 vLLM 的 SharedFusedMoE 只在 flashinfer + DP 场景启用该重叠，一旦启用 EP（`use_ep`）就直接禁用（注释称此场景"nothing to be gained"），导致 TP+EP 小 batch 配置失去这段可隐藏的计算。本 PR 是 #26440 的后续，目标是让 TP+EP 也能吃到 shared experts 重叠收益。

## 优化点

改动集中在 `SharedFusedMoE`：其一，移除 `use_overlapped` 判定中的 `self.use_ep` 禁用条件，使 EP 场景不再一刀切关闭重叠（flashinfer cutlass + DP>1 的禁用保留）；其二，把 TP reduce 判定从缓存的 `self.tp_size` 改为运行时 `get_tensor_model_parallel_world_size()`，避免 EP 组合下 TP world size 语义不一致；其三，在返回 `(shared_out, fused_out)` 前，若 `reduce_results` 且 TP>1 且 `must_reduce_shared_expert_outputs()`，先对 shared expert 输出做 `tensor_model_parallel_all_reduce`——即在重叠计算返回后尽早完成归约，保证后续与 routed experts 输出相加时语义正确（shared experts 层需以 `reduce_results=False` 创建）。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/shared_fused_moe.py: 放开 EP 下 shared experts 重叠；运行时查询 TP world size；重叠路径返回前提前 all-reduce shared_out
vllm/model_executor/layers/fused_moe/layer.py: 仅 docstring 注释修正（all_reduce 拼写）
```

```python
self.use_overlapped = (
    use_overlapped
    and not (self.use_flashinfer_cutlass_kernels and self.dp_size > 1)
    and self._shared_experts is not None
)
```

## 性能声明

PR 正文声称：TP+EP 下启用重叠获得 3.7% E2E 性能提升、TTFT 快 24%（配图，小 batch 场景）。测试环境：B200，`deepseek-ai/DeepSeek-R1 -tp 8 --enable-expert-parallel`，`vllm bench serve` random 数据集 input-len 4 / output-len 1024、160 prompts、并发 32。

注意：正文贴出的该配置基准表中 With overlap 与 main 相比为小幅回退——Output token throughput 719.24 vs 746.73 tok/s，Mean TTFT 168.90 vs 135.40 ms，Mean TPOT 39.90 vs 38.42 ms；PR 的 follow-up TODO 也承认"local node 下较大 batch 性能可能不佳，需 sweep batch size 找收益点"。精度：GSM8K flexible-extract 与 strict-match 均 0.9515。

## 验证与回归风险

测试：lm_eval GSM8K（1024 并发）精度对比 + `vllm bench serve` 吞吐/延迟基准；diff 未含新单测。风险面：影响所有 TP+EP 且 `use_overlapped` 开启的 MoE 模型（如 DeepSeek 系列）——shared experts 从被 torch.compile 隐藏的状态转为重叠执行，EP 下新增的提前 all-reduce 若 `must_reduce_shared_expert_outputs()` 判定与模型实现不匹配会破坏输出正确性；大 batch local node 场景 PR 自述可能负收益。

## 关联

前序：#26440（本 PR 为其 follow up）。同区域：EPLB/MoE 重叠相关优化（索引内另有多张 Overlap/EP 卡片）；follow-up TODO 提到 stream 复用优化与 batch size sweep，无编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
