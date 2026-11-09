# PR #2542: Fused MOE for Mixtral

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pcmoritz |
| 合并 | 2024-01-30 |
| 分类 | MoE (tier: major) |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/2542 |

## 一句话结论

为 Mixtral 融合 MoE kernel，H100 上 1000 in/50 out 从 28600 提升到 33600 tok/s。

## 动机

本 PR 建立在 #2453 与 #2293 之上，为 Mixtral 模型融合 MOE kernel。正文给出作者本机观察：融合后吞吐从 28600 提升到 33600 tok/s（1000 input tokens、50 output tokens、H100）。正文未描述更多此前瓶颈的机制细节，也未给出精度或 workload 的额外说明。

## 优化点

把 `MixtralMLP`（每个 expert 用 `ReplicatedLinear` 的 w1/w2/w3 加 `nn.SiLU` 逐 expert 计算）替换为 `MixtralMoE`：tensor-parallel MoE 实现，把每个 expert 的权重分片到所有 rank，forward 用 fused MoE kernel，最后跨 rank reduce 输出。新增 `top_k` 参数并引入 `fused_moe` 与 `set_weight_attrs`；权重加载改用带 `weight_loader` 的分片路径。CUDA 侧修正 `moe_align_block_size` 的 kernel 名拼写（`moe_alig_block_size_kernel` → `moe_align_block_size_kernel`），在 `csrc/ops.h` 中把 `moe_align_block_size` 声明移到 `#endif` 之前，`csrc/pybind.cpp` 调整对应绑定缩进。正文未描述 fused kernel 内部算法细节。

## 关键改动

```text
vllm/model_executor/models/mixtral.py: MixtralMLP → MixtralMoE，权重按 TP 分片 + fused_moe forward + 跨 rank reduce，带 weight_loader 的加载路径
csrc/moe_align_block_size_kernels.cu: 修正 moe_align_block_size_kernel 名称拼写
csrc/ops.h: 把 moe_align_block_size 声明移到 #endif 之前
csrc/pybind.cpp: moe_align_block_size 绑定与描述
```

关键代码（来自 `vllm/model_executor/models/mixtral.py` diff）：

```python
class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """
```

## 性能声明

作者本机："from 28600 to 33600 tok / s with 1000 input tokens and 50 output tokens on H100"（未给出精度或并发数）。Latency 条件：`python benchmarks/benchmark_latency.py --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --input-len 1000 --output-len 50 -tp 8 --num-iters 100 --batch-size <bs>`。This PR：bs=1 `0.459s`、bs=2 `0.515s`、bs=4 `0.610s`、bs=8 `0.813s`、bs=16 `1.044s`、bs=32 `1.489s`、bs=64 `2.419s`；master：bs=1 `0.590s`、bs=2 `0.631s`、bs=4 `0.709s`、bs=8 `0.838s`、bs=16 `1.086s`、bs=32 `1.615s`、bs=64 `2.727s`。

## 验证与回归风险

PR 正文未注明测试命令或回归测试，仅给出上述吞吐与 latency 对比；正文未说明验证套件。diff 涉及的路径：Mixtral 模型 forward 与权重加载、`moe_align_block_size` 的 CUDA/pybind/ops.h 接口；正文未注明对 ROCm 或其他模型的影响。

## 关联

#2453 与 #2293（正文明确说明本 PR 建立在这两个 PR 之上）。无其他明确 PR 编号线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
