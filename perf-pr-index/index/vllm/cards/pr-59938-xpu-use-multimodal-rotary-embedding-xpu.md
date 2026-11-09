# PR #59938: [XPU] use multimodal_rotary_embedding xpu kernel for mrope

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yma11 |
| 合并 | 2026-10-29 |
| 分类 | Attention / Kernel(major) |
| Label | intel-gpu |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/59938 |

## 一句话结论

XPU 上 mrope 改用精调的 `multimodal_rotary_embedding` XPU kernel 替代 triton 路径,Qwen3-VL-32B TP=4 多模态 serving 吞吐约 +13%。

## 动机

XPU 的 mrope 此前走 `triton_mrope` 路径(positions.ndim==2 时),与 CUDA 实现的分派行为分叉,需要单独维护;且 triton kernel 及其周边 launch 开销在多模态 decode 图中占比可观。依赖 vllm-xpu-kernels PR #616 提供的精调 `multimodal_rotary_embedding` kernel。

## 优化点

`mrope.py` 重构 `forward_cuda`:把 1D positions(纯文本)的 RoPE 逻辑抽成 `_forward_text_rope`,2D positions(多模态)继续走 `triton_mrope`。`forward_xpu` 增加对应分支:2D positions 时调用 `torch.ops._xpu_C.multimodal_rotary_embedding`(传入 head_size、cos_sin_cache、is_neox_style、mrope_section、mrope_interleaved,in-place 更新 query/key),1D 时复用 `_forward_text_rope`,使 XPU 与 CUDA 的分派结构一致。需注意 diff 呈现中该新分支代码位于既有 `return self.forward_cuda(...)` 之后(不可达位置),实际意图是以新 kernel 替换 triton 路径。PR 说明收益不止来自 kernel 本身,还包括减少 kernel launch 次数与 all-reduce trace 变化等。

## 关键改动

```text
vllm/model_executor/layers/rotary_embedding/mrope.py: 抽出 _forward_text_rope;forward_cuda 保持 CUDA 行为;forward_xpu 接入 multimodal_rotary_embedding xpu kernel
vllm-xpu-kernels PR #616: 提供精调后的 multimodal_rotary_embedding kernel(外部依赖)
```

XPU 分支节选:

```python
torch.ops._xpu_C.multimodal_rotary_embedding(
    positions, query, key, self.head_size, cos_sin_cache,
    self.is_neox_style, self.mrope_section, self.mrope_interleaved)
return query, key
```

## 性能声明

Qwen/Qwen3-VL-32B-Instruct,XPU TP=4,enforce-eager,gpu-memory-util 0.92,max-num-batched-tokens 8192,max-model-len 6000,每请求 1 张 512×512 图;bench:random-input-len 4096、random-output-len 1024、num-prompts 45、max-concurrency 15、request-rate inf、dataset random-mm:

- 输出吞吐:219.64 → 248.17 tok/s(约 +13%)
- Total token 吞吐:1155.27 → 1305.33 tok/s
- Mean TPOT:59.29 → 51.27 ms;Mean ITL:59.23 → 51.22 ms
- TTFT 基本持平(Mean 9221.92 → 9393.63 ms)

总 token 数完全一致(确定性负载)。GPU 具体型号 PR 未注明。

## 验证与回归风险

PR 的 Validation/Details 章节为空,未列单元测试;性能对比为同一命令两轮 serving benchmark。风险面:改动共享的 `mrope.py`,CUDA/ROCm 路径行为由 `_forward_text_rope` 重构保持(逻辑等价搬移);XPU 2D mrope 数值一致性依赖 vllm-xpu-kernels #616;checkpoint checklist 多项未勾选。

## 关联

依赖:vllm-xpu-kernels PR #616。同区域为 XPU 算子替换 CUDA 对齐系列;其他编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
