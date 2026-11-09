# PR #17228: [NPU] make torch_native lora backend a little bit faster

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @VDV1985 |
| 合并 | 2026-03-07 |
| 分类 | Other (tier: major) |
| Label | lora, run-ci |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/17228 |

## 一句话结论

NPU 上 LoRA torch_native backend 把 einsum 换成 mm+T 并去掉中间张量,Qwen3-14B + LoRA 整体执行快约 3%。

## 动机

LoRA 的 torch_native backend 在 NPU 上走 `torch.einsum`,einsum 在该硬件上的算子分解/调度开销高于直接 GEMM,且 b 侧投影每步多产生一个中间结果张量。作者目标是小幅提升该 backend 在 NPU 上的执行效率。

## 优化点

改动集中在 `python/sglang/srt/lora/torch_ops/lora_ops.py` 两个函数:(1) `sgemm_lora_a_fwd` 与 `sgemm_lora_b_fwd` 中 `torch.einsum("si, oi -> so", x, w)` 全部替换为 `torch.mm(x, w.T)`,用显式转置加标准 matmul 替代 einsum 的解析与降维;(2) `sgemm_lora_b_fwd` 原先"先算 result 再 `output[...] += result`"的两步写法合并为 `output[...].add_(torch.mm(...))`,减少一个中间张量分配。数学运算的种类与顺序不变(PR 明确说明 algorithm logic 与 math operation sequence 未变),属纯算子表达层面的改写。

## 关键改动

```text
python/sglang/srt/lora/torch_ops/lora_ops.py: sgemm_lora_a_fwd / sgemm_lora_b_fwd 的 einsum 改 mm+T,b 侧 inplace add_
```

```python
result = torch.mm(x_seq, w_seq.T)
...
output[
    token_offset : token_offset + seq_len,
    slice_start_output:slice_end_output,
].add_(torch.mm(x_slice, w_slice.T))
```

## 性能声明

- bench_serving 显示 Qwen3-14B(带 LoRA adapter)在 Atlas A2 与 Atlas A3 硬件上整体模型执行提升约 3%。
- PR 未注明 bench_serving 的具体请求分布、并发与输入/输出长度等条件。

## 验证与回归风险

验证:数学逻辑与运算顺序不变的论证 + 用 curl 请求 Qwen3-14B 带 LoRA adapter 的快速抽查,前后答案相同;无新增单元测试,checklist 各项合并时均未勾选。风险面:影响所有使用 torch_native LoRA backend 的路径(以 NPU 为主要目标,`lora_ops.py` 是共享的 torch_ops 实现,其他设备走此 backend 时同样被改写);`add_` 是就地写,要求 output 切片可写且无别名;`w.T` 为非连续视图,依赖底层 matmul 对转置视图的支持效率(NPU 上正是收益来源)。

## 关联

同区域:sglang LoRA 后端体系(torch_native 与其他硬件 backend);pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
