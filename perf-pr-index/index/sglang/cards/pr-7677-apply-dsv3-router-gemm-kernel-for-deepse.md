# PR #7677: Apply dsv3 router gemm kernel for deepseek-r1 fp4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Fridge003 |
| 合并 | 2025-07-02 |
| 分类 | Quantization (tier: major) |
| Label | high priority |
| Diff | pack 未提供 diffstat（diff 内容可用：1 个文件） |
| 链接 | https://github.com/sgl-project/sglang/pull/7677 |

## 一句话结论

DeepSeek-R1 FP4 的 router GEMM（M<4、hidden 7168、256 experts、SM>=90）改用 dsv3_router_gemm 专用 kernel，bs=1 吞吐约 +1%。

## 动机

#7627 已为 DeepSeek-V3 加入 `dsv3_router_gemm` kernel，但未应用到 DeepSeek-R1 FP4 模型。此前该模型走通用 `F.linear`；而且 router logits dtype 需要额外转回 bf16，否则会触发 CUDA illegal memory bug——router 这条小 M GEMM 路径存在专用 kernel 可用而未用的浪费。

## 优化点

在 `deepseek_v2.py` 的 router forward 里加形状门控分支：当 `hidden_states.shape[0] < 4` 且 hidden=7168、weight 输出维=256、`_device_sm >= 90` 时，调用 `dsv3_router_gemm(hidden_states, self.weight)` 并转回输入 dtype；其余情况保持 `F.linear`。即只在 decode 单请求/极小 batch、恰好是 DSV3/R1 router 形状时启用专用 kernel，避免大 batch 下额外 dtype 转换的负收益（作者明说 bs=4/16 会因额外 dtype 转换掉速）。附带把同文件的 `get_device_sm()` 调用收敛到模块级 `_device_sm` 变量。

## 关键改动

```python
python/sglang/srt/models/deepseek_v2.py: router forward 加 dsv3_router_gemm 分支（M<4, hidden 7168, 256 experts, SM>=90）
```

关键条件（摘录）：
```python
if (hidden_states.shape[0] < 4
    and hidden_states.shape[1] == 7168
    and self.weight.shape[0] == 256
    and _device_sm >= 90):
    logits = dsv3_router_gemm(hidden_states, self.weight).to(hidden_states.dtype)
```

## 性能声明

DeepSeek-R1-FP4（modelopt_fp4，TP=8，flashinfer moe），bench_serving random 1000/1000、num-prompts 5、max-concurrency 1：output_throughput 137.522 -> 138.614 tok/s（约 +1%）；mean_tpot 7.038 -> 6.951 ms。TTFT 反而变差（mean 236.763 -> 263.883 ms；median 152.591 -> 177.071 ms），PR 未解释。GSM8K（8-shot，1319 questions）0.904。

## 验证与回归风险

GSM8K 0.904 作为准确度验证；无新增单元测试。风险面小：分支条件严格限定 M<4、hidden 7168、256 experts、SM>=90，其他模型/形状不受影响；kernel 输出需 `.to()` 转回输入 dtype（多一次转换，大 batch 反而更慢是已知取舍）；fp4 量化路径结合 flashinfer moe 的组合未在 CI 覆盖的迹象（PR checklist 未勾选）。

## 关联

前序 PR：#7627（加入 dsv3_router_gemm kernel，本 PR 将其应用到 deepseek-r1 fp4）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
