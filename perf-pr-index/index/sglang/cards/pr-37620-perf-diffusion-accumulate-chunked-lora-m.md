# PR #37620: [Perf] diffusion: Accumulate chunked LoRA merges with addmm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Levius-Fubuki |
| 合并 | 2026-10-22 |
| 分类 | Attention / Kernel (tier: major) |
| Label | lora, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/37620 |

## 一句话结论

2D diffusion LoRA 合并用 `addmm_` 把每个 chunk 的 B@A 直接累加进基础权重,30 例微基准几何平均 1.695x,临时分配降为 0。

## 动机

2D diffusion LoRA merge 路径对每个 chunk 先物化 `B @ A`,再单独启动一次 in-place add。在配置的 32 MiB chunk 上限下,这会产生一大块临时分配和一次额外的 kernel launch。改用标准 PyTorch 矩阵原语即可同时消除两者。

## 优化点

`_merge_lora_into_data` 的分块循环从 "matmul 出 chunk_delta 再 `add_(alpha=scale)`" 改为 `data_2d[start:end].addmm_(lora_B_2d[start:end], lora_A_sliced, alpha=scale)`,把乘加融合为一次 `addmm_` 直接累加进基础权重。chunk 大小与 scaling 策略保持不变,包括多 adapter 的有序累加;不改变 LoRA 加载、merge 精度策略、stacked/大于 2D 的回退行为。使用无 device/compute-capability 分支的标准原语,与 #20562 的 torch-native LoRA 优化一致。作用域仅在 adapter 加载/合并路径(`@torch.no_grad()` 装饰),不在编译的模型 forward 或 serving 请求路径上。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/lora/linear.py: 分块合并循环改用 Tensor.addmm_
python/sglang/multimodal_gen/test/unit/manual/bench_lora_merge_addmm.py: 配对 CUDA 微基准(含数值 parity 校验)
python/sglang/multimodal_gen/test/unit/test_lora_inference_mode.py: 新增分块 addmm 累加回归测试
```

```python
data_2d[start:end].addmm_(
    lora_B_2d[start:end], lora_A_sliced, alpha=scale
)
```

## 性能声明

RTX 4090 (SM89),driver 595.71.05,CUDA 13.0,PyTorch 2.13.0+cu130;30 个 dtype/shape/rank 用例、7 轮交替顺序、50 warmup/500 repetitions,报告配对中位数。全量 30 例:几何平均加速 **1.695x**,最小 1.217x,最大 2.653x,无回退用例(0/30),候选路径额外峰值分配每例 0 字节;最大新旧差 BF16 3.05e-5、FP16 3.81e-6、FP32 9.31e-10。代表性用例:BF16 4096x4096 rank32:42.408 → 21.207 µs(2.000x,分配 32 → 0 MiB);BF16 12288x4096 rank64:428.542 → 225.182 µs(1.903x);FP32 12288x4096 rank4:1284.776 → 484.248 µs(2.653x);FP32 1280x1280 rank8:29.098 → 23.919 µs(1.217x)。profiler 显示热路径算子从 `matmul/mm + add_` 变为 `addmm_`。作者明确:声明仅限 merge 延迟与临时分配,不含端到端生成吞吐。

## 验证与回归风险

4 个 LoRA 回归测试文件跑通(28 passed);新增测试覆盖 FP32/FP16/BF16、连续与转置非连续张量、负 strength、`alpha/rank` scaling、强制多 chunk,并用 `TorchFunctionMode` 断言 `addmm_` 调用次数(6)与基础权重 storage 指针不变;独立微基准在测延迟前先做数值 parity 校验。black/isort/Ruff/git diff --check 通过。风险面小:改动限于 diffusion LoRA merge 路径的一处循环;数值与旧路径在容差内一致但非逐位一致(BF16 最大差 3.05e-5);CI Extra 与 AMD ROCm 10 未通过。

## 关联

与 #20562(torch-native LoRA 优化)风格一致;正文未列其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
