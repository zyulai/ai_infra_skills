# PR #34277: [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mattteochen |
| 合并 | 2026-08-17 |
| 分类 | Quantization(major) |
| Label | high priority, quant, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/34277 |

## 一句话结论

DSV4 量化内核直接产出 DeepGEMM TMA 对齐的 packed UE8M0 scale,免去每次 fp8_einsum 前的 scale 布局转换(BS1 每步省约 114 µs、61 次启动)。

## 动机

#27926 之后,DSV4 仍输出 FP32 activation scale,导致 DeepGEMM 在每次 `fp8_einsum` 前都要运行 `transpose_and_pack_fp32_into_ue8m0` 做布局转换:BS1 decode 每步 61 次内核启动、约 114 µs。

## 优化点

改动 DSV4 专用的 `fp8_wo_a_group_major_quant_ue8m0` 内核:lane 0 在算出 `scale_ue8m0` 指数后,直接按 4 个 UE8M0 字节打包进一个 int32 写出,物理布局为 `[G, ceil((D/128)/4), align_up(T, 4)]`(hidden 维打包、token 维 TMA 对齐),Python wrapper 返回 DeepGEMM 原生可消费的逻辑 `[T, G, ceil((D/128)/4)]` 视图,`fp8_einsum` 因此不再需要 scale-layout 转换内核。末尾不足 4 字节的 pack 由 kernel 显式清零,避免 allocator 垃圾变成 activation scale。量化输出码与 einsum 数值均不变(FP8 码 bit-exact、packed scale 与 `get_mn_major_tma_aligned_packed_ue8m0_tensor` 参考逐位相等、einsum 输出与 fp32 scale 路径 `torch.equal`)。host 侧 TensorMatcher 校验 packed 形状、TMA 对齐 token 数、stride 与 16B 指针对齐。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepseek_v4/fp8_wo_a_group_major_quant.cuh: scale 输出改 int32 packed UE8M0([G, packed_hidden, aligned_T] 布局 + 尾部清零)
python/sglang/kernels/ops/attention/dsv4/fp8_wo_a.py: wrapper 分配 packed int32 storage 并返回 DeepGEMM 原生 stride 视图
test/registered/kernels/ops/attention/test_fp8_wo_a.py: 断言改为 packed 布局 + 与 fp32 scale 路径 einsum 输出相等
```

## 性能声明

DeepSeek-V4-Pro,TP8,8×B200,ISL 16 / OSL 1024,1,023 个 decode 步的中位数:BS1 10.5033→10.3778 ms(省 125.5 µs,+1.21%);BS2 +1.01%;BS4 +0.97%;BS8 +0.46%;BS16 13.3792→13.0686 ms(+2.38%);BS32 16.0405→15.4845 ms(+3.59%);BS64 +0.50%。几何平均:decode 时延低 1.42% / 吞吐高 1.44%。

## 验证与回归风险

GSM8K 96.74%(1,319 examples,single-shot,sgl-eval,32 线程;stop/truncated/error 100.00%/0.00%/0.00%)。单测覆盖:FP8 码与 flat UE8M0 参考 bit-exact、packed scale 与 DeepGEMM 参考布局逐位相等、空 token 维、大 token 轴、DSV4 形状(T=1/G=2/D=256 新增)、`fp8_einsum` 消费契约(packed 与 fp32 scale 输出 `torch.equal`)。风险面:契约收窄——`output_s` 必须是 contiguous `[group, packed_hidden, aligned_token]` int32 且 16B 对齐,hidden 必须被 128 整除;任何绕过该 wrapper 直接构造 scale 的调用方需同步改布局;TMA 对齐的 token 维(向上取整到 4)带来少量 scale 存储冗余。

## 关联

#27926(前序:引入 DSV4 wo_a quant 与 UE8M0 路线,本 PR 消除其遗留的 FP32 scale 转换)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
