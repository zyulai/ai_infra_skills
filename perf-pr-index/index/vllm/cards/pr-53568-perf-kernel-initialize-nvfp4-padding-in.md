# PR #53568: [Perf][Kernel] Initialize NVFP4 padding in quant kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-09-01 |
| 分类 | Quantization (tier: major) |
| Label | performance, ready, nvidia, quantization |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/53568 |

## 一句话结论

把 NVFP4 量化 scale 的 padding 初始化移进 CUDA quant kernel,让 wrapper 可用 `torch.empty`,省掉一次独立的 zero-fill launch,小 M 场景最高提速 1.64x。

## 动机

#45739 为避免 padding scale 未初始化导致下游 NVFP4 GEMM 读到脏数据(曾引发 Blackwell NVFP4 decode 吞吐/输出长度严重回归),在 `create_fp4_scale_tensor` 里用单独的 `torch.zeros` 分配。这多出一次 zero-fill kernel launch,在 launch-bound 的小 batch 量化路径上成为额外开销。这是 #49775 与 #45739 的后续;与 #47587(FlashInfer TRT-LLM 8x4 布局)不重复,本 PR 针对 vLLM 内置 128x4 swizzled kernel。

## 优化点

机制:`cvt_fp16_to_fp4` kernel 新增 `PADDED_OUTPUT` 模板参数,在 kernel 内对无效行/列的 swizzled scale 项显式写 0(有效列外 `*sf_out = 0`),从而 scale buffer 可安全改用 `torch.empty`;host 侧按 `output_n == n` 选择模板实例——无 padding 时走保留的 fast path(#32520,输入输出宽度一致,直接用原地址算术并省掉 padded-output 判定),有 padding 时走 PADDED_OUTPUT 实例。PDL(programmatic dependent launch)支持保留。wrapper `_custom_ops.py` 中 `create_fp4_scale_tensor` 由 `torch.zeros` 改为 `torch.empty`。

## 关键改动

```text
csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu: PADDED_OUTPUT 模板参数,kernel 内清零 padding scale
vllm/_custom_ops.py: create_fp4_scale_tensor 从 torch.zeros 改为 torch.empty
tests/kernels/quantization/test_nvfp4_quant.py: 哨兵值回归测试(0xFF/0x7F7F7F7F 预填充)
```

## 性能声明

NVIDIA B300,CUDA 13.4 (sm_103a),Llama 3.3 70B(N=8192, K=28672),内置 128x4 swizzled quant kernel,before/after(µs):M=1 3.133→1.912 (1.639x);M=16 1.560x;M=32 1.531x;M=64 1.550x;M=128 1.324x;M=256 1.231x;M=512 1.170x;M=1024 1.099x;M=2048 0.993x;M=4096 0.990x;M=8192 0.993x。即 launch-bound 的中小 M 提升最高 1.64x,大 M(量化计算主导)基本持平;对照 FlashInfer CuTe-DSL 同场景数据一并列出(如 M=1: 2.418 µs)。

## 验证与回归风险

新增 `test_swizzled_scale_padding_initialized_in_kernel`:用哨兵值预填 output 与 scale buffer,断言 kernel 覆写所有 padding 行/列(无 memset 也无脏数据)。PR 报告 NVFP4 quantization suite 62 passed、Compute Sanitizer 2 passed(0 errors)。风险面:触及所有 NVFP4 量化路径的 scale 分配与写入约定;若未来有绕过该 kernel 直接写 scale buffer 的路径,padding 语义依赖 kernel 内清零这一契约;unpadded fast path 地址算术改变需与 #32520 语义保持一致。

## 关联

同区域:#49775、#45739(前置,引入 torch.zeros 方案)、#47587(FlashInfer TRT-LLM 8x4 布局,不重复)、#32520(fast path 来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
