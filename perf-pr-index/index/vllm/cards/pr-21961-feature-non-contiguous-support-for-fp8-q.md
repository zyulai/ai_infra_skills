# PR #21961: [Feature] Non-contiguous Support for FP8 Quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @yewentao256 |
| 合并 | 2025-08-05 |
| 分类 | Quantization(major) |
| Label | performance, ready, deepseek |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 与文件数);diff 内容可见 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/21961 |

## 一句话结论

让 FP8 per-tensor/per-token 量化 CUDA kernel 支持 non-contiguous(最后一维连续)输入,免去量化前的整块 contiguous 拷贝。

## 动机

此前 `scaled_fp8_quant` 系列入口在 C++ 侧 `TORCH_CHECK(input.is_contiguous())` 强制要求整张量连续,Python 侧还先调 `input.contiguous()`:遇到带 row padding 或切片的输入会触发一次额外的整块拷贝(修复 issue #19630)。DeepSeek 等模型激活常带 padding,该约束让本可直接做的量化多付一趟内存带宽。

## 优化点

重写 `csrc/quantization/fp8/common.cu` 中三个量化 kernel 为 strided 版本:grid 保持 one block per token,行地址按 `in_row_stride`/`out_row_stride`(取 `stride(-2)`)计算,只要求最后一维 stride == 1。向量化路径改用公共的 `vectorize_with_alignment<16>` 工具,删除 `common.cuh` 里手写的 `thread_max_vec`/`scaled_fp8_conversion_vec`。动态 per-tensor 量化中 `segmented_max_reduction_strided` 每 block 归约一行 absmax,再以 `atomicMaxFloat` 累积到全局 scale;scale 缓冲由 `torch.zeros` 改为 `torch.empty` + kernel 启动前 `cudaMemsetAsync` 清零。Python 入口删除 `.contiguous()` 调用。

## 关键改动

```text
csrc/quantization/fp8/common.cu: 新增 scaled_fp8_quant_kernel_strided 等 strided kernel,按 row stride 寻址 + 统一 16B 向量化
csrc/quantization/fp8/common.cuh: 删除旧 segmented_max_reduction / thread_max_vec / scaled_fp8_conversion_vec
vllm/_custom_ops.py: scaled_fp8_quant 去掉 input.contiguous(),scale 由 zeros 改 empty
tests/quantization/test_fp8.py: 新增 non-contiguous(padded)输入的 static/dynamic/padding 量化用例
```

关键片段(strided kernel 按行 stride 寻址):

```cuda
const scalar_t* token_in = input + token_idx * in_row_stride;
fp8_type* token_out = out + token_idx * out_row_stride;
const float inv_scale = 1.0f / (*scale);
vectorize_with_alignment<16>(
    token_in, token_out, hidden_size, tid, blockDim.x,
    [=] __device__(fp8_type & dst, const scalar_t& src) {
      dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                  inv_scale);
    });
```

## 性能声明

DeepSeek-Coder-V2-Lite-Instruct-FP8,`vllm bench throughput --input-len 1000 --output-len 100 --enforce_eager`:本 PR 49.45 req/s、54306.45 total tok/s、4944.88 output tok/s,vs main 48.92 req/s、53730.95 total tok/s、4892.48 output tok/s(总吞吐约 +1%)。精度:gsm8k 5-shot flexible-extract 0.7574 / strict-match 0.7354,与 main 完全一致。硬件未注明。

## 验证与回归风险

`tests/quantization/test_fp8.py` 新增用例:m=975、n=512、padded_stride=576 的 non-contiguous 输入,覆盖 static、dynamic 与 `num_token_padding` 场景并与参考实现比对。回归面:所有调用 `scaled_fp8_quant` 的 FP8 路径(静态/动态 per-tensor/动态 per-token);新约束只要求最后一维连续,否则 TORCH_CHECK 报错,首维 stride 任意。

## 关联

修复 issue #19630。素材未列出同分类其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
