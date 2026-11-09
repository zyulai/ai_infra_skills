# PR #36518: [Kernel] Fuse FP8 output quantization into merge_attn_states

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @carlyou |
| 合并 | 2026-04-03 |
| 分类 | Quantization |
| Label | documentation, performance, rocm, ready, ci/build, v1, llama, qwen, nvidia |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 10 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/36518 |

## 一句话结论

给 `merge_attn_states`(CUDA + Triton)加可选 `output_scale`,把 FP8 静态 per-tensor 量化融进最终 store,省一次量化 kernel 启动与 BF16 中间往返,H100 microbenchmark 均值约 1.6x。

## 动机

split-KV/混合 prefill 场景下,merge_attn_states 把 prefix/suffix 部分注意力结果按 LSE 合并。当下游需要 FP8 输出时,原先流程是:merge 写出 BF16 中间张量,再由独立量化 kernel 读回并量化写 FP8,多一次 kernel launch 和一次完整 BF16 张量的显存往返(修复 issue #33097)。

## 优化点

`merge_attn_states` 全链路新增可选 `output_scale`:CUDA kernel 模板化 `output_t` 与 `USE_FP8_OUTPUT`,输入仍按标量类型 128-bit pack 加载,FP8 输出时合并值在 float32 计算后经 `scaled_fp8_conversion` 转换,以更小的 pack(64-bit/32-bit)写出;scale 在 kernel 内先取倒数(`1.0f / *output_scale`)用乘法。三条写路径(suffix 直接拷贝、-inf LSE 的 prefix 输出兜底、正常 merge)都接入 FP8 转换。Triton kernel 加 `USE_FP8` constexpr,store 前 `clamp` 到 FP8 范围再 cast,scale 取倒数预乘。dispatcher(`vllm/v1/attention/ops/merge_attn_states.py`)改为按 `prefix_output` 的 dtype/headdim 判断 CUDA 可用性,输出为 FP8 时要求必须传 `output_scale`,否则维持原 dtype 一致性校验;不传 `output_scale` 的调用方行为完全不变。op schema 与 Python 绑定同步加 `Tensor? output_scale=None`。

## 关键改动

```text
csrc/attention/merge_attn_states.cu: kernel 模板化 output_t/USE_FP8_OUTPUT,三条路径融合 FP8 store
vllm/v1/attention/ops/triton_merge_attn_states.py: Triton kernel USE_FP8 分支(clamp + cast)
vllm/v1/attention/ops/merge_attn_states.py: dispatcher 按输入 dtype 判定,FP8 输出需 output_scale
tests/kernels/attention/test_merge_attn_states.py: use_fp8 参数化,反量化空间对比(atol/rtol=1e-1)
benchmarks/fused_kernels/merge_attn_states_benchmarks.py: 新增 fused/unfused CUDA/Triton 四路对比 benchmark
```

## 性能声明

H100 SXM 80GB microbenchmark(`merge_attn_states_benchmarks.py`,tokens 1-4096、heads 4-128,覆盖 DeepSeek-V3 MLA/Llama-70B/Llama-8B 头配置、TP 1-8、fp32/fp16/bf16 输入):fused CUDA vs unfused CUDA 提速 1.41x-2.24x(overall mean 1.65x);fused Triton vs unfused Triton 提速 1.17x-2.05x(mean 1.58x),大 token 数区间 Triton 增益收窄(1.17x-1.36x)。smoke test 在 H100/B200 上编译并通过 serve。

## 验证与回归风险

- 测试:`test_merge_attn_states.py` 972 个用例全过(165.46s),新增 `use_fp8` 参数化(输入保持 bf16/fp16/fp32,输出 FP8),对比在乘回 scale 的反量化空间进行(atol=rtol=1e-1);torch 参考实现同步加 output_scale 分支。
- 风险:FP8 路径要求 output dtype 为 FP8 且必须带 scale(TORCH_CHECK/assert);dispatcher 判定依据从 output 改为 prefix_output,若上游传入 FP8 输入仍走 Triton fallback;Triton 的 clamp 使用 e4m3 min/max 常量,ROCm(fnuz)下范围不同但 `float8_info` 由 `current_platform.fp8_dtype()` 决定;未传 output_scale 的既有调用不受影响。

## 关联

- 同区域:#16173(merge_attn_states CUDA kernel 引入)、#32950(MLA decode cat+quant 融合,同为 QuantFP8/fp8 输出融合线)、#36518 修复的 issue #33097。
- 依赖:`QuantFP8`(static/per-tensor)、`scaled_fp8_conversion` CUDA 工具。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
