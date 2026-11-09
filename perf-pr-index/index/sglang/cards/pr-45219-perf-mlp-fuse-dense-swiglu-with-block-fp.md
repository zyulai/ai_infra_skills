# PR #45219: [Perf] mlp: Fuse dense SwiGLU with block-FP8 quantization on decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JiaFengYu |
| 合并 | 2026-10-22 |
| 分类 | Quantization (tier: major) |
| Label | quant, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/45219 |

## 一句话结论

decode 时把 dense SwiGLU 与 block-FP8 activation quant 融合进一个 kernel,Qwen3-8B-FP8 单请求 decode TPOT -1.92%。

## 动机

dense block-FP8 MLP 的常规序列是:先 launch SwiGLU 把 BF16 activation 物化到 HBM,再 launch activation quantization 为 down-projection GEMM 准备 FP8 输入。每个 MLP 层多一次 kernel launch、多一次 BF16 中间结果的写/读。该开销在 decode(TPOT 敏感)路径上尤为扎眼,且与 KV-cache dtype 无关,属于纯 activation/MLP 融合收益。

## 优化点

在既有 JIT `per_token_group_quant` kernel 上新增显式 dense SwiGLU 模式(`kDenseSiluAndMul` 模板参数):与 dense CUDA activation 的编译器数学严格对齐——FP32 SiLU 后与 up 相乘,把乘积(而非 SiLU 中间值)round 回源精度再量化,全程不物化 BF16 中间结果;量化 scale 保持 standalone quantizer 的 `__fdividef` 快除法,dense 编译 flags 用 `get_activation_cuda_cflags()` 而非 `--use_fast_math`。DeepGEMM block-FP8 wrapper 扩展为可消费校验过的预量化 FP8 activation + column-major packed UE8M0 scales(`_deepgemm_packed_input_valid` 校验 TMA 对齐与布局),不兼容的 packed 布局直接 raise 而不是丢进无法消费的 Triton fallback。模型侧新增共享 helper `DenseSiluAndMulQuant`(Llama/Qwen2 MLP 接入,Qwen3 复用 Qwen2 实现),仅 decode 模式调用;prefill/extend、speculative verification、其他非 decode 模式一律直走原 activation。运行时 fast path 收得很窄:SM100 系 CUDA、TP1、普通 decode、BF16 activation、dynamic block-128 FP8 + DeepGEMM/UE8M0,并保留 LoRA、确定性执行、torch.compile、fused-op 覆盖等一整套 fallback 门。

## 关键改动

```text
python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh:
  kDenseSiluAndMul 模板分支——乘积 round 到源精度、__fdividef 量化 scale
python/sglang/kernels/ops/quantization/per_token_group_quant.py:
  silu_and_mul_mode="dense" 参数与 JIT 编译选择
python/sglang/srt/layers/quantization/fp8_activation.py(新):
  DenseSiluAndMulQuant helper(构造期门控 + forward_decode 逐项校验)
python/sglang/srt/layers/quantization/fp8_utils.py:
  DeepGEMM 接受 packed UE8M0 预量化输入,拒绝时 raise
python/sglang/srt/models/llama.py, qwen2.py:
  MLP forward 按 forward_mode.is_decode() 分派 helper
```

```python
# dense 模式:round 乘积而非 SiLU,匹配 dense activation.cuh
if constexpr (kDenseSiluAndMul) {
  const auto up_f32 = cast<float2>(up[i]);
  in[i] = cast<T2>(float2{(gate.x / (1.0f + expf(-gate.x))) * up_f32.x,
                          (gate.y / (1.0f + expf(-gate.y))) * up_f32.y});
}
```

## 性能声明

B200、TP1、`Qwen/Qwen3-8B-FP8`、DeepGEMM、BF16 activation、TRTLLM MHA、KV BF16、全 decode CUDA graph、torch.compile off;十组配对新服务器 A/B(5 组 baseline-first、5 组 fused-first),97.5% 双侧 Student-t 区间:decode concurrency 1 中位 TPOT 3.02064 ms → 2.96433 ms,paired -1.921% [-2.082, -1.759]%,10/10 组更快;decode concurrency 32 TPOT -0.919% [-1.643, -0.188]%(因 stock client 空文本计时缺陷,标注 provisional);prefill TTFT conc1 +0.662%(对 +1% 非回归界 inconclusive)、conc32 +0.062%(过界,非加速声明);整请求延迟 -1.901%/-0.837%。Profile:混合 10 步捕获中独立 activation launch 从 360 降到 36(净 -324)。作者明确声明"不是无条件性能放行";并发 32 decode 有 5084/5120 响应配对一致,差异原因未定位;GSM8K sanity 两版各 19/20。

## 验证与回归风险

B200 上 298 registered tests passed/1 skipped + 50 subtests:quantizer 全套 167 passed;扩展数值 sweep 240 case、184,023,040 FP8 bytes 与真实 CUDA SwiGLU+quantizer 逐 bit 相等;Compute Sanitizer memcheck 零错误;non-decode bypass 回归在 dispatch 改动前失败、改动后通过;真 MLP parity 与 CUDA graph replay 零容断言。风险面:fast path 门控失效会静默走 fused kernel(布局/dtype/后端校验集中在 helper);dense 数学与 legacy fused 模式(SiLU 先 round)数值不同,依赖 bit-exact 测试锁定;DeepGEMM packed 输入校验从"落到 fallback"改为 raise,构造非法 packed scale 的调用方会从静默错误变硬失败;仅覆盖 SM100/TP1/BF16,H100、TP>1、MoE 在 fast path 之外。

## 关联

同一 JIT quant kernel 体系:本批次 #34257(per-token FP8 quant AOT→JIT 迁移)奠定了 `sglang.kernels.jit` 基础设施,本 PR 在其 `per_token_group_quant` 上扩展;diff 注释引用 #6890 的 DeepGEMM 讨论。其余同分类编号 pack 未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
