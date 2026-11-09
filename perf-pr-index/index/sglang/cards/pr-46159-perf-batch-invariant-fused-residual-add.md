# PR #46159: [Perf] batch-invariant: Fused residual add + RMSNorm kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @aarnetalman |
| 合并 | 2026-10-28 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, deterministic |
| Diff | 素材未提供 diffstat(diff 含 batch_invariant_ops.py、layernorm.py 及 1 个新测试文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/46159 |

## 一句话结论

deterministic 模式下把 eager 的 add+RMSNorm 回退路径换成单个融合 Triton kernel,单请求 decode 吞吐 +56~73%。

## 动机

deterministic 模式中,带 residual 的 RMSNorm(每层 decoder 两个 norm)回退到 eager PyTorch,约需十次 kernel launch 而非一次融合 kernel。该回退占确定性 decode 时间很大份额:单请求 decode(1k prompt / 256 输出,CUDA graphs off)GPU 时间中,Qwen3-8B 回退约 205ms(34%,deterministic 603ms vs 正常 323ms),Qwen3-30B-A3B 约 215ms(32%),Phi-4 约 205ms(22%)。而正常模式下同样的 norm 作为融合 kernel 只需 12.6ms(Qwen3-8B)。

## 优化点

在 `batch_invariant_ops.py` 新增 `_fused_add_rms_norm_kernel` 与 `fused_add_rms_norm_batch_invariant`,风格与既有 `rms_norm` kernel 一致:每个 program 处理一行,先在 float32 中把 residual(及可选 post-residual addend)加到输入,把和存为新 residual(输入 dtype 或 float32 的 `fp32_residual`),再对 float32 和做 RMS 归一化并乘 weight。reduction 在单行上以固定 BLOCK_SIZE=1024 运行,因此单行结果与 batch 大小无关,保持 batch-invariant。`layernorm.py` 在 `RMSNorm` 的 `forward_cuda`/`forward_aiter`/`forward_hip`/`forward_xpu` 四个 batch-invariant 分支中,凡带 residual 且 `forward_native` 数学适用的调用都改走新 kernel;`cast_x_before_out_mul`、variance-size override、padded output、FSDP RL target 四种情形继续回退 `forward_native`。仅影响 deterministic 模式,使用与 `forward_native` 相同的 float32 数学;GemmaRMSNorm 路径(Gemma、Qwen3-Next/3.5/3.8)不受影响,正常模式不变。

## 关键改动

```text
python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py:
  新增 _fused_add_rms_norm_kernel 与 fused_add_rms_norm_batch_invariant
python/sglang/srt/layers/layernorm.py: RMSNorm 四个 forward 分支接入融合 kernel,
  新增 _batch_invariant_add_rms_norm 守门函数
test/registered/unit/batch_invariant_ops/test_fused_add_rms_norm_batch_invariant.py:
  新增单测,注册 CUDA CI 与 AMD MI35x CI
```

layernorm.py 中的接入点:

```python
if is_batch_invariant_mode_enabled():
    if residual is not None:
        fused = self._batch_invariant_add_rms_norm(
            x, residual, post_residual_addition
        )
        if fused is not None:
            return fused
```

## 性能声明

MI350X,deterministic 模式(CUDA graphs on),Qwen3-8B / Qwen3-30B-A3B / Phi-4,`benchmark.serving --dataset-name random --random-range-ratio 1.0 --request-rate inf`:

- Qwen3-8B,1k in/256 out,并发 1:105 → 173 tok/s(+65%),median ITL 9.42 → 5.64ms(-40%)
- Qwen3-8B,并发 16(1k/256):1,663 → 2,033 tok/s(+22%),ITL -20%;8k/256 并发 16:582 → 653 tok/s(+12%),ITL -15%
- Qwen3-30B-A3B,并发 1:94 → 163 tok/s(+73%),ITL 10.5 → 6.0ms(-43%);并发 16:1,070 → 1,282 tok/s(+20%)
- Phi-4,并发 1:72 → 112 tok/s(+56%),ITL -37%
- 参照:正常模式同 setup 并发 1 的 ITL 为 4.8ms(Qwen3-8B)/ 4.4ms(Qwen3-30B-A3B),即本 PR 关闭了确定性模式单请求 decode 的大部分差距

## 验证与回归风险

新单测验证 batch 不变性:batch 1/3/17/64 下逐行输出与新 residual `torch.equal` 位级一致;新 residual 与 eager float32 和完全一致,输出在舍入内匹配;覆盖 hidden 2048/4096/5120/2880、bf16/fp16、有无 addend、fp32_residual 开关。E2E:Qwen3-8B TP1/TP2 与 Qwen3-30B-A3B 通过 `test_deterministic` 的 single/prefix/radix_cache 检查及位级 batch-size 校验(1,275/1,275 一致);GSM8K(sgl-eval 0.1.2,Qwen3-8B deterministic)93.18%,改前 93.25%(deterministic)/ 93.48%(normal)。测试环境 MI350X、ROCm 7.2.4。回归风险集中在四个 forward 分支的 batch-invariant 路径;四种回退情形与无 residual 调用、正常模式行为不变。

## 关联

依赖正文列出的 MI350X deterministic-mode 系列 PR:#43788、#44045、#44438、#44489、#45599、#45601、#34543(即 MI350X 上 deterministic 模式当前的运行配置)。CI 显示 PR Test (Base) 通过,PR Test (Extra) 与 AMD ROCm 10 失败。
