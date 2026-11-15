# PR #44861: [Fix] [NVIDIA] modelopt: Keep SM12x FP8 batch-invariant in deterministic mode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yibol9768-alt |
| 合并 | 2026-11-15 |
| 分类 | Quantization(tier: major) |
| Label | quant, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/44861 |

## 一句话结论

SM12x deterministic 模式下,ModelOpt static-FP8 linear 改走固定 N=64/K=128 reduction tiles(M-only tiles 16/32/128)的 Triton scaled MM,消除 cold prefill 与 cache-hit batch 间 reduction schedule 差异带来的输出不一致;三阶段 pipeline 相对两阶段把 Graph 矩阵 pooled median wall time 再降 2.4–10.3%(对上游 Graph reference 为 −2.1% 到 +9.2%,不构成普遍加速)。

## 动机

SM12x 上 ModelOpt static-FP8 linears 在 deterministic inference 开启时仍可能于 cold prefill 与更短的 prefix-cache-hit batch 之间改变 reduction schedule,同一输入序列随后得到不同分数。关联 #41351 的 ModelOpt 路径;诊断与初始 fallback 归功 `furedericca-lab`,small-M gap 由 `twu3202` 定位。作者注明不声称解决该 thread 中所有 hybrid/GDN 问题。

## 优化点

- 仅 SM12x deterministic 模式:复用 Triton scaled MM,固定 N=64/K=128 reduction tiles,M-only tiles 16/32/128(`block_m`:m≤16 取 16,m<128 取 32,否则 128)。Normal-mode 与 forced-Marlin dispatch 不变。
- partial K tiles 的 masked A/B load 置零(`tl.load(..., other=0.0)`)。
- 2026-10-22 的 refinement 只把 deterministic GEMM 的 pipeline 从两阶段改为三阶段(`num_stages=3`),tiles、warps 4、量化、算术与 scaling 全不变——重叠加载同时保持 reduction schedule。
- `modelopt_quant.py` 的 `ModelOptFp8LinearMethod.apply` 在 `use_sm120_fp8` 且 `get_exec().deterministic.enable_deterministic_inference` 时分派到新的 `apply_fp8_linear_deterministic`。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8_utils.py(新增 apply_fp8_linear_deterministic): 固定 N=64/K=128 tiles、M-only tiles 16/32/128 的 Triton scaled MM,num_stages=3
python/sglang/srt/layers/quantization/modelopt_quant.py: deterministic 模式分派到 apply_fp8_linear_deterministic
python/sglang/kernels/ops/gemm/fp8_kernel.py: scaled_mm_kernel 对 partial K tiles 的 masked A/B load 置零(other=0.0)
test/registered/kernels/ops/gemm/test_sm120_fp8_linear.py: 新增 cached-suffix 对照与 CUDA Graph replay 用例
```

```python
    # Keep K reduction and N tiling fixed so prefill and cache hits agree.
    # Only the row tile varies, avoiding wasted decode work and small tiles
    # on long prefills without changing each row's reduction.
    m = input_2d.shape[0]
    block_m = 16 if m <= 16 else 32 if m < 128 else 128
    output = triton_scaled_mm(
        qinput, weight, x_scale, weight_scale, input.dtype, bias,
        block_size_m=block_m, block_size_n=64, block_size_k=128,
        use_heuristic=False, num_warps=4, num_stages=3,
    )
```

## 性能声明

(2026-10-22 在一块 RTX 5090、400W、TP=1 上,官方 SGLang nightly `0.5.23.dev20261020+g55b2aef7ad`:PyTorch 2.14.1、Triton 3.8.0、FlashInfer 0.7.2;模型 `nvidia/Qwen3-8B-FP8`,BF16 激活、自动 FP8 KV、deterministic mode、Triton attention)

- Graph 矩阵(每请求 32 个强制输出 token,每 cell 每 boot 1 warmup + 5 个样本,PR 臂 boot 顺序 A(两阶段)/B(三阶段)/B/A,中位数池化每臂 10 个样本):两阶段→三阶段 128/1: 0.3119→0.2796 s(−10.3%);128/4: 0.3539→0.3251 s(−8.1%);2048/1: 0.5314→0.5143 s(−3.2%,boot 间有波动);2048/4: 0.8900→0.8688 s(−2.4%)。上游 Graph reference 为 0.2684/0.3322/0.4766/0.7953 s;三阶段路径相对它 wall time 从 −2.1% 到 +9.2%。Eager 差异约 0.5% 以内,不声称 eager 加速。stage-only 更新把该 Graph 矩阵 pooled median wall time 降 2.4–10.3%,短 prompt 上最稳定;不消除 determinism/cost 折中,也不构成普遍加速。历史上两阶段测量为 5.7–25.8%(nightly)与 5–20%(mixed runtime)Graph 开销。十个成功 job 均低于 72 C,峰值 66 C。
- 定性:隔离 probe 表明这些 case 中 GEMM 是主要组件成本;加深 pipeline 对若干 projection shape 更快,64 个探索 case 全部保持 exact。

## 验证与回归风险

- 既有 SM120 注册测试文件完整通过:137 passed;额外两阶段/三阶段张量对照(BF16/FP16、两 seed、八个 N/K shape、M 边界、bias、strided 输入、K=80 tail)288/288 bit-exact,且与 full-batch suffixes 完全一致。
- 真实模型:prefixes 128/1024/4096、batches 1/4/2、各两轮,每 PR 臂两个 Graph boot + 一个 eager boot:18 组 paired cache case exact;cold/cached 分数与 warm 重复零 delta;selected logprobs 跨 pipeline 版本也匹配;完整固定输出(含 warmup 与测量轮)180/180 文本与 token 序列 exact;GSM8K rows 32–159、cap 512、Graph mode:128/128 文本与 token 序列 exact,各 120/128 正确。
- 作者注明:128 题复用既有数据,验证的是相对既有 PR 输出的保持性,不是独立质量评估、准确率提升或与上游统计等价的证据;未修改的 nightly FP8 路径在 fresh Graph control 中仍有 nonzero cold/cache selected-logprob delta;独立 RTX PRO 6000 Max-Q / Qwen3-14B-FP8 检查在更早 head `5f90880cc2` 佐证本修复(先于 stage-only refinement);原始 hybrid TP=2 模型、SM121 与 torch.compile 未本地测试。本地验证与独立复现不构成上游 CI 或 maintainer 批准。

## 关联

#41351(相关 ModelOpt deterministic 问题 thread;诊断与初始 fallback 出处);重复检查已核 #34429(小 M dispatch)、#42865(block/MXFP8)、#38921(SM89 prequantized inputs)、#44076(GDN chunk alignment)——均未实现该静态 per-tensor deterministic 路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
