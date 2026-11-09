# PR #60413: [Perf][ROCm][Kernel] Add W4A8 int8-dot MXFP4 GEMV for RDNA3/RDNA3.5 d…

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xiao-llm |
| 合并 | 2026-10-24 |
| 分类 | Quantization |
| Label | documentation, rocm, ready, ci/build, quantization |
| Diff | diffstat 缺失;pack 明示 diff 因超长截断(3 个文件未含) |
| 链接 | https://github.com/vllm-project/vllm/pull/60413 |

## 一句话结论

RDNA3/3.5 上 MXFP4 linear 新增 W4A8 int8-dot GEMV decode 路径,较 emulation 快 5.8x。

## 动机

RDNA3/RDNA3.5(gfx11)上 MXFP4 linear 此前 decode 走 emulation(dequant + `F.linear`),很慢;#46676 提供了 W4A16 scalar FMA(WMMA for M>8)路径,但小 M 场景每 weight 指令数偏多。本 PR 针对小 M(1–8 token)采用不同算法,每 weight 指令少 2–5x,方案沿用 ggml 的 MMVQ / q8_1 路径(llama.cpp,MIT)。

## 优化点

decode(1≤M≤8)时激活按 32 元素组量化为 int8(对称、每组一个 fp32 scale);E2M1 权重值×2 恰为 [-12,12] 内整数,经 16 项常量表精确映射到 int8;每组用 8 条 `v_dot4_i32_iu8`(`__builtin_amdgcn_sudot4`)归约,每组一次 fp32 乘同时应用 E8M0 weight scale、activation scale 与 1/2。M>8(prefill)保持与 `EmulationMxfp4LinearKernel` 相同的数学(activation QDQ、dequant、`F.linear`),即 weight-only。M 分派封装在 custom op 内,torch.compile 与 CUDA graph 只看到一个 op。权重直接按 Quark OCP MX checkpoint 布局读取([N,K/2] uint8 E2M1、[N,K/32] uint8 E8M0),无 repack、无额外显存。整体 opt-in:`VLLM_ROCM_MXFP4_W4A8=1` 或 `--quantization-config.linear.activation int8_per_group_32`;flag 关闭或非 gfx11 时 kernel 选择不变。配套给 kernel selection 机制加 `_supports_quant_scheme`,按 activation 格式分派;device 代码仅 `__GFX11__` 编译,host 入口运行时再查 gfx11。

## 关键改动

```text
csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu: int8 activation quant + W4A8 GEMV(模板 M=1..8、fp16/bf16)
vllm/model_executor/kernels/linear/mxfp4/rdna3_hip.py: Rdna3W4A8MxFp4LinearKernel,M-based dispatch custom op
vllm/model_executor/kernels/linear/__init__.py: ROCm MXFP4 kernel 列表首位;_supports_quant_scheme 机制
vllm/envs.py / vllm/config/quantization.py: VLLM_ROCM_MXFP4_W4A8 开关与 int8_per_group_32 activation key
tests/kernels/quantization/test_mxfp4_w4a8_rdna3.py: 新增测试
```

```c
const int w0 = kE2M1x2[two & 0xF];  // 2 * E2M1, exact in int8
...
isum = __builtin_amdgcn_sudot4(true, w4[j], true, ap[j], isum, false);
acc[m] += (float)isum * (w_scale * a_s[(int64_t)m * groups + g]);
```

## 性能声明

AMD Strix Halo(gfx1151,RDNA 3.5),`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`,`--max-num-seqs 1` 单流,CUDA graphs + torch.compile 开,`vllm bench serve` random 128 in / 256 out,MTP=3,三配置同样附加 MXFP4 lm_head(该量化非本 PR 内容):decode tok/s——本 PR **28.89**、#46676(W4A16)26.19、emulation 4.94;即 **10% faster than #46676 at equal accuracy, 5.8x faster than the default emulation path**。GSM8K 5-shot(lm-eval,greedy,1319 题):本 PR 与 #46676 均 96.51%,emulation 96.75%。参考:llama.cpp(ROCmFP4 GGUF,含自量化 lm_head)同机 29.1 tok/s。注意:decode 用 int8 activation(细于 MXFP4),prompt 用 bf16 weight-only;若 prompt 也套 MXFP4 activation QDQ,GSM8K 掉至 ~80%。

## 验证与回归风险

`pytest tests/kernels/quantization/test_mxfp4_w4a8_rdna.py` 与 `test_mxfp4_kernel_selection.py`,gfx1151 全通过。覆盖:M=1..8 × 四形状 × fp16/bf16 对参考实现、与 bf16 dequant 的接近度(rel<1e-2)、M=0/9 拒绝、`opcheck`、CUDA graph capture/replay、M=1/8/9/64 分派、selection 仅 flag on + gfx11 生效(MXFP6 与非 ROCm 拒绝)。风险:默认关闭;int8 activation rounding 与 checkpoint 声明的 MXFP4 activation QDQ 不同(已验证但属语义变更);每个 linear 多一次 activation quant launch(融合是 follow-up);M>8 每次调用都 dequant。

## 关联

#46676(gfx1100 的 W4A16 MXFP4 kernel;算法独立,两 PR 仅共享 kernel 注册列表)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
