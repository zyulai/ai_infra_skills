# PR #59774: [ROCm] Octave KV (`ROCM_OCTAVE`): native 3-bit KV cache backend for AMD GPUs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @JartX |
| 合并 | 2026-10-26 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | documentation, performance, rocm, ready, ci/build |
| Diff | diffstat 缺失(pack 未记录);绝大部分为新增文件,pack 保留 3 个,26 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/59774 |

## 一句话结论

为 AMD GPU 新增原生 3-bit KV cache 后端 ROCM_OCTAVE,KV 容量提升至 fp16 的 4.6 倍,RDNA3 上 decode 较 TurboQuant 快 4.6×。

## 动机

ROCm 侧此前缺少 8 bit 以下且支持 MTP 的原生 KV 量化方案:TurboQuant/UltraQuant(#57057)等方案在 gfx942 上要么走 Triton/FlyDSL 回退、要么目标 gfx950,RDNA3(gfx1100)更是无人覆盖。作者检索了 `octave`、`turboquant rocm`、`kv cache 3-bit` 等关键词,确认没有 PR 以原生 HIP kernel 瞄准 RDNA 或在 ROCm 上以 8 bit 以下运行 MTP。RFC 见 #59773,格式文档在 `docs/design/rocm_octave.md`。

## 优化点

新增注意力后端 `ROCM_OCTAVE`,把 full-attention 层的 KV cache 保持在每个 value 3-4 bit:三种 cache dtype `octave_k3v3`/`octave_k3v4`/`octave_k3v3_compact`,head size 256 时每 slot 216/248/196 字节(fp16 为 1024)。存储机制:K/V 在写入时做符号翻转 + Walsh-Hadamard 旋转,再按组做 int8 码本(Lloyd-max 中点阈值)量化打包,scale 的选取使 `x·x_hat = |x|²` 以免量化衰减 score 与输出;decode 时 K 点积在码本值上用 int8 `dot4` 完成,query 每块量化一次,V 在 MFMA 路径用 `v_perm` 把码本字节精确展开为 fp16 denormal。decode 分两级(split-KV + reduce),MTP verify 的连续 query token 共享 KV 读;metadata builder 无 host sync,支持 UNIFORM_BATCH CUDA graph。仅 gfx1100 走 WMMA 快速 tier,gfx942 走 MFMA(`-amdgpu-mfma-vgpr-form` 保 occupancy),其余架构无 Triton 回退、直接不支持。QSA(qwen4_exp)稀疏 decode 可把 Octave 作主 cache(exact sink/recent window 仅 gfx1100)。

## 关键改动

```text
vllm/v1/attention/ops/rocm_octave.py: 格式定义与 PyTorch 参考(字节布局基准)
vllm/v1/attention/backends/rocm_octave_attn.py: 后端与 metadata builder
csrc/attention/octave_*.cu/_*.cuh: store/reduce/rotate/portable decode、
  gfx1100 WMMA、gfx942 MFMA decode+prefill、octave_torch_bindings.cpp
vllm/config/cache.py 等: 注册三个 octave cache dtype;CMake/平台选择/注册表
```

关键机制摘录(octave_attn.cuh,跨架构 int8 dot4):

```cpp
__device__ __forceinline__ int dot4_i8(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx942__)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  ... // portable fallback
#endif
}
```

## 性能声明

- 容量:4× RX 7900 XTX + Qwen3.8-27B 为 fp16 的 4.6× KV tokens;MI300X 上为 AITER fp8 cache 的 1.95×(7.0M vs 3.6M tokens,即 53 vs 27 个并发 128k 请求)。
- MI300X 逐层 attention(6 query/1 KV head):decode 单 token 128k 65 µs(AITER fp8 103 µs),32k 起 Octave 更快;64 请求@8k decode 220 vs 352 µs;prefill 带 8k 前缀两 2k chunk 3.28 vs 6.76 ms(无前缀 8k 略慢 2.76 vs 2.50 ms)。
- MI300X E2E(1× MI300X, TP1, MTP k=3, max-model-len 131072,Qwen3.8-27B-AWQ-INT4):`octave_k3v3` GSM8K 98.0、NIAH 28/28、decode@126k 52 tok/s、16 用户@8k 221 tok/s(fp8 AITER 为 97.0/28/28/50/235;fp16 ROCM_ATTN decode@126k 仅 6 tok/s)。AITER 在 16 用户吞吐与 100k TTFT 上仍领先,作者归因于带前缀 prefill 待优化。
- RDNA3(4× RX 7900 XTX, TP4, 380k 文档):逐层 decode 1 token `octave_k3v3_compact` 563 µs vs TurboQuant 2565 µs(4.6×)、fp16 2370 µs;MTP verify 4 tokens 0.58 ms vs TurboQuant 6.65 ms(11×);380k TTFT(MTP, 冷前缀)297 s vs fp16 371 s、TurboQuant 621 s。GSM8K:`octave_k3v3` 92.4 / fp16 91.8 / TurboQuant 89.8;NLL 相对 fp16 +0.0009~0.0015 nats。

## 验证与回归风险

测试 `tests/kernels/attention/test_rocm_octave.py`:gfx1100 105 passed;gfx942 87 passed + 18 skipped(sparse decode 为 gfx1100 专属);对照 PyTorch 参考覆盖三种 dtype 的 store/decode/prefill、1/4 query tokens(MTP)、padding 与 1-256 KV splits、fp16/bf16、带前缀多请求 prefill、QSA sparse decode;作者以故意破坏内核验证测试灵敏度;`VLLM_GPU_SYNC_CHECK=error` 下 gfx942 服务 40 题 GSM8K(MTP k=3)无 sync error(97.5%)。gfx1100 不在 CI,为本地验证。风险:仅 head size 256 + 64 RoPE dims(Qwen3-Next/3.5/3.6/3.8),每 GPU 最多 8 query heads per KV head;`customize_spec` 需要在 vLLM config 下运行以获取 rotary dims;对其他后端与 cache dtype 无改动。

## 关联

- RFC:PR #59773
- 相邻方案:UltraQuant(#57057,FP4/FlyDSL,gfx950)、KVarN(#46613/#46812)、#43835(HIP int8/int4 CDNA)、#40396(Triton TurboQuant decode MI300X)、#53231(TurboQuant for Gemma 4 ROCm)、#59132(RDNA Triton attention 无 KV 量化)
- 同分类 KV Cache:与本索引 TurboQuant/UltraQuant 卡对照

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
