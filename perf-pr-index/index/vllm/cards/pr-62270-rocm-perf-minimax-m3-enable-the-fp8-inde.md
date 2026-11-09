# PR #62270: [ROCm][Perf][MiniMax-M3] Enable the fp8 indexer cache on gfx942

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @akii96 |
| 合并 | 2026-11-02 |
| 分类 | Quantization (tier: major) |
| Label | rocm, ready, ci/build, minimax |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/62270 |

## 一句话结论

在 ROCm gfx942(MI300 系列/CDNA3)上为 MiniMax-M3 启用 fp8 indexer cache,同显存下 KV cache 容量 +32%,长上下文吞吐最高 +4%。

## 动机

MiniMax-M3 的 sparse attention indexer 在此之前于 ROCm 上只能用 bf16 index cache:一是 `select_indexer_impl_cls` 在所有 ROCm 平台拒绝 `indexer_kv_dtype="fp8"`;二是 index cache dtype 钉死为 e4m3fn(`torch.float8_e4m3fn`),gfx942 无法转换;三是 fused qknorm/rope/kv-insert 写入 kernel 用 `#if defined(HIP_FP8_TYPE_OCP)` 选转换分支,但该宏总是被定义(值为 0 或 1),导致 gfx942 走 OCP 分支、写出按 fnuz 解读就错误的字节(如 1.1→1.0、0.77→0)。而 fp8 index cache 只带约 2% 相对分数误差就能把 index-K 读带宽减半——这是长上下文 decode 中随上下文线性增长的主要开销(前序 #45892 已在 SM100 MSA 路径启用)。

## 优化点

机制上分三层:cache dtype 改用 `current_platform.fp8_dtype()`(gfx942 上为 e4m3fnuz,gfx950/CUDA 保持 fn),indexer 选择器经 `on_gfx942()` 谓词对 ROCm gfx942 放行 fp8(gfx950 继续走 AITER indexer 的 fp8 路径,CUDA 行为不变);fused CUDA kernel 的写出口按目标架构选 flavour(`__gfx942__` → fnuz),饱和上限取 `quant_type_max_v`(fnuz 224、fn 448),并要求 tensor dtype 与设备自身 flavour 一致、拒绝外来 flavour(gfx942 无 OCP convert,硬件会 trap);Triton 侧 index-cache insert(AITER sparse-PA 路径)同步按 `get_fp8_min_max()` 饱和 clamp,避免越界 key 写成 NaN。

## 关键改动

```text
vllm/models/minimax_m3/common/indexer.py: cache/query dtype 改用 current_platform.fp8_dtype();selector 对 gfx942 放行 fp8
csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu: ROCm 写出口按 __gfx942__ 选 fnuz/fn,饱和到设备各自上限
vllm/models/minimax_m3/amd/ops/sparse_pa.py: Triton insert kernel 增加 FP8_MAX clamp,防 NaN
vllm/models/minimax_m3/amd/mtp.py: gfx942 上 MTP draft 层允许跟随 fp8 indexer 配置
```

饱和写入片段:

```cuda
float const kFp8Max = static_cast<float>(quant_type_max_v<RocmFp8T>);
for (int i = 0; i < kElemsPerLane; i++) {
  float vv = fminf(fmaxf(elems[i], -kFp8Max), kFp8Max);
  dst[i] = fp8::cvt_c10<RocmFp8T>(vv).x;
}
```

## 性能声明

测试环境:8×MI325X(gfx942),nightly `vllm/vllm-openai-rocm:nightly-rocm100-39983a5e`,`EmbeddedLLM/MiniMax-M3-FP8-dynamic`,TP4,`ROCM_AITER_UNIFIED_ATTN`,`--kv-cache-dtype fp8`,`--max-model-len 262144`,`--gpu-memory-utilization 0.92`。

- KV cache 容量:4,554,368 → 6,020,992 tokens(+32%),262k 上下文最大并发 17.37x → 22.97x。
- `vllm bench serve`(openai-chat,random 数据集,精确 ISL/OSL,`--ignore-eos --temperature 0`,中位数):输出吞吐 8k/1k/8:468.6→470.4 tok/s(+0.4%);32k/1k/8:284.2→290.0(+2.0%);128k/1k/4:92.6→96.3(+4.0%),该档 TTFT p50 −2.2%(21,587→21,104 ms)、TPOT p50 −5.3%(22.12→20.94 ms)。短上下文基本持平。
- GSM8K(lm-eval 0.4.13,local-chat-completions,8-shot,temp 0,1,319 样本):strict-match 94.69%±0.62 → 95.00%±0.60,精度不变。

## 验证与回归风险

kernel 测试 gfx942 上 61 passed、32 skipped(nvfp4-only)、0 failed,新增 insert 饱和、外来 flavour 拒绝、fp8-vs-bf16 top-k 等价三类用例;`test_minimax_m3.py` 的 fp8 decode top-k 测试扩展到 gfx942(含 spec decode 的 per-row causal masking)与 decode_query_len>1。Buildkite CI 同步把 fused insert kernel 测试纳入 MI300 与 kernels attention 套件。风险面:改动只影响 ROCm MiniMax-M3 路径与该 fused kernel 的 dtype 校验(gfx950/CUDA 语义不变);`--attention-config '{"indexer_kv_dtype":"fp8"}'` 现在在 MI300 系列可用,但需要配合 `--kv-cache-dtype fp8` 场景验证。

## 关联

前序 #45892(SM100 MSA 路径 fp8 index cache);关联 NVIDIA 侧相近工作 #60930(CuTe prefill scorer)、#59979/#59980(FMHA/MSA),PR 明确说明本 PR 不是其重复。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
