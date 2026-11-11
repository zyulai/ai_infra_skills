# PR #48638: [Perf] [NVIDIA] moe: Add an opt-in persistent MXFP4 decode MoE kernel for Kimi-K3.1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-11-07 |
| 分类 | MoE(tier: major) |
| Label | hicache, run-ci, jit-kernel |
| Diff | 素材未提供 diffstat(超长 diff 被截断:保留前置 PTX 头改动与新文件 moe_mxfp4.cuh 前半,另 16 个文件略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/48638 |

## 一句话结论

为 Kimi-K3.1 新增 opt-in 的 persistent MXFP4 decode MoE kernel(SM100/SM103),B300 上单层 MoE 比 flashinfer 快 12-28%。

## 动机

Kimi-K3.1 的 decode MoE 此前走 flashinfer trtllm-gen 的 MXFP4 routing + FC1 + FC2 调用。本 PR 面向 TP-MoE rank(896 本地专家、intermediate shard 384、top-16、MXFP4 权重为引擎 flashinfer_mxfp4 布局、MXFP8 激活、SiTU)且至多 256 tokens 的 decode 场景,用两个 JIT kernel 取代该 flashinfer 调用,经 `SGLANG_K31_DECODE_MOE=1` 显式开启(默认关)。

## 优化点

两个 JIT kernel 以 PDL 串联。(1) routing kernel:CTA 0 由预算好的 top-k 构建 per-expert token 表(expert histogram + scan 得出 token 排列),CTA 1 复位 MoE kernel 的 work counters;输入可接 plain ids 或 route+quant handoff 的 packed `(id << 16) | bf16(weight)` 形式。(2) persistent tcgen05 kernel:FC1 + SiTU + MXFP8 requant 写入 intermediate buffer,随后 FC2;每个 CTA 先从全局队列认领 FC1 item、耗尽后认领 FC2 item,FC2 等待同一虚专家的全部 FC1 item 完成;权重按 flashinfer_mxfp4 布局就地读取(加载期不 repack),输出引擎的 deferred-finalize 行(gemm2_out 按排列行序 + expanded_idx),top-k 权重仍由引擎最后套用。专家数、hidden/intermediate、top-16、SiTU 常量均为编译期常量;scratch 每次调用分配,无跨调用状态。runner 层仅对 standard top-k 且 T<=256 分流(deferred 路径直接返回 kernels 的输出,非 deferred 用 moe_finalize_fuse_shared(out=) 直写引擎 symmetric 输出 buffer),其余回落 flashinfer;层支持性在权重准备后由 k31_decode_moe_supported 一次性判定,不满足时日志打出第一个不满足的条件。与 #48586 fused routing 的 handoff 冲突由 guard 解决:select_trtllm_routed_body 对 k31_decode_moe 层且 T<=256 返回 None,保住 packed handoff。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/k31_decode/moe_mxfp4.cuh: 新文件,route_kernel + persistent moe_kernel(tcgen05、PDL、L2 evict_first hint、tma gather4)
python/sglang/kernels/ops/moe/k31_decode_moe.py: 两个 kernel 的 op 封装
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: T<=256 分流到新 kernel,其余回落 flashinfer
python/sglang/srt/layers/quantization/mxfp4.py: k31_decode_moe_supported 支持性判定
python/sglang/srt/environ.py: SGLANG_K31_DECODE_MOE 开关
```

```cpp
//   route_kernel (2 CTAs): CTA 0 builds the permutation tables (expert histogram
//       + scan of the precomputed top-k), CTA 1 resets the MoE kernel's counters;
//   moe_kernel (persistent): FC1 + SiTU + MXFP8 requant, then FC2, writing the
//       engine's deferred-finalize rows (gemm2_out in a permuted row order +
//       expanded_idx); the engine applies the top-k weights later.
```

## 性能声明

B300,per MoE layer,real routing,cold L2,CUDA graphs:非 deferred 路径在 T 96-240 比 flashinfer 快 18-24%;deferred 路径(do_finalize=False)在 T 30-186 快 12-28%。PR 未给出端到端 serving 数字。

## 验证与回归风险

精度(B300,随机 checkpoint 布局 MXFP4 权重 + 真实 decode routing):FC2 rows 对 fp64 参考(MXFP8 intermediate)mean relative error 1.09e-3 至 1.12e-3,flashinfer 同输入 1.08e-3 至 1.11e-3;finalized output 与 flashinfer 差 1e-5 至 1.1e-4;expanded_idx 是排列、逐 bit 确定,packed handoff 与 plain ids 的行逐 bit 相等;CUDA graph replay 与 eager 逐 bit 相等。测试:test_moe_finalize_fuse_shared.py 131 passed/6 skipped,test_tiny_gemm.py 87 passed/4 skipped(B300,nvcc 13.4);所有 JIT 源(含 PTX 头)在 sm_100a 编译,top-k v2 build 与 main SASS 一致。风险:仅覆盖 T<=256、standard top-k、896 experts 全本地等条件,其余层回落 flashinfer;与 #48586 的 handoff 交互依赖 guard;默认关闭。

## 关联

#47469(前置:PTX 头拆分 sgl_kernel/ptx/{barrier,async,tcgen05}.cuh)、#47683(tiny GEMM include 修复)、#48174(JIT moe_finalize_fuse_shared out=)、#48586(fused decode routing,handoff 冲突与 guard)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
