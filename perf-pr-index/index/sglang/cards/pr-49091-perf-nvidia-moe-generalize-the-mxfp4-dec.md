# PR #49091: [Perf] [NVIDIA] moe: Generalize the MXFP4 decode MoE kernel to covered shapes and DeepSeek-V4

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @DarkSharpness |
| 合并 | 2026-11-08 |
| 分类 | MoE（tier: major） |
| Label | run-ci, jit-kernel |
| Diff | +256/-137, 1 file（pack diffstat 缺失且 diff 在 40k 截断、注明另有 11 file(s) dropped；计数按 pack 保留的 diff 文本） |
| 链接 | https://github.com/sgl-project/sglang/pull/49091 |

## 一句话结论

把 SM100 MXFP4 decode MoE 持久内核泛化为 shape traits 并接入 DeepSeek-V4/V4.1，K3.1 TP8 下 0.66–0.81× flashinfer。

## 动机

#48638 的持久 MXFP4 decode MoE 内核只支持 Kimi-K3.1 TP8 的硬编码常数（896 experts、top-16、hidden 3584、intermediate shard 384、kNCap 32），其他模型的 MXFP4 decode 路径只能走 flashinfer。本 PR 把内核泛化为任意被覆盖的 per-rank shape，并接入 DeepSeek-V4 / V4.1 的 MXFP4 路径；仍为 opt-in，仅服务 ≤256 token 的调用。

## 优化点

内核移至 `moe/mxfp4_decode/`，shape 改为模板 traits `<E, top-k, H, I, BlockN>`（E 为本 rank local experts，BlockN 为每个 virtual expert 的 token 数即 MMA N），workspace、smem 环深、virtual expert 上限等均由 traits 编译期推导。FC1 激活成为模板参数：`situ<beta, linear_beta>` 或 `swiglu<limit>`（clamped SwiGLU，对齐 flashinfer fused FC1 epilogue）。routing 读平铺 ids（任意 top-k）并接受 EP `expert_offset`：一次无符号比较同时拒绝其他 EP rank 的 expert 与 -1 id（invalid id 以无分支方式 OR 0 进 lane-distinct word），其 `expanded_idx = -1`；routing kernel 按调用所需的 bitmask word 数实例化（小 T 跳过未用 word）。一个 host 入口用同一 workspace 启动 routing + MoE 两 kernel（PDL 链接）。小 T 调度：CTA 最多持有两个 FC1 item（gated claim prefetch，早期 CTA 不再囤积全局队列）。Op 更名 `mxfp4_decode_moe`（替代 `k31_decode_moe`），带 `activation`/`activation_constants`/`expert_offset`/`is_supported_shape()`；引擎侧统一 flag `SGLANG_USE_CUDA_MXFP4_DECODE_MOE`（默认关，`SGLANG_K31_DECODE_MOE` 保留为 deprecated 别名），形状/激活/MXFP8 激活均被覆盖的层在 ≤256 token 调用时使用，否则 flashinfer；接入 `Mxfp4MoEMethod`（SiTU，新增 EP）与 `Mxfp4FlashinferTrtllmMoEMethod`（DeepSeek-V4/V4.1 SwiGLU limit），共享 `run_cuda_mxfp4_decode_moe`。

## 关键改动

```text
python/sglang/kernels/jit/csrc/moe/mxfp4_decode/moe_mxfp4.cuh: 自 k31_decode 移入并模板化（pack 可见部分，其余 11 files 被 drop）
python/sglang/kernels/ops/moe/…（mxfp4_decode_moe op；依 PR 正文，pack diff 未含）
python/sglang/srt/layers/quantization/…（Mxfp4MoEMethod / Mxfp4FlashinferTrtllmMoEMethod 接线；依 PR 正文，pack diff 未含）
```

```cuda
// local expert ids; one unsigned compare rejects other ranks' experts and -1
ids[j] -= p.expert_offset;
valid[j] = in[j] && ids[j] < kNumExperts;
uint32_t* const word = valid[j] ? &s_bits[t[j] / 32][ids[j]] : &s_bits[0][lane_id];
atomicOr(word, valid[j] ? 1u << (t[j] % 32) : 0u);   // branch-free, one ATOMS per id
```

## 性能声明

B300，µs/layer，真实 routing（captured K3.1 decode batch；DSV4.1 verify batch 每请求 6 token），cold L2，CUDA graph，flashinfer 0.7.2 锁定 sweep 得到的 best tactic：K3.1 TP8 T=6/24/96/252：27.1/65.8/150.7/208.0 vs 41.2/89.3/187.9/257.1（ratio 0.66/0.74/0.80/0.81）；K3.1 TP4 EP2：26.9/65.3/152.8/206.4 vs 36.1/81.9/167.9/227.2（0.75/0.80/0.91/0.91）；DeepSeek-V4.1-Flash TP4：28.7/58.4/123.9/180.7 vs 31.4/70.7/139.5/206.2（0.91/0.83/0.89/0.88）。对 #48638（K3.1 TP8）：T≤12 快 3–5%，T≥36 持平。

## 验证与回归风险

注册测试 `test_mxfp4_decode_moe.py`：7 个 shape（K3.1 TP8 / TP4 EP2、V4.1-Flash TP4、V4-Flash TP4/TP8、V4-Pro TP4/TP8）× T{1,7,48,96,256} × BlockN{16,32} 对 fp64 参考，覆盖 EP offset 与 -1 ids、packed/unpacked top-k、确定性、CUDA graph replay；DeepSeek-V4 shape 的 FC2 行与 flashinfer bitwise 相等。V4.1 `Mxfp4FlashinferTrtllmMoEMethod.apply()` flag on/off 输出 bitwise 相等（TP4 与 EP2，T 1–256，finalized 与 deferred）。K3.1 TP8 内核 SASS 与 #48638 参数化前一致。风险：默认关闭的 opt-in 路径；仅 SM100（tcgen05）、仅 ≤256 token；shape 需满足 traits 的 static_assert（H%512、I%128、E%4、kNumExperts≤1024 等）；其他 spec/量化路径未验证。

## 关联

前序 #48638（Kimi-K3.1 TP8 专用内核，本 PR 的直接基础）；decode 小 batch 上替代 flashinfer TRT-LLM-gen MXFP4 MoE 路径。无其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
