# PR #37213: [XPU] Qwen3.8-flash-next enablement

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Xia-Weiwen |
| 合并 | 2026-09-24 |
| 分类 | Speculative Decoding |
| Label | intel, xpu, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 mamba_state_scatter_triton.py、qsa_indexer.py、hyperconnection.py、xpu fla kernel,新增 XPU 测试 |
| 链接 | https://github.com/sgl-project/sglang/pull/37213 |

## 一句话结论

Qwen3.8-Flash-Next 在 XPU 后端的 Stage 1 功能使能:修复四处设备可移植性 bug 与 QSA OOB gather,使其在 1x B60 TP4 上 BF16/FP8 端到端跑通(性能调优留待 Stage 2/3)。

## 动机

Qwen3.8-Flash-Next(经 #37500 合入 main)引入了 SGLang 的新架构特性:QSA 稀疏注意力 indexer、HyperConnection(门控多流残差混合)、PLE/N-gram 辅助 embedding,以及 hybrid linear+full attention 骨干上的 MTP/NEXTN 投机解码。这些新特性在多处假设 CUDA,导致模型在 XPU 后端无法运行。该 PR 是三阶段计划(功能就绪 → 可接受性能 → 峰值性能)中的 Stage 1:用现有可移植 Triton/torch kernel 让模型在 XPU 上无错跑通 E2E,不写新 SYCL kernel。

## 优化点

属使能/修复性质:(1) MTP/NEXTN 修 4 个设备可移植性 bug——Mamba state pool 硬编码 CUDA 设备、硬编码 CUDA sync 调用、GDN Triton wrapper 缺 kernel 参数(mamba scatter kernel 增传 `stride_h0_source` 以支持 per-slot pitch 非连续的 state pool)、XPU 指针打包进 scatter-kernel metadata 的 int64 溢出;(2) `mamba_state_scatter_triton` 的两个 kernel 从仅 CUDA 放宽到 CUDA/XPU(设备校验改为 `dst.is_cuda or dst.is_xpu`);(3) QSA indexer 把 `forward_cuda` 的编排体提取为平台无关的 `_forward_impl`,新增 `forward_xpu` 分发,底层复用 device-generic Triton/torch fallback(与 CUDA 无快速 kernel 时相同),稠密注意力走 sgl-kernel-xpu 现有 FMHA SYCL kernel;(4) HyperConnection 的 `torch.cuda.current_device()` 改为 `torch.get_device_module().current_device()`;(5) PLE/N-gram embedding、MoE(routed+shared)、vision encoder 验证即用(sgl-kernel-xpu 的 moe_align_block_size 与 non-paged-prefill FMHA SYCL kernel)。另修复 QSA compressed-KV 更新的真实 OOB gather bug(`update_key_state_and_compress`),后来由上游 #38346 的 clamp 取代本 PR 的手写版。

## 关键改动

```text
python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py: 校验放宽到 CUDA/XPU
python/sglang/srt/hardware_backend/xpu/kernels/fla/fused_sigmoid_gating_recurrent.py: 传入 stride_h0_source 支持非连续 state pool
python/sglang/srt/layers/attention/qsa/qsa_indexer.py: 提取 _forward_impl,新增 forward_xpu 分发
python/sglang/srt/layers/hyperconnection.py: 设备选择改 get_device_module
test/registered/xpu/test_qsa_indexer_xpu.py: XPU 分发测试 + OOB clamp 真机回归测试
```

设备可移植的编排分发:

```python
def forward_cuda(self, ...): return self._forward_impl(...)
def forward_xpu(self, ...): return self._forward_impl(...)
# _forward_impl: Portable orchestration shared by different platforms.
# Fast paths are gated per platforms inside kernel calls.
```

## 性能声明

本 PR 目标是功能正确性,无端到端性能数字。PR 正文引用的 2.15x speedup 来自被跟踪的上游 sgl-kernel-xpu PR #341(Xe2 FP8 W8A16 grouped GEMM,相对被评估的 W8A8 路径 1.18x-2.15x),是 XPU FP8 MoE 的长期方案,不属本 PR 的测量结果。重新验证覆盖 BF16 + FP8、1xB60 + TP=4,均在两台机器上对 main rebase 后复验通过。Stage 2(性能调优)尚未开始。

## 验证与回归风险

新增 `test/registered/xpu/test_qsa_indexer_xpu.py`(stage-b-test-1-gpu-xpu suite):`forward_xpu` 的 prefill/decode 分发与 QSAIndexer 行为一致性的 mock 测试,以及 #38346 compress-group OOB clamp 的真机回归测试(未 clamp 时 XPU 的 bounds-checked SYCL gather 会 abort 进程;用 padded token_k 构造无 clamp 的等价组验证压缩结果一致)。既有 `test_qsa.py` 同步更新以绑定 `_forward_impl`。CI Base/Extra/AMD ROCm 10 均显示失败标记(x)。风险点:rebase 后发现 QSA 强制 `page_size=64` 与 mamba radix-cache 的 `no_buffer` 策略冲突,且 XPU 的 `extra_buffer` 硬编码不支持,故 XPU 上该模型必须 `--disable-radix-cache`(已记入 docs/KNOWN_ISSUES.md);`mamba_state_scatter` 校验放宽使这两个 kernel 可被 XPU 数据触发;XPU FP8 MoE 目前走 Stage-1 的 generic-Triton fallback 停滞方案(性能未调)。

## 关联

同区域优化:#37500(Qwen 3.8 Flash Next 主线集成,本 PR 的 base)、#36601(AMD ROCm 平行使能 PR)、#36651(QSA/PLE PD 状态传输,已合入)、#38346(QSA compress-group OOB clamp,取代本 PR 手写版)、#35051(XPU USM 指针 uint64 ptr table 修复,取代本 PR 的 `_signed_i64()` workaround)、#36644(QSA FP8 KV cache 修复,仍 open);sgl-kernel-xpu #341(Xe2 FP8 W8A16 grouped GEMM);设计先例 #33354/#32227/#35775/#24390。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
