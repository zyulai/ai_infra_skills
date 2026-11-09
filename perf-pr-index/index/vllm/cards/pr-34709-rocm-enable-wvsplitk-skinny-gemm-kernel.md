# PR #34709: [ROCm] Enable wvSplitK skinny GEMM kernel for RDNA4/gfx1x decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @laudney |
| 合并 | 2026-03-20 |
| 分类 | Attention / Kernel (tier: major) |
| Label | new-model, rocm, ready, ci/build |
| Diff | diffstat 缺失(素材未提供 additions/deletions/文件数);diff 全文含 3 个文件(skinny_gemms.cu 为主体) |
| 链接 | https://github.com/vllm-project/vllm/pull/34709 |

## 一句话结论

把 wvSplitK/wvSplitKQ skinny GEMM 从 gfx9 扩展到 RDNA gfx11/gfx12,decode 阶段(AWQ-4bit)token/s 提升约 15%。

## 动机

decode 阶段的 GEMM 形状极瘦(M=1..4),此前 vLLM 的 wvSplitK/wvSplitKQ 专用 kernel 仅覆盖 gfx9(MI 系列);RDNA 用户只能回落到 `torch.nn.functional.linear`(非量化)或通用 `torch._scaled_mm`(FP8),在瘦形状下效率低。该 PR 把这组 kernel 移植到 gfx1100/1101/1150/1151/1200/1201,让 RDNA(含 RDNA4/gfx12)decode 也走专用 skinny GEMM 路径。

## 优化点

Commit 1 面向非量化 BF16/FP16 wvSplitK:新增 `__HIP__GFX1X__` 编译期宏与 `on_gfx1x()` 运行时检测,launch 宏 `WVSPLITK_CFG`/`WVSPLIT_TILE_CFG` 参数化 THRDS 与 WvPrGrp,使 wave64(gfx9,64 线程)与 wave32(RDNA,32 线程)共用模板;归约阶段把 gfx9 专用的跨行指令(row_bcast:15/31)换成 RDNA wave32 的 `row_shr:1 + __shfl_xor(val, 16)`;half 路径的 DOT2C 在 gfx1x 上改用 `v_dot2_f32_f16`(VOP3-P/dot10-insts)而非 gfx9 的 `v_dot2c_f32_f16`(VOP2);MFMA bf16 分支加 `__HIP__GFX9__` guard 防止 Clang 校验死分支里的 gfx9 专用 asm。Commit 2 面向 FP8 per-tensor wvSplitKQ:gfx12 路径用标量 float 累加器替代 gfx9 的 floatx16 MFMA 向量,点积用 `__builtin_amdgcn_dot4_f32_fp8_fp8()`(硬件按目标架构解释 FP8 位型:gfx9 fnuz、gfx12 fn),归约用 DPP row_shr:8/4/2/1 + shfl_xor,写回 lane 统一为 `THRDS-1`;host 侧 `WVSPLITKQ` 宏按 `on_gfx12()` 运行时选 THRDS=32/WvPrGrp=16(wave32)或 THRDS=64(wave64)。Python 调度(`rocm_unquantized_gemm_impl`)的 `use_skinny` 条件加入 `on_gfx1x()`,gfx1x 与 gfx9 共享 wvSplitK → LLMM1 分发,并移除此前 LLMM1-only 的 gfx1x workaround;另将 `device_guard` 前移到 wvSplitKrc 的 tensor 分配之前。

## 关键改动

```text
csrc/rocm/skinny_gemms.cu: GFX1X/GFX12 宏、wave32 DPP 归约、gfx1x DOT2C、gfx12 dot4 FP8 累加、参数化 launch 宏
vllm/model_executor/layers/utils.py: use_skinny 条件加入 on_gfx1x(),gfx1x 走 wvSplitK→LLMM1 分发
tests/kernels/quantization/test_rocm_skinny_gemms.py: wvSplitKrc 增加 bias_mode 3,容差改为 eps*sqrt(k)
tests/model_executor/layers/test_rocm_unquantized_gemm.py: 新增 gfx1x/gfx950 分发路径 mock 单测
```

## 性能声明

PR 正文给出的数字:**~15% decode token/s improvement on AMD Radeon AI PRO R9700 (gfx1201)**;测试计划中确认项为 Qwen3-Coder-30B-A3B AWQ-4bit 推理于 gfx1201,输出正确且约 15% decode speedup;Qwen3-14B-FP8 于 gfx1201 FP8 路径功能正常(注明 block-wise FP8 模型走 Triton kernel,per-tensor FP8 才走 wvSplitKQ)。PR 未注明具体吞吐数值与请求配置。

## 验证与回归风险

测试计划中仍为未勾选状态:gfx9/MI 系列无回归(kernel guard 保留原 wave64 路径,属作者预期但未验证)、`test_rocm_wvsplitk_kernel` 与 `test_rocm_unquantized_gemm` 分发单测。diff 中新增 `test_rocm_unquantized_gemm.py`(gfx1x wvSplitK 路径、N>4 回落、gfx950 wvSplitKrc 路径三个 mock 测试)并把 wvSplitK 容差改为 `eps*sqrt(k)`(注明 fp16 累加误差随 sqrt(K) 增长)。风险面:`skinny_gemms.cu` 是 gfx9 与 gfx1x 共用源文件,inline asm 与模板实例化变化会影响 MI 系列编译产物;`use_skinny` 条件放宽使所有 gfx11/gfx12 的非量化 decode GEMM 改走新路径;wvSplitKQ 的 N=3/4 情形 UNRL 由 2 调为 1。

## 关联

同区域(RDNA4/gfx12 系列):#34740(supports_fp8() 替代 arch check 做 FP8 门控)、#34741(RDNA4 custom paged attention 的 FP8 KV-cache)、#34632(MXFP4 inline dequant Triton kernel)、#34631(Whisper causal attention backend 无关化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
