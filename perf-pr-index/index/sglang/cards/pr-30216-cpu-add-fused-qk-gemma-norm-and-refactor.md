# PR #30216: [CPU] add fused_qk_gemma_norm and refactor norm kernel implementation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mingfeima |
| 合并 | 2026-07-07 |
| 分类 | Attention / Kernel(major) |
| Label | sgl-kernel, intel, cpu, run-ci |
| Diff | pack 未提供 diffstat(additions/deletions/files 均缺失),diff 内容可用(pack 注明 5 个文件因超长被省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/30216 |

## 一句话结论

重构 sgl-kernel CPU norm 内核为统一框架并新增 `fused_qk_gemma_norm`,QK 归一化 bf16 场景最高 11.39x 加速。

## 动机

sgl-kernel 中的 CPU norm 内核(L2Norm、RMSNorm、Gemma 变体、LayerNorm、gated RMSNorm、fused add 路径)各自实现了大量重复循环,每个内核都重新实现一遍 reduce -> scale -> (weight/bias/gate) 逻辑,仅张量布局与 stride 处理略有不同。新增 Gemma / QK 风格 RMSNorm 变体的维护成本高,且缺少为 AVX512 优化的 BF16 快路径。

## 优化点

核心是把 `sgl-kernel/csrc/cpu/norm.cpp` 从十来份重复 kernel 重写为 trait 驱动的单一框架:`NormParams` 把 2D/3D/4D 输入统一视为逻辑 [B,H,T,D] 布局,输入按 stride 索引、输出连续;`NormMode` 枚举 L2Norm/RMSNorm/GemmaNorm/LayerNorm/RMSNormGated 五种模式,`NormTraits` 以编译期常量声明各模式是否含 weight/bias/shift/mean/gate 并提供共享的 apply_* helper。执行层分两条路径:`NormReduceGeneric` 是所有模式的通用向量化 reduce+apply 路径;`NormReduce<M, BFloat16, D>` 是 AVX512 BF16 专用快路径,当 head_dim D 属于 {32,64,128,256,512} 时走 AVX512-BF16 指令(如 `_mm512_dpbf16_ps` 点积)做 reduce。gated RMSNorm 的 SiLU 用 `exp_u20` 近似指数加 `_mm512_rcp14_ps` 倒数替代 vdivps。新增 `fused_qk_norm4d_kernel_impl` 将 Q/K 两路 Gemma norm 融合为一次调用(可选 copy gate),并在 `qwen3_5.py` 把 CPU 与 HIP/XPU 同等接入 `fused_qk_gemma_rmsnorm(_with_gate)`。公开的 `torch.ops.sgl_kernel.*_cpu` API 保持不变。

## 关键改动

```text
sgl-kernel/csrc/cpu/norm.cpp: NormParams/NormMode/NormTraits 统一框架,
  AVX512 BF16 fast path(D in {32,64,128,256,512}),SiLU 用 rcp14 近似,
  新增 fused_qk_norm4d_kernel_impl
python/sglang/srt/models/qwen3_5.py: CPU 启用 fused_qk_gemma_rmsnorm(_with_gate)
test/registered/cpu/test_norm.py: 重组并扩展覆盖
```

## 性能声明

PR 正文给出的 benchmark 条件:bf16,q_heads=16,kv_heads=4,head_dim=256(未注明 CPU 型号):

- batch=1:6.280 us(fused)vs 71.551 us(native),11.39x speedup。
- batch=4000:294.770 us(fused)vs 2,387.317 us(native),8.10x speedup。

## 验证与回归风险

`test/registered/cpu/test_norm.py` 覆盖 Gemma/Gemma3(含 4D QK-style 布局)、Gemma4(with_scale/scale_shift)、gated RMSNorm、LayerNorm,并用非标准 hidden_size(如 33)验证尾块处理。风险:该重构一次性迁移全部 CPU norm 算子(l2norm、rmsnorm、fused_add_rmsnorm、gemma/gemma3/gemma4 系列、layernorm、fused_add_layernorm、fused_rmsnorm_gated),任何模式 trait 声明错误都会影响 CPU 路径的数值正确性;LayerNorm 方差按 `E(X^2)-E(X)^2` 计算(注释注明参照 FlashInfer 实现),数值上与两遍法有细微差异;BF16 快路径仅覆盖特定 head_dim,其他维度走 generic 路径。仅影响 CPU 后端,不涉及 GPU/CUDA 路径。

## 关联

同区域:#19434([Qwen3-Next] gdn fused_rms_norm_gated)、#16397(layernorm-gated for Qwen3-Next)。前序:qwen3_5.py 中该 fused kernel 原仅 HIP/XPU 可用,本 PR 将其扩展到 CPU。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
