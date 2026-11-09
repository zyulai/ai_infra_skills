# PR #41880: [Perf] [AMD] Qwen3.8: AITER FlyDSL hc_mix for Qwen3.8 hyper-connections on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-nprotaso |
| 合并 | 2026-10-30 |
| 分类 | Attention / Kernel(tier: major) |
| Label | amd, run-ci, amd-aiter-not-ready |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/41880 |

## 一句话结论
gfx950 上用 AITER FlyDSL `hc_mix` 替换 Qwen3.8 超连接的 Triton `fused_hc_mix`,decode 尺寸 kernel 提速 1.73–2.38x。

## 动机
`GatedResidual` 的 input mix 是对 `[M, 4·2560]` hyper-state 的低秩 gate,每个 Qwen3.8 decode step 都会执行;decode 尺寸(M ≤ 16)下受权重读取带宽限制。在 gfx950 上,AITER 的 FlyDSL `hc_mix` 比当前 SGLang 使用的 persistent Triton `fused_hc_mix` 快 1.7–2.4x。

## 优化点
新增 `hc_mix_flydsl.py` 门控:`flydsl_hc_mix_weights_supported`(每层评估一次:HIP、gfx950、BF16、K 按 64/R 按 32 对齐、hc=4、AITER/FlyDSL 可导入)与 `flydsl_hc_mix_supported`(每次调用:1 ≤ M ≤ 128、contiguous、16 字节对齐);检查顺序保证 NVIDIA 与禁用路径绝不 import FlyDSL。`hyperconnection.py` 的 `GatedResidual.mix()` 在 Triton `fused_hc_mix` 之前尝试 FlyDSL 路径:权重经 `pack_hc_weights` 打包一次并缓存在层上,`weight_cache_key`(data_ptr + version,inference tensor 无 version)变化时才重打包。依赖 ROCm/aiter#5993,不可导入时路径完全不激活(本 PR 不改变任何行为)。

## 关键改动

```text
python/sglang/srt/layers/hc_mix_flydsl.py: 新增门控函数与 weight_cache_key
python/sglang/srt/layers/hyperconnection.py: GatedResidual.mix() 插入 FlyDSL 分支(优先于 fused_hc_mix)与打包权重缓存
test/registered/kernels/ops/gemm/test_hc_mix_flydsl.py: dispatch/门控/缓存失效测试(AMD stage-b 注册)
benchmark/kernels/bench_hc_mix_flydsl.py: FlyDSL vs Triton vs torch.compile 对比(CUDA-graph replay、旋转权重冷 L2)
```

## 性能声明
MI355X (gfx950)、torch 2.11.0+rocm7.2、CUDA-graph replay 每 call 中位数、旋转权重(权重冷 L2)、`--rows 1 2 4 8 16 --weights 64`:M=1:FlyDSL 18.1us vs Triton 36.3us(2.00x)、torch.compile 34.3us;M=2:18.8 vs 44.9(2.38x);M=4:20.3 vs 44.9(2.21x);M=8:22.1 vs 45.7(2.07x);M=16:26.3 vs 45.5(1.73x)。`_MAX_ROWS=128` 内 FlyDSL 保持领先(128 rows 冷权重 26 vs 42us)。

## 验证与回归风险
测试(AMD `stage-b-test-1-gpu-small-amd`)覆盖:门控在禁用/不支持 shape 时拒绝;`GatedResidual` dispatch 到 FlyDSL 且与 float64 参考对齐(atol=4e-3, rtol=1e-2,AITER op 测试与本 PR dispatch 测试均满足);打包权重缓存复用与权重变更后重打包;inference tensor 的 key 无 version。风险:label 含 `amd-aiter-not-ready`,硬依赖 ROCm/aiter#5993 的构建;打包权重缓存的失效依赖 data_ptr/version,推理 tensor 的原地更新需重初始化缓存(与 CuTe weight cache 同一约定)。

## 关联
依赖 ROCm/aiter#5993;同类 FlyDSL 融合见 #33838(Kimi-K3 SiTU MoE,依赖 FlyDSL 0.3.0)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
