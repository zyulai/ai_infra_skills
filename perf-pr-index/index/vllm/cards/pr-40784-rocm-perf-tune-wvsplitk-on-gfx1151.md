# PR #40784: [ROCm][Perf] Tune wvSplitK on gfx1151

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-06-25 |
| 分类 | Other（tier: major） |
| Label | rocm, ready |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 仅改动 `csrc/rocm/skinny_gemms.cu` 一个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/40784 |

## 一句话结论

为 gfx1151（AMD Strix Halo，20 CUs）定制 wvSplitK skinny GEMM 的 launch 配置启发式：kernel 带宽 bf16 N=1 平均 +18.7%，decode-bound 模型 E2E TPOT 最高 −9.2%。

## 动机

wvSplitK 是 decode 阶段小 N（batch 1–5）skinny GEMM 的关键 kernel；此前 gfx1151 只能落入通用的 `on_gfx1x()`（gfx1100/gfx1150/GFX12）wave32 配置分支，其 YTILE/UNRL 启发式并非为 Strix Halo 的 20 CU + LPDDR5X 内存体系调过，decode-bound 的小模型在 skinny GEMM 上带宽利用率不足。需要为该 SoC 单独选择 THRDS/WVPRGRP/YTILE/UNRL 组合。

## 优化点

在 `csrc/rocm/skinny_gemms.cu` 新增 `on_gfx1151()` 设备判定与 `WVSPLIT_TILE` 宏：gfx1151 命中专用决策树（THRDS=32、WVPRGRP=16 固定），按 `_sYT`（每 CU 输出行数）、K 的规模与对齐（`K%1024==512 && K>=1536` 且 YTILE≥40 或 K≥4096 → YTILE=4/UNRL=1；K<1024 → YTILE=2/UNRL=4；K≤2048 且 N≥2 或 YTILE≤26 → YTILE=1/UNRL=4；N≥2 且 B 放不进 LDS（`fit_lds = Kbp_in * N_in <= max_lds_len`）→ YTILE=1/UNRL=4；N==1 → UNRL=2；其余 → UNRL=1）选择配置。gfx1100/gfx1150/GFX12 继续走原 `WVSPLIT_TILE_CFG(32,...)` wave32 分支，GFX9 保持 wave64（THRDS=64），其余架构行为不变；N=1–5 的 switch 分支统一改经该宏分发。

## 关键改动

```text
csrc/rocm/skinny_gemms.cu: on_gfx1151() 判定 + WVSPLIT_TILE 宏（gfx1151 专用 YTILE/UNRL 决策树），N=1..5 分支改走该宏
```

```cpp
#define WVSPLIT_TILE(_sYT, __N) {
  if (on_gfx1151()) {
    bool fit_lds = (Kbp_in * N_in <= max_lds_len);
    if (_sYT <= 1)        WVSPLITK_CFG(32, 16, 1, 4, __N)
    else if ((K_in % 1024 == 512) && K_in >= 1536 &&
             (_sYT >= 40 || K_in >= 4096))
                          WVSPLITK_CFG(32, 16, 4, 1, __N)
    ...
```

## 性能声明

Kernel 级（bf16 wvSplitK，Strix Halo gfx1151、20 CUs，带宽提升均值）：N=1 +18.7%、N=2 +19.8%、N=3 +15.2%、N=4 +13.0%；fp16 N=1 +6.1%。E2E（AMD Strix Halo gfx1151、LPDDR5X-8000 128GB，input-len=128、output-len=128、num-prompts=5、max-num-seqs=1、关闭 prefix caching、temperature 0，Median TPOT）：Qwen2.5-0.5B-Instruct 5.18→5.18ms（0.0%）；Qwen3-4B 37.32→36.85ms（−1.3%）；Qwen3-8B 77.16→70.09ms（−9.2%）；gemma-3-4b-it 38.82→36.63ms（−5.6%）。作者结论：0.5B 模型无可测变化，decode-bound 于 skinny GEMM 的较大模型 TPOT 显著改善。

## 验证与回归风险

测试：kernel benchmark + 端到端 `vllm bench serve`（PR 给出完整复现命令）；diff 未含新增单测。风险面：改动只影响 gfx1151 的配置选择（其他 ROCM/GFX 架构分支保持原样），风险集中在gfx1151 上不同 K/N 组合的回归覆盖——决策树依赖 K 的对齐模式与 LDS 容量判断，未在 PR 中列出跨形状的系统扫描结果；不涉及非 ROCm 平台。

## 关联

前序：#34709（wvSplitK skinny GEMM kernel 启用）。索引内同区域：#53283（wvSplitK single-output 路径）、#50578（RDNA decode asm）、#60644（RDNA w4a16 prefill）。PR 正文未列其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
