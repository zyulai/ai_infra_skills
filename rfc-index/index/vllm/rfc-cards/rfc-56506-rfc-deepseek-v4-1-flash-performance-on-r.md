# RFC #56506: [RFC]: DeepSeek-V4.1-Flash performance on ROCm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56506](https://github.com/vllm-project/vllm/issues/56506) |
| 作者 | Fangzhou-Ai |
| 创建日期 | 2026-09-11 |
| 状态 | ⚪ closed (2026-09-30) |
| labels | DSv4.1, RFC, deepseek, performance, quantization, rocm |
| 热度 | 👍 0 · 💬 12 |
| 分类 | Hardware / Backend |

## 一句话结论
系统定位并修复 DeepSeek-V4.1-Flash 在 AMD MI355X 上的解码开销:mHC 融合、流重叠、GEMM/MoE 调优与 indexer 优化。

## 提案动机
8x MI355X (gfx950)、TP4、MXFP4 MoE + DSpark MTP 实测:并发 1→32 时 ITL 仅劣化 1.9x(0.0256→0.0477 ms)而单 GPU 输出增 13x,说明每解码步存在巨大固定成本;decode trace 显示单流 ITL 25.6 ms 时每步发起 14,020 次 kernel launch,每层 350 次中 85% 归于 mHC block(对 4x4 矩阵做 bookkeeping)。作者强调纯性能问题,精度已达标(gsm8k strict-match baseline 0.9704±0.0047,harness gate 0.91)。

## 提出的方案
- **mHC seam fusion**(85% 启动所在):mHC pre→AITER 已并入 #56503(每 seam 141→4 launches,最坏相对误差 1.2e-5,gsm8k 不变);post+pre 融合由 #56513(draft)跟进;AR+mHC post 融合与 1.2 互斥而非叠加,作者选自包含的 1.2;平行 TileLang 方案 #56342 需先协调。
- **Multi-stream overlap**:3 条 aux 流的重叠在 breakable CUDA graph capture 时退化为串行,decode 路径全部失效;需让重叠存活于 capture 并确认 gfx950 不复现 V4 AMD 的 hang。MoE shared expert 因 TP8/几何(hidden 7168 vs 5120)/FP8 scale 形状门槛,融合与流重叠两路都不通,aux-stream 更省。
- **GEMM autotuning**:indexer/compressor 形状每步约 1,220 次 tuned-config miss、落回 `F.linear`;收集后把 gfx950 行 upstream 到 aiter/configs。
- **MoE backend**:A/B `auto`/`aiter` 与强制的 `aiter_triton_mxfp4_bf16`(仅启发式配置);给 `benchmark_moe.py` 加 MXFP4 模式;upstream gfx950 a16w4 配置(与 #56543 主张相反,需协调)。
- **Prefill top-k combine** 的 Torch 回退由 #56638 修复(该回退因 `repeat_interleave` 同步主机而无法被 graph 捕获)。
- **DSA sparse indexer**(131k 上下文下占步时 4.1%→11.4%,conc 1→16):candidate mask 按整个 workspace 而非上下文分配大小(#56628 修复);`next_n` 花在 program 数而非 GEMM tile(报给 ROCm/aiter#5477)。

## 优化点与收益
正文照抄的数字(条件:8x MI355X (gfx950),TP4,MXFP4 MoE + DSpark MTP):
- 每 decode step **14,020 次 kernel launch**;每层 350 次中 mHC sinkhorn loop 占 162(48.6%)、pre/post mix + eager RMSNorm 占 122(36.6%)。trace 须 `--enforce-eager` 捕获,仅 launch 计数与内核构成可作依据,wall-clock 数字被刻意弃用。
- #56503:**141→4 launches per seam**;并发 1 下 ITL p50 25.58→6.89 ms(作者注明仅方向性,非 3.7x 每 token 提速)。
- #56513:mean ITL 降 **1.28%**,95% CI [0.07%, 2.48%](固定 24 请求 replay、双向交叉)。
- #56628:candidate mask 内核提速 **2.7x/4.3x/6.0x**(conc 1/4/16),gsm8k 0.9727/0.9719。
- M-scaling:M=64 较 M=32 每单位功效率高 **1.27x**,M=128 高 **1.53x**(当前处于曲线最差点)。

## 现状与落地
已关闭(2026-09-30)。#56503 已合并,#56513/#56638/#56628 已推进,#56342/#56543 待协调;其余(MoE A/B、GEMM 调优 upstream 等)为提案。注:pack 正文在 20k 字符处截断(截于第 6 节 DSA sparse indexer 处),本卡基于现有内容成卡。

## 关联
- [PR #56503](https://github.com/vllm-project/vllm/issues/56503)(mHC pre→AITER,已合并)
- [#56513](https://github.com/vllm-project/vllm/issues/56513)(fused mHC post+pre,draft)
- [#56628](https://github.com/vllm-project/vllm/issues/56628)(candidate mask 修复)
- [#56638](https://github.com/vllm-project/vllm/issues/56638)(top-k combine 内核修复)
- [ROCm/aiter#5477](https://github.com/ROCm/aiter/issues/5477)(MQA logits GEMM M-scaling)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
