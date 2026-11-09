# PR #65087: [ROCm][Perf][MiniMax-M3.1] Four-wave decode top-k for the NVFP4 indexer

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chao-xu-spec |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, minimax |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 3 个文件(topk host、prefill_kernel 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/65087 |

## 一句话结论

MiniMax-M3.1 NVFP4 indexer 的 decode top-k 由单波改四波,gfx950 上提速 4.1x。

## 动机

PR 正文 Overview:在 gfx950 上,MiniMax-M3.1 的 NVFP4 indexer 对每条 decode row 用「每 row 一个 wave」选出 16 个最佳 block。Details 说明旧 kernel 单 wave 扫描 1,664 个 block(213k tokens)的窗口,超长 row 会在这个单 wave 上串行化;且真实 index score 落在很窄区间,12-bit bound 让约三分之一 row 的候选超过 64 个(emulated 33%),触发候选重建与精确位搜索,一次调用耗时取决于最慢的 row。

## 优化点

给每条 decode row 四个 wave:wave w 只走 w、w+4… 这些窗口(每窗口 1,024 blocks),各带自己的候选与运行 top 16,四份列表再按 (score, block id) 精确 rank 合并,复现 AITER 的顺序(含 ties、NaN、inf),有序 block id 不变。无 live window 的 wave 用死条目补齐,至多一个窗口的 row 跳过合并;lane-bound 阈值从 12 位改为 16 位,过多 key 时回退全 32 位。decode kernel 名从 `..._w26` 变为 `..._w16`,prefill 编译结果与之前相同,不改其他 kernel、flag 或调用点。

## 关键改动

```text
vllm/models/minimax_m3/amd/ops/index_topk_decode/prefill_kernel.py: decode 引入 WAVES=4、NTHR=64*WAVES,四波各自的候选/选中列表按 (key,index) 精确 rank 合并
vllm/models/minimax_m3/amd/ops/index_topk_decode/host.py: _get_decode_kernel 从 compile_topk_prefill(26) 改为 (16),decode 阈值断言同步改为 strips==16
tests/kernels/attention/test_minimax_m3_index_topk_decode.py: 新增 3 个 decode shape(100 requests 与两个 wave 交界长度)
```

关键机制代码(四波与线程数):

```python
-    assert strips == 26 if decode else strips in PREFILL_STRIPS
+    assert strips == 16 if decode else strips in PREFILL_STRIPS
     STRIPS = strips
+    WAVES = 4 if decode else 1
+    NTHR = 64 * WAVES
     WIN = STRIPS * 64
...
-        cand: fx.Array[fx.Int32, 2 * CAND, 16]
-        sel: fx.Array[fx.Int32, 4 * TOPK, 16]
+        cand: fx.Array[fx.Int32, WAVES * 2 * CAND, 16]
+        sel: fx.Array[fx.Int32, WAVES * 4 * TOPK, 16]
```

## 性能声明

PR 正文 Claims/Microbench:MI355X(gfx950)、ROCm 10.0.0、torch 2.13.0+rocm10.0.0、真实 index scores、按调用频率加权,decode top-k 18.9 → 4.65 us/call(4.06x,正文概述称 4.1x faster),live server 21.2 → 5.2 us/call(4.07x);相同 shape 随机分数仅 1.26x。真实分数 grid(64k~512k、1~100 requests)里 requests=4/512k 为 35.3 → 4.68 us(7.6x)。E2E(amd/MiniMax-M3.1-Flash-Quark-MXFP4,TP4 EP1,4× MI355X,DSpark 3 draft tokens,DBO,max-model-len 1M,2 runs/arm):c16 吞吐 168.70 → 170.95 tok/s/GPU(+1.34%)、TPOT p50 5.285 → 4.640 ms(−12.2%)、p90 −8.3%;c32 304.06 → 311.57(+2.47%)、TPOT p50 −12.0%、p90 −8.2%;interactivity p50 +13.9%/+13.6%。GSM8K(1319 题 5-shot)0.9689 → 0.9687,spec-decode accept length 3.116 → 3.123,长上下文 98/98 正确。代价:随机分数且每行都短于约 213k tokens 时最多慢 12%。

## 验证与回归风险

Validation 含:仓库测试 221 passed;4,500 sweep cases / 772,650 rows 与 AITER `pa_sparse_block_topk` 及旧 kernel bit-for-bit 一致(0 differing rows),eager 与 CUDA-graph replay 0 mismatches;真实 served 回放 2,816 decode rows 仅 1 行不同(duplicate page ids,旧 kernel 自身输出也不稳定)。影响路径:gfx950 上 MiniMax-M3.1 的 NVFP4 indexer decode top-k。Limits 明确:仅 gfx950 验证、无 Buildkite step 跑该测试、microbench 不在 benchmarks/kernels、E2E 只跑 c16/c32 单一模型且基于 CI 镜像 overlay 两个文件而非 main 构建。

## 关联

正文提及的非重复 PR:#64198(MXFP4 K/V 与 index cache,不动 top-k 文件)、#64264(NVFP4 kernel 启动 warmup)、#64273(M3.1 monokernel decode,后两者为已关闭 draft)、#64448(DSpark 7 的 decode CP,互补);结果对齐目标为 AITER `pa_sparse_block_topk`。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
