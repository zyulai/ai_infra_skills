# PR #48745: [Fix] kda: Align unfused safe-gate diagonal normalization anchor

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @wirybeaver |
| 合并 | 2026-11-17 |
| 分类 | Other (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录仅含 1 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/48745 |

## 一句话结论

把 KDA safe-gate 对角 kernel 的 unfused 路径归一化锚点从 16-token 子块中点改为子块起点,与 fused 路径对齐,消除"并发请求打包改变 prefill batch 组成 → 网格启发式切换 fused/unfused 路径 → 序列输出翻转"的数值问题。

## 动机

KDA 的 safe-gate diagonal kernels 使用不同的归一化锚点:fused 路径用每个 16-token 子块的首 token,unfused 路径用其中点。锚点在代数上可消,但改变有限精度舍入;网格尺寸启发式因此可以在其他序列加入某序列的 prefill batch 时改变它的输出。KDA 仅在 `B * NT * H <= 256` 时选 fused diagonal/recompute kernel,更大网格走 unfused 以避免 register-pressure spills;对 packed 变长 prefill,`B=1` 且 `NT=sum(ceil(sequence_length / 64))`,触发条件取决于总 prefill chunk 数与头数而非请求数:在捕获的 Ming ASR 案例(`H=16`)中,单个请求用 2-3 chunks(grid 32-48,fused),打包 8 个请求产生 21 chunks(grid 336,unfused)。并发请求因此会在打包成更大 prefill batch 时跨过该阈值;async decode 改变调度与分组,同样的切换也发生在同步基线上。问题来源于 [sglang-omni#3564](https://github.com/sgl-project/sglang-omni/pull/3564) 的 Ming-3.0-ASR-tiny 转写差异:相同首层输入与零初态下,只切换 diagonal 融合即可复现输出差异,只切换 recompute 融合则不能。

## 优化点

在 unfused safe-gate diagonal kernel 的 gather 与 direct-load 两个变体中统一改用子块起点 gate,匹配既有 fused 路径;保留融合启发式与 batched prefill 不变。本改动与 #47413(batch-invariant 模式下选择固定路径)互补:它只在 normal 模式执行切换路径时对齐锚点,不声称普遍的 bitwise batch invariance。

## 关键改动

```diff
python/sglang/kernels/ops/attention/fla/chunk_intra.py(唯一改动文件,chunk_kda_fwd_kernel_intra_sub_chunk):
+    # Match the subchunk-start anchor used by the fused diagonal path.
     if USE_GATHER:
-        b_gn = gather(
-            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
-        )
+        b_gn = gather(b_g, tl.full([1, BK], 0, dtype=tl.int16), axis=0)
     else:
         # calculate offset
-        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
+        p_gn = g + i_ti * H * K + tl.arange(0, BK)
         b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
         b_gn = b_gn[None, :]
-    # current block, keep numerical stability by subtracting the left boundary
-    # less than 85 to avoid overflow in exp2
     b_gm = (b_g - b_gn).to(tl.float32)
```

## 性能声明

本 PR 无针对修复本身的独立性能对比。正文引用 sglang-omni#3564 的基准(两个兼容补丁的 Omni backports,正文明确"This comparison does not isolate either upstream fix",即该对比不隔离本修复的单独收益):单 H200,BF16,SGLang 0.5.23,PyTorch 2.14.1+cu130,SeedTTS EN,c16,每次 timed visit 相同的 256 samples,顺序 A1 → B1 → C1 → C2 → B2 → A2,每次访问重启 server。Normal 模式吞吐:A1 40.06 / B1 37.57 / C1 44.63 / C2 42.83 / B2 39.01 / A2 38.98 req/s(median latency 0.393/0.413/0.344/0.360/0.409/0.393 s);Deterministic 模式:A1 34.98 / B1 33.11 / C1 39.67 / C2 37.65 / B2 32.18 / A2 33.39 req/s。"Median throughput improvement from async decode (C/B): **+14.22%** in normal mode; **+18.41%** in deterministic mode."(即 async decode 相对 sync 的收益,而非本修复的收益;A 为 Omni base,B 为 middle commit 两补丁 + 同步 decode,C 为 head commit 两补丁 + async decode)。WER:normal 各 visit 7.11%-9.10%;deterministic B1/B2/C1/C2 的 256 samples 原始转写完全一致(WER 7.17272%)。

## 验证与回归风险

diff 未新增单元测试,正文亦无 Accuracy Tests 小节内容;复现依据是 sglang-omni#3564 中"相同首层输入下只切 diagonal 融合即可复现差异"的实验。CI 状态:Base 失败(x),Extra 失败(x),AMD 通过。回归风险:(1) 修复只覆盖 `chunk_kda_fwd_kernel_intra_sub_chunk` 一处锚点,gather 与 direct-load 两变体均改,但其他 kernel 是否存在同类中点锚点正文未说明;(2) 数值结果有意与旧版不一致,依赖逐位对比的下游测试需更新基线;(3) 正文明言不声称普遍 bitwise batch invariance——normal 模式下 fused/unfused 的 recompute 融合切换仍可能带来差异;(4) 与 #47413 的交互:batch-invariant 模式已固定路径,本修复作用于 normal 模式。

## 关联

#47413(batch-invariant 模式下选择固定路径,互补);sglang-omni#3564(问题来源与基准)。同批次 KDA 相关:PR #50842(KDA DWDP 权重分片)、PR #38344(K3 prefill CP)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
