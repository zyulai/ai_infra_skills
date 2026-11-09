# PR #38928: [Bugfix][Perf] Indexer upcast WK to BF16 for fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2026-04-15 |
| 分类 | Attention / Kernel \| tier: major |
| Label | bug, performance, ready, deepseek |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/38928 |

## 一句话结论

Indexer WK 加载时 upcast 到 BF16 保住 fusion，B200 TP8 FP8 下 Decode 11.90 ms。

## 动机

PR 是 #38870 的替代修复（alternative fix），目的是在修复的同时保住（maintain）fusion。正文把该方案与 multi-stream 方案及分离方案放在同一配置下做了对比，但未进一步详述更早的问题背景。

## 优化点

在权重加载阶段把 FP8 的 indexer wk 权重 upcast 成 BF16，从而让 `wk` 与 `weights_proj` 继续合并为单个 `MergedColumnParallelLinear`（`wk_weights_proj`，一次 GEMM 产出 [head_dim + n_head]，`quant_config=None`）。新增 `_try_load_fp8_indexer_wk` / `_pending_wk_fp8` 去量化缓冲，并从 `quant_utils` 引入 `GroupShape`、`scaled_dequantize` 在 `load_weights` 中处理 FP8 wk；删除原先仅对 `is_fp4_ckpt`（modelopt_fp4）才启用融合的分支，改为无条件融合。MTP 模型（deepseek_mtp.py）同步接入同一 helper。

## 关键改动

```text
vllm/model_executor/models/deepseek_v2.py: 新增 _try_load_fp8_indexer_wk、把 FP8 wk upcast 到 BF16 并统一用 wk_weights_proj 融合
vllm/model_executor/models/deepseek_mtp.py: 引入同一 helper、删除 is_fp4_ckpt 分支、加入 _pending_wk_fp8
```

```python
+        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
+        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
+        self.wk_weights_proj = MergedColumnParallelLinear(
+            hidden_size,
+            [self.head_dim, self.n_head],
+            bias=False,
+            quant_config=None,
+            disable_tp=True,
+            prefix=f"{prefix}.wk_weights_proj",
+        )
```

## 性能声明

B200、TP8、FP8、BS1、8k/1k；PR 未注明模型名称。对比 this PR（Upcast+Fused WK）、multi-stream（#35968）、baseline（#38870）：Upcast+Fused WK Decode 11.90 ms / TTFT 375.0 ms；Multi-Stream Decode 12.74 ms / TTFT 376.5 ms；Separate Decode 12.59 ms / TTFT 378.2 ms。

## 验证与回归风险

作者说明自身环境损坏，即使使用 #38870 时 GSM8k 也给出 0.00，暂无法给出可信精度结果、只有中等信心；建议他人复跑，并表示如需周末紧急修复可接受先合并 #38870。回归风险：精度验证未完成（GSM8K 不可用）；改动涉及 DeepSeek-V2 indexer 权重加载与 MTP 路径，且把融合从 FP4 checkpoint 扩展到所有量化配置。

## 关联

正文明确列出替代修复对象 #38870、对比方案 multi-stream #35968；无其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
