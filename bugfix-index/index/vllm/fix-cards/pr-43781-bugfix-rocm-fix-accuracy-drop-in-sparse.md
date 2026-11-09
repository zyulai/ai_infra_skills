# PR #43781: [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#43781](https://github.com/vllm-project/vllm/pull/43781) |
| 作者 | kliuae |
| 合并日期 | 2026-05-28 |
| labels | bug, deepseek, ready, rocm, v1 |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
ROCm sparse indexer 为支持 DeepSeek-V4 所做的配置在 gfx950 上令 DeepSeek-V3.2/GLM-5 精度崩塌:cache 布局硬编码 SHUFFLE、RoPE 快路径误用于非 in-place 实现;分别修复。

## 问题现象
ROCm 的 sparse indexer 路径新增 DeepSeek-V4 支持后,其配置使 gfx950 路径上的旧模型(DeepSeek-V3.2、GLM-5)精度严重下降。Test Plan:MI355X 上用 `vllm serve zai-org/GLM-5.1-FP8 --tensor-parallel-size 8 ...` 起 server,lm_eval 跑 gsm8k(20-shot,64 并发)。修复前 main:flexible-extract exact_match 0.0129±0.0031、strict-match 0.0023±0.0013(几乎全错);修复后 PR:flexible 0.9507±0.0060、strict 0.9515±0.0059;PR 且 `custom_ops +rotary_embedding` 时 0.9477/0.9484。

## 根因
正文明确列出两个问题:
1. indexer cache 布局被硬编码为 `SHUFFLE`,当 indexer cache 的 `block_size=1` 时会引起错误的 cache 读取(erroneous cache reads);
2. ROCm 上 sparse indexer 有一条省略 tensor split/cat/copy 的 RoPE 快路径,它依赖 rotary embedding 实现**原地(in-place)修改输入参数**;但当 TorchInductor 开启时默认使用 pytorch-native rotary embedding,该实现并不原地修改而是返回新张量,快路径拿到的是未旋转的 q/k,导致精度错误。

## 修复方案
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`:`indexer_k_quant_and_cache_triton` 与 `cp_gather_indexer_k_quant_cache_triton` 中布局不再硬编码,改为 `layout = "NORMAL" if block_size == 1 else "SHUFFLE"` 传给 triton kernel;`rocm_fp8_paged_mqa_logits` 的 `Preshuffle` 条件由 `block_size == 64` 放宽为 `block_size > 1`。
- `vllm/model_executor/models/deepseek_v2.py`:indexer 构造函数新增 `is_inplace_rope` 参数(传入 `self.indexer_rope_emb.enabled()`),快路径条件由 `current_platform.is_rocm()` 收紧为 `current_platform.is_rocm() and self.is_inplace_rope`,并加注释说明快路径仅对作为 custom ops 的原地 RoPE kernel 有效。

```diff
-        if current_platform.is_rocm():
+        if current_platform.is_rocm() and self.is_inplace_rope:
...
+    layout = "NORMAL" if block_size == 1 else "SHUFFLE"
...
-                Preshuffle=block_size == 64,
+                Preshuffle=block_size > 1,
```

## 影响与回归风险
- 影响窗口:正文指出问题随 ROCm sparse indexer 的 DeepSeek-V4 支持配置引入;具体版本号未提及。
- 回归面:`deepseek_v2.py` 的 sparse indexer RoPE 快路径在非 in-place RoPE(custom op 未启用,如 inductor 融合的 pytorch-native rope)时退回通用路径,可能增加 split/cat/copy 开销;`rocm_aiter_mla_sparse.py` 的 cache 布局分发影响 gfx950 上不同 block_size 的 indexer cache 读写,block_size=1 走 NORMAL、其余仍走 SHUFFLE,DeepSeek-V4 支持路径需确认兼容。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
