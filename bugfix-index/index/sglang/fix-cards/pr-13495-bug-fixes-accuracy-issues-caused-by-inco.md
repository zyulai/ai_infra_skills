# PR #13495: [Bug] Fixes accuracy issues caused by incorrect use of rope

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#13495](https://github.com/sgl-project/sglang/pull/13495) |
| 作者 | Baidu-AIAK |
| 合并日期 | 2025-11-18 |
| labels | high priority, run-ci |
| 热度 | 👍 2 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
nsa_indexer 的 RoPE 用了 is_neox_style=False 而 DeepSeek-V3.2 应为 neox 风格,长上下文检索精度大跌;改为 True 后 Ruler niah_multikey_3 各长度全部 100 分。

## 问题现象
Ruler 测试中 dsv3.2_exp 相比 dsv3.1-terminus 分数显著下降;专项测 niah_multikey_3 发现:DeepSeek 官方 API 上几乎 100% 正确,而 sglang + dsv3.2 得分很低——修复前 ruler_niah_multikey_3 得分:4k 100.00、8k 100.00、16k 87.50、32k 62.50、64k 25.00、96k 43.75、128k 0.00。修复后 4k-128k 全部 100.00。

## 根因
正文披露:rope 使用方式不正确。作者按 DeepSeek 的建议运行其随模型发布的推理 demo 仍复现问题;DeepSeek 修改推理 demo 后 bad cases 通过(对应 HF commit 1938e9df3dea7218cb36c21fe8287384b99acd96)。作者参照该修改,把 sglang `nsa_indexer.py` 中 rope 相关代码做了对应改动(is_neox_style 由 False 改为 True),bad cases 同样通过。

## 修复方案
`python/sglang/srt/layers/attention/nsa/nsa_indexer.py`:RoPE 构造参数 `is_neox_style=False` 改为 `is_neox_style=True`(一行)。

关键 diff:
```diff
             max_position=max_position_embeddings,
             base=rope_theta,  # type: ignore
             rope_scaling=rope_scaling,
-            is_neox_style=False,
+            is_neox_style=True,
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:DeepSeek-V3.2(NSA)注意力 indexer 的 RoPE 计算路径;`is_neox_style` 由 False 改 True 改变 indexer 的位置编码风格,经过该 indexer 的输入的注意力打分与检索行为随之改变。正文只验证了修复带来的提升,未评估其他依赖旧风格行为的模型/路径。

## 关联
- DeepSeek 官方修复 commit:https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/commit/1938e9df3dea7218cb36c21fe8287384b99acd96

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
