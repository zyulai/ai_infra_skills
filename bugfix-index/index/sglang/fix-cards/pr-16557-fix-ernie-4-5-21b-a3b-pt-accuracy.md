# PR #16557: fix ERNIE-4.5-21B-A3B-PT accuracy

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#16557](https://github.com/sgl-project/sglang/pull/16557) |
| 作者 | syy-hw |
| 合并日期 | 2026-01-06 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
ERNIE-4.5-21B-A3B-PT 精度问题:给 attention 层加 force_native_attention_path 开关强制走 native 前向路径,GSM8K 达 0.905。

## 问题现象
ERNIE-4.5-21B-A3B-PT 在 SGLang 下输出精度不对(标题:"fix ERNIE-4.5-21B-A3B-PT accuracy")。正文 Motivation/Modifications 节为空,未描述修复前的具体分数、报错或复现步骤,仅给出修复后的验证结果:`python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --data-path /test.jsonl` 得 Accuracy 0.905、Invalid 0.000、Latency 44.730 s、Output throughput 803.599 token/s。

## 根因
正文未披露根因。(由 diff 推断)ERNIE-4.5 的 attention 层此前未加 `force_native_attention_path`,在 NPU 上当 rotary_emb 带 `get_cos_sin_with_position` 方法时会走 fused attention 前向(非 `forward_prepare_native`),该路径对该模型产生了不正确的输出;修复通过新增开关强制该模型回到 native attention 前向路径。

## 修复方案
两个文件:

- `python/sglang/srt/models/llama.py`:`LlamaAttentionBase.__init__` 新增参数 `force_native_attention_path: bool = False` 并存为实例属性;`forward` 中走 native 路径的条件由 `not _is_npu or not hasattr(self.rotary_emb, 'get_cos_sin_with_position')` 扩为再 `or self.force_native_attention_path`:

```diff
-        if not _is_npu or not hasattr(self.rotary_emb, 'get_cos_sin_with_position'):
+        if not _is_npu or not hasattr(self.rotary_emb, 'get_cos_sin_with_position') or self.force_native_attention_path:
             q, k, v = self.forward_prepare_native(
```

- `python/sglang/srt/models/ernie4.py`:构造 attention 层时传入 `force_native_attention_path=True`,使 ERNIE-4.5 始终走 native attention 前向。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:`llama.py` 的 attention 基类新增参数默认 False,其余基于该类的模型行为不变;ERNIE-4.5(ernie4.py)在 NPU 上不再使用 fused rotary/attention 快路径,精度恢复但该平台可能损失相应加速。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
