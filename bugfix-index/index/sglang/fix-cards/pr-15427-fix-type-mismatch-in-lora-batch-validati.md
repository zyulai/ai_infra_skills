# PR #15427: Fix type mismatch in LoRA batch validation causing assertion failures

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#15427](https://github.com/sgl-project/sglang/pull/15427) |
| 作者 | ConnorLi96 |
| 合并日期 | 2025-12-21 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

`can_run_lora_batch` 把 list 直接传给期望 set 的 `validate_lora_batch`,多 LoRA 高并发合批时校验失败,现补类型转换。

## 问题现象

高并发多 LoRA 推理时可能触发断言失败(assertion failures)。触发场景:使用多个 LoRA adapter 跑基准(如 `--lora-name adapter1 adapter2 ... adapter16`)且 LoRA pool 容量小于 adapter 数量时,scheduler 尝试把使用不同 LoRA adapter 的请求合并进同一 batch,此时批校验逻辑可能失败。

## 根因

正文披露:`can_run_lora_batch` 方法持有的 `lora_ids` 是 `list[str]`,却直接传给期望 `set[str]` 的 `validate_lora_batch`,类型不匹配导致校验逻辑在合批判定时潜在失败。

## 修复方案

`python/sglang/srt/managers/tp_worker.py` 的 `can_run_lora_batch`:传入前先把 `lora_ids` 转为 set(已是 set 则原样传递):

```diff
     def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
-        return self.model_runner.lora_manager.validate_lora_batch(lora_ids)
+        lora_ids_set = set(lora_ids) if isinstance(lora_ids, list) else lora_ids
+        return self.model_runner.lora_manager.validate_lora_batch(lora_ids_set)
```

正文未提供修复后的复测数据(Accuracy Tests / Benchmarking 均为 None)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:仅 `tp_worker.py` 的 `can_run_lora_batch` 单点改动,单行语义为「list 转 set 后再校验」;set 去重不改变「batch 内不同 adapter 数量」的判定语义(由 diff 推断)。不影响 LoRA 加载/卸载与模型前向路径。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
