# PR #2688: Fixes assertion failure in prefix caching: the lora index mapping should respect `prefix_len`.

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#2688](https://github.com/vllm-project/vllm/pull/2688) |
| 作者 | sighingnow |
| 合并日期 | 2024-01-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | KV Cache / Memory |
| bug 类型 | crash |

## 一句话结论
prefix caching 复用前缀时 LoRA index 映射仍按全 prompt 长度构建,触发断言失败;改为按 prompt_len - prefix_len。

## 问题现象
启用 prefix caching 且使用 LoRA 时出现断言失败(assertion failure)。正文仅写「Address issue #2612」,未描述具体报错信息、日志与影响面;触发条件按标题为:命中前缀缓存(prefix_len > 0)的 LoRA prompt。

## 根因
正文未披露根因分析(正文只有一句「Address issue #2612」)。从 diff 看机制:`_prepare_prompt` 里 `lora_index_mapping` 与 `lora_prompt_mapping` 此前均按完整 `prompt_len` 填充 LoRA id,但开启 prefix caching 后实际参与本次前向的 token 数是 `prompt_len - prefix_len`(前缀部分由缓存复用),映射与实际 token 数不对齐,进而在下游触发断言失败(由 diff 推断)。

## 修复方案
修改 `vllm/worker/model_runner.py` 的 `_prepare_prompt`,两处映射均改按去掉缓存前缀后的长度构建:

```diff
-            lora_index_mapping.append([lora_id] * prompt_len)
+            lora_index_mapping.append([lora_id] * (prompt_len - prefix_len))
             lora_prompt_mapping.extend(
                 [lora_id] *
-                (prompt_len
+                (prompt_len - prefix_len
                  if seq_group_metadata.sampling_params.prompt_logprobs else 1))
```

即 `lora_index_mapping` 每个序列只对非缓存部分的 token 填 LoRA id;`lora_prompt_mapping` 在请求 prompt_logprobs 时同样按 `prompt_len - prefix_len` 展开。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:改动仅位于 `_prepare_prompt` 的 LoRA 映射构建,影响所有 prompt 命中 prefix caching(`prefix_len > 0`)的 LoRA 请求;`prefix_len == 0` 时两式等价,行为不变。`prompt_logprobs` 分支的映射长度同步缩短,请求 prompt logprobs 且命中前缀缓存的 LoRA 场景返回的映射也随之对齐。

## 关联
- vllm-project/vllm#2612(正文所述要解决的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
