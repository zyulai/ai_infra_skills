# PR #2945: Fix qwen accuracy issue

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2945](https://github.com/sgl-project/sglang/pull/2945) |
| 作者 | ispobock |
| 合并日期 | 2025-01-17 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论

#2910 让 qwen2 加载权重时跳过 `lm_head.weight`,导致 Qwen 系列 GSM8K 准确率跌到 0;本 PR revert 该改动恢复加载。

## 问题现象

`neuralmagic/Qwen2-72B-Instruct-FP8` 的 nightly eval 失败(见 CI run 12820064955)。作者实测其他 Qwen 模型同样复现,如 `Qwen/Qwen2-7B` 与 `Qwen/Qwen1.5-7B-Chat-AWQ`。触发条件:启动 Qwen 系列模型服务后跑评测。复现命令与结果(照抄原文):

```
python3 -m sglang.launch_server --model Qwen/Qwen2-7B
python3 benchmark/gsm8k/bench_sglang.py

Accuracy: 0.000
Invalid: 1.000
Latency: 17.218 s
```

影响面为 Qwen 系列模型的生成质量(全部输出无效)。

## 根因

正文未直接披露机制,仅定位问题由 #2910 引入并 revert。(由 diff 推断)#2910 在 `Qwen2ForCausalLM.load_weights` 的权重遍历中,对名为 `lm_head.weight` 的条目直接 `continue` 跳过(原注释语境是 GPTQ 模型跳过额外 bias),该逻辑对所有权重来源生效,导致非 GPTQ 模型的 `lm_head.weight` 一直未被加载,输出层权重停留在初始化值,生成全部无效、准确率归零。

## 修复方案

整体 revert #2910:删除 `python/sglang/srt/models/qwen2.py` `load_weights` 中对 `lm_head.weight` 的跳过逻辑,使该权重正常加载:

```diff
             else:
                 # Skip loading extra bias for GPTQ models.
-                if "lm_head.weight" in name:
-                    continue
                 if name.endswith(".bias") and name not in params_dict:
                     continue
```

作者表示随后会补 Qwen 的 PR 测试。

## 影响与回归风险

- 影响窗口:自 #2910 合并起至本 revert。
- 回归面:仅 `qwen2.py` 的 `load_weights` 权重遍历路径;恢复加载 `lm_head.weight` 后,Qwen 系列各模型的输出层权重重新生效。#2910 原本针对 GPTQ 跳过额外 bias 的场景,若确有此类权重文件,可能需后续再以不误伤普通模型的方式处理(正文未展开)。

## 关联

- #2910(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
