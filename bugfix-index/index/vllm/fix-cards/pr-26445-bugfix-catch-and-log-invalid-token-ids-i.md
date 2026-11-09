# PR #26445: [Bugfix] Catch and log invalid token ids in detokenizer #2

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26445](https://github.com/vllm-project/vllm/pull/26445) |
| 作者 | njhill |
| 合并日期 | 2025-10-09 |
| labels | bug, ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Sampler / Detokenization |
| bug 类型 | correctness |

## 一句话结论
tokenizers 0.22.1 下偶发非法 token id 从 OverflowError 变为 TypeError,原有保护未接住;扩大捕获范围并改用 %r 记录。

## 问题现象
正文说明:模型偶尔会产生负数(非法)token id,该问题的根因至今未知(见 #21951)。此前 #24351 加了"workaround"把 detokenizer 与这类 id 隔离,避免引擎直接崩溃。升级到 `tokenizers` 0.22.1 后,同样的错误从 `OverflowError` 变成 `TypeError`,旧补丁捕获不到,相关崩溃在 #26438、#26071、#25821 中报告(本 PR 声明缓解这三个 issue)。

## 根因
正文披露:根因仍在诊断中(#21951 "still to be diagnosed",偶发产生的负 token id 原因未知);本 PR 处理的是依赖升级带来的表现变化——`tokenizers` 0.22.1 对非法 id 抛 `TypeError` 而非 `OverflowError`,导致既有异常捕获漏接。

## 修复方案
`vllm/v1/engine/detokenizer.py` 的 `_protected_step`:

```diff
-        except OverflowError:
+        except (OverflowError, TypeError):
             # Handle rare observed overflow, still to be diagnosed.
             # See https://github.com/vllm-project/vllm/issues/21951.
-            logger.exception("Encountered invalid token id: %d", next_token_id)
+            logger.exception("Encountered invalid token id: %r", next_token_id)
             token = None
```

即同时捕获两类异常,记录日志后返回 None 继续运行;日志格式从 `%d` 改为 `%r` 以便如实记录非法值。

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;风险敞口是 `tokenizers` 升级到 0.22.1 后仍带 #24351 旧补丁的部署。
- 回归面:仅 detokenizer 的异常处理分支,扩大捕获范围;正常 token id 的 detokenize 路径不变;非法 id 时行为从崩溃变为记日志并跳过该 token(与 #24351 的意图一致)。

## 关联
- #21951(根因未诊断的原始问题)
- #24351(原 workaround PR)
- #26438、#26071、#25821(本 PR 缓解的三个 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
