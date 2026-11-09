# PR #9336: Fix incorrect logic in chat template handling.

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#9336](https://github.com/sgl-project/sglang/pull/9336) |
| 作者 | lifuhuang |
| 合并日期 | 2025-08-20 |
| labels | ready-to-merge |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

HF 模板路径提前 return 跳过 reasoning 检测且日志错误;调整控制流让检测始终执行。

## 问题现象

正文列出两个问题:

1. chat template 相关的日志信息不正确(正文附截图显示错误提示);
2. 选择 `hf_template` 时 `_force_reasoning` 没有被设置。

正文未给出复现命令与影响面说明。

## 根因

正文点出两个问题但未展开机制。由 diff 可描述:`load_chat_template` 在使用默认 HuggingFace chat template 且检测到内容格式后直接 `return`,提前退出使函数末尾的 reasoning pattern 检测被跳过,这就是 hf_template 场景 `_force_reasoning` 未设置的来源;同时「No chat template found, defaulting to 'string' content format」的默认值设置与这条日志放在函数尾部,模板实际存在时也会走到,产生误导性日志(由 diff 推断)。

## 修复方案

改 `python/sglang/srt/managers/template_manager.py` 的 `load_chat_template`:把「默认 string content format + 日志」移入 `else` 分支,去掉提前 `return`,使后续 reasoning pattern 检测对 HF 模板路径也能执行;另在 `_detect_reasoning_pattern` 中为硬编码的 reasoning pattern 加 TODO 注释,不改其逻辑。

```diff
                 else:
-                    return
-
-            # Default to string content format if no template was found
-            self._jinja_template_content_format = "string"
-            logger.info("No chat template found, defaulting to 'string' content format")
+                    # Default to string content format if no template was found
+                    self._jinja_template_content_format = "string"
+                    logger.info(
+                        "No chat template found, defaulting to 'string' content format"
+                    )
```

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:所有走 `load_chat_template` 的启动路径。行为变化:hf_template 现在也会做 reasoning pattern 检测并可能设置 `_force_reasoning`;日志只在真正没有模板时打印。对显式提供模板文件的路径,逻辑不变。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
