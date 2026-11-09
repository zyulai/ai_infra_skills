# PR #34774: [Fix] has_hf_quant_config crashes on local dirs without the config

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34774](https://github.com/sgl-project/sglang/pull/34774) |
| 作者 | gilfordting |
| 合并日期 | 2026-08-14 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Quantization |
| bug 类型 | crash |

## 一句话结论
`has_hf_quant_config()` 对没有 `hf_quant_config.json` 的本地模型目录会把路径当 repo id 传给 hub 助手而抛 `HFValidationError`,在调度器初始化时把进程杀死;本地路径改为纯文件系统判断。

## 问题现象
`has_hf_quant_config()` 对任何**不含 `hf_quant_config.json` 的本地模型目录**都会崩溃,并在 init 阶段杀死 scheduler。实际触发场景:本地目录的投机解码 draft 模型搭配量化 target(草稿 checkpoint 不带 `hf_quant_config.json`)——作者正是在用 NVFP4 target + 本地 draft 提供服务时发现该问题。该函数对本地目录只能回答"有配置",永远无法回答"没有":不是返回 True 就是直接崩溃。

## 根因
正文披露:现行代码先检查正向本地情形(`os.path.exists(<model_path>/hf_quant_config.json)` → 返回 True),但当文件不存在时会继续落到 `huggingface_hub.try_to_load_from_cache(model_path, ...)`,该函数把第一个参数当作 hub repo id 校验,遇到文件系统路径(如 `/models/my-draft/`)就抛 `HFValidationError`,而该异常未被任何 handler 捕获,直接逃逸导致崩溃。

## 修复方案
改 `python/sglang/srt/utils/common.py` 的 `has_hf_quant_config`:本地路径完全由文件系统判定——`os.path.isdir(model_path)` 为真时直接返回 `os.path.isfile(<model_path>/hf_quant_config.json)`,只有真正的 repo id 字符串才会走到 hub 助手。hub repo id 与"本地目录含配置"两种情形的行为不变:

```diff
-    # Check if the model_path is a local path
-    if os.path.exists(os.path.join(model_path, "hf_quant_config.json")):
-        return True
+    # Local paths are decided on the filesystem; the hub helpers below
+    # reject them as invalid repo ids.
+    if os.path.isdir(model_path):
+        return os.path.isfile(os.path.join(model_path, "hf_quant_config.json"))
 
     from huggingface_hub import try_to_load_from_cache
```

并在 `test/registered/unit/utils/test_common.py` 增加三个 CPU-only 回归测试:本地目录含配置 → True;本地目录无配置 → False(此前抛异常);本地目录绝不调用 `try_to_load_from_cache`。后两个在修复前的 main 上失败、修复后通过。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `has_hf_quant_config` 一个函数与新增测试。所有调用该函数做量化配置探测的路径受影响:本地目录现在走一次 `isdir`/`isfile` 而非可能抛异常的 hub 调用(正文说明还省掉异常开销);hub repo id 情形与本地含配置情形行为不变。风险点在于对"本地目录"与"repo id"的判定边界依赖 `os.path.isdir`——一个恰好存在的同名本地目录会被当作本地路径处理(此为 diff 所示行为,正文未讨论该边界)。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
