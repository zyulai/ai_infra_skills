# PR #24562: Fix performance regression on Deepseek V3 on `moe-runner-backend=triton` on SM90

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24562](https://github.com/sgl-project/sglang/pull/24562) |
| 作者 | b8zhong |
| 合并日期 | 2026-05-09 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
PyTorch 2.11 带来 Triton 3.6.0 无调优配置,回退错选旧版 3.3.1 配置;改为磁盘发现配置目录并按新到旧匹配。

## 问题现象
DeepSeek V3 在 SM90 上使用 `moe-runner-backend=triton` 出现性能回归。正文披露触发链:近期 PyTorch 升级到 2.11 后使用 Triton 3.6.0,而该版本没有对应的 tuned config;回退逻辑落到了 3.3.1 的配置上,这不是正确选择——应优先使用更新的 3.5.1。使用旧配置导致 fused MoE kernel 性能下降。

## 根因
正文披露:`get_moe_configs` 的回退搜索依赖一份硬编码的受支持 Triton 版本列表(`["3.4.0", "3.3.1", "3.2.0", "3.1.0"]`),不包含 3.5.1/3.6.0 等新版本;当运行环境使用 Triton 3.6.0(随 PyTorch 2.11)时,找不到精确匹配的调优配置,回退逻辑按该列表顺序选中了过旧的 3.3.1 配置,造成性能回归。正确的回退目标是更新的 3.5.1。

## 修复方案
改 `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` 的 `get_moe_configs`:删除硬编码的 `supported_triton_versions` 列表,改为在磁盘上动态发现可用配置目录并按版本号从新到旧排序后遍历:
```diff
-    # Supported Triton versions, should be sorted from the newest to the oldest
-    supported_triton_versions = ["3.4.0", "3.3.1", "3.2.0", "3.1.0"]
...
-    for try_triton_version in supported_triton_versions:
+    configs_root = os.path.join(config_dir, "configs")
+    available_versions = sorted(
+        (
+            d.removeprefix("triton_").replace("_", ".")
+            for d in os.listdir(configs_root)
+            if d.startswith("triton_")
+        ),
+        key=lambda v: tuple(int(x) for x in v.split(".")),
+        reverse=True,
+    )
+
+    for try_triton_version in available_versions:
```
即:列出 `configs/` 下所有 `triton_*` 目录,把目录名还原为版本号(如 `triton_3_5_1` → `3.5.1`),按数值降序尝试与当前 Triton 版本不同的最近版本配置文件,找到即返回。这样 Triton 3.6.0 会优先命中 3.5.1 的新配置而非 3.3.1。

## 影响与回归风险
- 影响窗口:PyTorch 升级到 2.11(随带 Triton 3.6.0)之后至本修复前。
- 回归面:所有走 `get_moe_configs` 的 MoE triton 路径(不限于 DeepSeek V3/SM90)。回退候选集从固定四个版本变为「磁盘上实际存在的 triton_* 目录」:若某环境 configs 目录被裁剪,可选配置可能变少;目录名不符合 `triton_<版本>` 约定或版本段含非数字时,排序的 `int()` 转换可能报错(diff 未做容错)。命中顺序由列表顺序变为按数值版本降序,匹配结果的优先级可能与旧硬编码顺序不同。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
