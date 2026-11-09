# PR #806: Revert "Revert "fix: update flashinfer to 0.1.2 to fix sampling for cu118""

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#806](https://github.com/sgl-project/sglang/pull/806) |
| 作者 | Ying1123 |
| 合并日期 | 2024-07-29 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
恢复 flashinfer 版本要求从 0.1.1 升到 0.1.2(修 cu118 采样问题),因 0.1.2 已就绪,撤销此前的回滚。

## 问题现象
这是对回滚的再回滚(revert of revert):此前 PR 「fix: update flashinfer to 0.1.2 to fix sampling for cu118」把 flashinfer 版本要求升到 0.1.2 以修复 cu118 下的采样问题,随后被 PR #805 回滚;本 PR 撤销该回滚。正文未描述具体故障现象,仅说明「0.1.2 ready」——即 flashinfer 0.1.2 已可用,原升级可以恢复落地。

## 根因
按标题与回滚链推断:cu118 环境下 flashinfer 0.1.1 存在采样问题,需升级到 0.1.2 修复(标题原文「fix sampling for cu118」);#805 回滚的原因正文未说明,本 PR 以「0.1.2 ready」为由恢复升级。

## 修复方案
改 `python/sglang/srt/server.py` 的 `launch_server`:当未设置 `--disable-flashinfer` 时,`assert_pkg_version` 对 flashinfer 的版本要求从 `"0.1.1"` 恢复为 `"0.1.2"`:
```diff
         assert_pkg_version(
             "flashinfer",
-            "0.1.1",
+            "0.1.2",
             "Please uninstall the old version and "
             "reinstall the latest version by following the instructions "
             "at https://docs.flashinfer.ai/installation.html.",
```
即服务器启动时强制校验 flashinfer >= 0.1.2,不满足则提示按官方安装文档重装。

## 影响与回归风险
- 影响窗口:正文未说明(围绕 #805 回滚前后的 cu118 采样问题窗口)。
- 回归面:所有启用 flashinfer 的部署的启动版本校验——环境仍装 0.1.1 的用户启动会被 assert 拦下,需升级到 0.1.2;cu118 下的采样行为回到 0.1.2 的实现。`--disable-flashinfer` 路径不受影响。

## 关联
- https://github.com/sgl-project/sglang/pull/805(被本 PR 回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
