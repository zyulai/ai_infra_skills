# PR #805: Revert "fix: update flashinfer to 0.1.2 to fix sampling for cu118"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#805](https://github.com/sgl-project/sglang/pull/805) |
| 作者 | Ying1123 |
| 合并日期 | 2024-07-29 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
回滚 #803:flashinfer 0.1.2 的 wheel 尚未上传,版本断言无法满足;恢复要求 0.1.1。

## 问题现象
正文说明:#803 把 flashinfer 版本要求升到 0.1.2 以修复 cu118 的采样问题,但 0.1.2 的 wheel 并未上传("The wheel of 0.1.2 has not been uploaded"),新版本断言无法满足。正文未附报错。

## 根因
正文披露:0.1.2 的 wheel 未发布/未上传,导致 #803 设置的版本要求在实际环境中不可安装、启动断言失败,故回滚。

## 修复方案
`python/sglang/srt/server.py` 的 `launch_server`:flashinfer 的 `assert_pkg_version` 目标版本由 `"0.1.2"` 改回 `"0.1.1"`,即整体回滚 #803 的版本要求变更。

```diff
         assert_pkg_version(
             "flashinfer",
-            "0.1.2",
+            "0.1.1",
```

## 影响与回归风险
- 回归面:仅影响 flashinfer 的启动版本校验:恢复要求 0.1.1。#803 想解决的 cu118 采样问题会随回滚而重新存在,待 0.1.2 wheel 发布后需再次升级(正文未给出后续计划)。

## 关联
- [#803](https://github.com/sgl-project/sglang/pull/803)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
