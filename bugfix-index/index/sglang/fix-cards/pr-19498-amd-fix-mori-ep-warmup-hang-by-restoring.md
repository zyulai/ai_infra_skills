# PR #19498: [AMD] Fix MoRI EP warmup hang by restoring deepep_mode=normal default

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#19498](https://github.com/sgl-project/sglang/pull/19498) |
| 作者 | bingxche |
| 合并日期 | 2026-03-01 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 4 |
| 分类 | MoE |
| bug 类型 | hang |

## 一句话结论
PR #19216 移除 MoRI EP 自动设 `deepep_mode="normal"`,使其以 LOW_LATENCY 模式初始化并在 MI325X 8 卡 warmup 时挂死;恢复 normal 默认值修复。

## 问题现象
MoRI EP(moe_a2a_backend=mori)在 MI325X 8-GPU 配置下,warmup 阶段 hang。正文附 CI 失败日志佐证:sgl-project/sglang CI run 22471492647、job 65100816457(pr=19113)step:7:12894 处挂住。影响面为使用 MoRI EP 的 AMD MI325X 8 卡部署。

## 根因
正文披露:PR #19216 删除了 MoRI EP 自动设置 `self.deepep_mode = "normal"` 的逻辑,使 `deepep_mode` 保持默认值 `"auto"`;这导致 MoRI EP 以 `EpMode.LOW_LATENCY` 模式初始化,而 LOW_LATENCY 模式在 MI325X 8-GPU 配置下 warmup 时会 hang。

## 修复方案
在 `python/sglang/srt/server_args.py` 的 `_handle_a2a_moe` 中,mori 分支里当 `deepep_mode == "auto"` 时强制改回 `"normal"` 并打 warning:

```diff
         if self.moe_a2a_backend == "mori":
             self.ep_size = self.tp_size
+            if self.deepep_mode == "auto":
+                self.deepep_mode = "normal"
+                logger.warning("auto set deepep_mode=`normal` for MORI EP")
```

正文表述为「deepep_mode 为 unset 或 auto 时」,而 diff 实际仅判断 `"auto"`(即依赖 unset 时参数默认值即为 auto,由 diff 推断)。显式指定了 `deepep_mode` 的用户不受影响。

## 影响与回归风险
- 影响窗口:自 PR #19216 合并引入回归起,至本 PR 合并(2026-03-01)为止;正文未给出具体版本号。
- 回归面:仅影响 `moe_a2a_backend == "mori"` 且 `deepep_mode` 处于 auto 的配置:这些配置将固定以 `normal` 模式初始化 MoRI EP,并新增一条 warning 日志;显式设置 `deepep_mode` 的用户与其他 a2a 后端(deepep 等)路径无任何改动。改动只有 server_args 启动参数归一化,不触及 kernel/通信实现。

## 关联
- PR #19216(引入回归): https://github.com/sgl-project/sglang/pull/19216
- CI 失败日志: https://github.com/sgl-project/sglang/actions/runs/22471492647/job/65100816457?pr=19113#step:7:12894

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
