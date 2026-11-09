# PR #21003: Revert "Fix DeepSeek V32 FP4 test"

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21003](https://github.com/sgl-project/sglang/pull/21003) |
| 作者 | merrymercy |
| 合并日期 | 2026-03-20 |
| labels | deepseek |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#20984 给 DeepSeek V32 FP4 测试强注 HF_HUB_OFFLINE=0 破坏 CI,整体回滚恢复 popen_launch_server 原行为。

## 问题现象
正文只有一句:"Reverts sgl-project/sglang#20984 because it breaks ci"——即 #20984 合并后破坏了 CI。具体失败的 job、报错与影响范围正文均未描述。

## 根因
正文披露的回滚原因:#20984 "breaks ci",未给出更多机制。

(由 diff 推断)被回滚的 #20984 做了两件事:在 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` 与 `test_deepseek_v32_fp4_mtp_4gpu.py` 的 `popen_launch_server` 调用中强制注入 `env={"HF_HUB_OFFLINE": "0"}`(强制在线模式),并在 `python/sglang/test/test_utils.py` 的 `popen_launch_server` 里新增 `offline_force_disabled` 逃生口——当环境变量 `HF_HUB_OFFLINE == "0"` 时跳过 CI 离线模式。回滚把这两处改动整体移除,恢复 CI 中无条件走离线模式标记的原逻辑。

## 修复方案
1. `python/sglang/test/test_utils.py`(`popen_launch_server`):删除 `offline_force_disabled` 变量,恢复判断:

```diff
-    offline_force_disabled = env.get("HF_HUB_OFFLINE") == "0"
@@
-        if is_in_ci() and not offline_force_disabled:
+        if is_in_ci():
             per_run_marker_path = _try_enable_offline_mode_if_cache_complete(
```

2. `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`:两处 `popen_launch_server` 调用删除 `env={**os.environ, "HF_HUB_OFFLINE": "0"}` 参数及 `import os`。
3. `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`:两处同样删除 env 覆盖及 `import os`。

## 影响与回归风险
- 影响窗口:自 #20984 合并起至本回滚。
- 回归面:纯测试基础设施——`popen_launch_server` 恢复"CI 内一律尝试离线模式"的行为,所有经该工具启动 server 的测试都回到原路径;两个 DeepSeek V32 FP4(含 MTP)测试不再强制在线拉取。#20984 原本想修复的 DeepSeek V32 FP4 测试问题(如其标题 "Fix DeepSeek V32 FP4 test" 所指)可能随之复现,正文未说明后续处理。

## 关联
- 回滚对象:sgl-project/sglang#20984

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
