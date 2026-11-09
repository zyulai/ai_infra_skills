# PR #20984: Fix DeepSeek V32 FP4 test

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20984](https://github.com/sgl-project/sglang/pull/20984) |
| 作者 | Fridge003 |
| 合并日期 | 2026-03-20 |
| labels | deepseek |
| 热度 | 👍 0 · 💬 40 |
| 分类 | Quantization |
| bug 类型 | general |

## 一句话结论
DeepSeek V32 FP4 注册测试受 CI 离线模式自动启用影响,令测试显式设 `HF_HUB_OFFLINE=0` 强制在线并跳过离线模式。

## 问题现象
正文未描述具体现象(Motivation/Modifications 等章节均为模板占位,无实际内容)。从标题可知是 DeepSeek V32 FP4 相关测试失败需要修复;正文未给出失败日志或复现步骤。

## 根因
正文未披露根因。由 diff 推断:测试基建 `popen_launch_server` 在 CI 中会调用 `_try_enable_offline_mode_if_cache_complete` 自动启用 HF 离线模式;DeepSeek V32 FP4 的三个注册测试启动路径与该自动离线模式不兼容(如需在线拉取权重/配置),导致测试失败。修复方式是让测试显式声明强制在线。

## 修复方案
两处改动:
- `python/sglang/test/test_utils.py` 的 `popen_launch_server`:新增 `offline_force_disabled = env.get("HF_HUB_OFFLINE") == "0"`;当调用方显式设置 `HF_HUB_OFFLINE=0` 时,即使处于 CI 也跳过 `_try_enable_offline_mode_if_cache_complete` 的自动离线启用;
- `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`(2 处)与 `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`(2 处):在 `setUpClass` 启动服务器时传入 `env={**os.environ, "HF_HUB_OFFLINE": "0"}`,强制该测试在线运行。

关键 diff 摘录:

```diff
+    offline_force_disabled = env.get("HF_HUB_OFFLINE") == "0"
@@
-        if is_in_ci():
+        if is_in_ci() and not offline_force_disabled:
             per_run_marker_path = _try_enable_offline_mode_if_cache_complete(
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,未写。
- 回归面:影响所有经 `popen_launch_server` 启动服务器的测试:只有显式设置 `HF_HUB_OFFLINE=0` 的调用才绕过 CI 自动离线模式,其余测试行为不变(由 diff 推断)。DeepSeek V32 FP4 三个测试场景改为在线模式,启动将依赖网络可达 HF(或在 CI 缓存路径上行为不同)。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
