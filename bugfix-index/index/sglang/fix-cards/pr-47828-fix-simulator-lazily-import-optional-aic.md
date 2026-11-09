# PR #47828: [Fix] simulator: Lazily import optional AIConfigurator dependency

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#47828](https://github.com/sgl-project/sglang/pull/47828) |
| 作者 | thetumbled |
| 合并日期 | 2026-11-07 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 29 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
未安装可选依赖时,Replay predictor 启动即被 eager import 拖垮;改为惰性导入,仅在选择对应 predictor 或需要自动 KV-cache 估算时才加载,并给出可操作的缺依赖报错。

## 问题现象
在未安装可选依赖(正文称 `aiconfigurator`;diff 中对应模块名为 `aisimulate`,类名 `AISimulateTimePredictor`)时,以 Replay predictor 启动 simulator 直接失败:predictor 包、configuration manager 与 simulation utilities 中的 eager import 会在以下配置下也强制加载该依赖——即使用户选择了 Replay 且显式配置了 `--max-total-tokens`。正文明确:此配置下 Replay 本应完全不需要该依赖即可运行。

## 根因
正文披露:可选依赖的 import 是 eager 的,分布在 `time_predictor/__init__.py`(包级导出)、`simulation/manager/config.py`(配置管理器顶部导入)与 `simulation/utils.py`(工具函数顶部导入)三处,导致只要 import 这些模块就会触发可选依赖加载,未安装时启动即抛 `ModuleNotFoundError`,与是否真正使用该 predictor 无关。

## 修复方案
- `time_predictor/__init__.py`:移除包级 eager import,通过模块级 `__getattr__` 惰性暴露 `AISimulateTimePredictor`,保持原有包级导入路径不变;同时修正 `__all__` 为导出名字符串元组。
- `simulation/manager/config.py`:`get_inference_time_predictor` 中仅在 `predictor_config.get("name") == "aisimulate"` 分支内局部导入;
- `simulation/utils.py`:`profile_device_available_bytes` 中把 `get_perf_model` 导入移到函数内;
- `time_predictor/aisimulate.py`:缺依赖错误信息增加安装指引与使用提示:

```diff
         raise ModuleNotFoundError(
             "Install the AISimulate SDK with: "
-            "pip install -e 'tools/sglang-simulator[aisimulate]'"
+            "pip install -e 'tools/sglang-simulator[aisimulate]'. "
+            "For Replay or ML without automatic KV-cache memory estimation, "
+            "set --max-total-tokens explicitly."
         ) from exc
```

- 新增 `tools/sglang-simulator/test/test_optional_aisimulate.py`:用 meta_path finder 屏蔽 `aisimulate` 的隔离子进程回归测试,覆盖"无依赖时 Replay 初始化与预测正常且 `aisimulate` 未进入 `sys.modules`"、"缺依赖时四个入口报错均含安装指引与 `--max-total-tokens` 提示"、"无关缺依赖错误不被掩盖";并在 CI workflow `_pr-test-simulator-cpu.yml` 中加入该测试。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `tools/sglang-simulator/` 子目录与对应 CI workflow。正文声明调度、时延预测、KV-cache 容量估算算法与 benchmark 指标计算均未改动。受影响路径:predictor 包的 `from sglang_simulator.time_predictor import AISimulateTimePredictor` 用法经 `__getattr__` 保持兼容,但直接访问其他不存在属性现在抛 `AttributeError`;aisimulate predictor 的加载时机后移到首次使用;非 aisimulate 场景启动不再要求可选依赖。

## 关联
- Fixes sgl-project/sglang#47827(正文提及)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
