# PR #18869: Fix AOPerModuleConfig name changes

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#18869](https://github.com/vllm-project/vllm/pull/18869) |
| 作者 | jerryzh168 |
| 合并日期 | 2025-06-06 |
| labels | ready, ci/build |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | hang |

## 一句话结论
torchao 上游把 AOPerModuleConfig 更名为 ModuleFqnToConfig,vLLM 同步改用新类名,并修正 float8/int4 测试与 CI 安装。

## 问题现象
正文未描述具体现象,仅写「Summary: also fixed float8 and int4 tests」,验证方式为运行 `python test/quantization/test_torchao.py`。结合 diff 看,受影响的是使用 torchao 量化且配置为 per-module config 的模型加载路径,以及引用旧模型仓的 quantization 测试。

## 根因
正文未披露根因。(由 diff 推断)torchao 新版本将 `AOPerModuleConfig` 类更名为 `ModuleFqnToConfig`,而 vLLM 的 `TorchAOConfig.get_quant_method()` 仍按旧类名 import 并做 isinstance 判断,旧类名在新版 torchao 中已不存在,导致该量化路径失效;同时 float8/int4 测试所引用的 HuggingFace 模型仓也已改名,旧仓名不再可用。

## 修复方案
- `vllm/model_executor/layers/quantization/torchao.py`:`get_quant_method()` 中 import 与 isinstance 判断由 `AOPerModuleConfig` 改为 `ModuleFqnToConfig`,并新增 logger;另在 `TorchAOConfig.__init__` 中加入一段(被注释掉的)standalone compile / compile cache 环境变量说明。
- `tests/quantization/test_torchao.py`:float8 测试模型由 `drisspg/float8_dynamic_act_float8_weight-opt-125m` 改为 `drisspg/fp8-opt-125m`;`test_opt_125m_int4wo_model_loading_with_params` 更名为 `test_opt_125m_int8wo_model_loading_with_params`,模型由 `jerryzh168/opt-125m-int4wo` 改为 `jerryzh168/opt-125m-int8wo-partial-quant`。
- `.buildkite/test-pipeline.yaml`:quantization 测试步骤临时添加 `pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126`,注释说明待 torchao 0.12 发布后移入 requirements/test.in。

关键 diff 摘录:

```diff
-        from torchao.quantization import AOPerModuleConfig
+        from torchao.quantization import ModuleFqnToConfig
         module_fqn = prefix
-        if isinstance(self.torchao_config, AOPerModuleConfig):
+        if isinstance(self.torchao_config, ModuleFqnToConfig):
```

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:torchao 量化路径中 `get_quant_method()` 的 per-module 配置分支在类名替换后需与新版 torchao 行为一致;quantization 测试的模型仓与测试函数名变更影响 CI;Buildkite 中临时安装 nightly torchao 可能带来版本波动。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
