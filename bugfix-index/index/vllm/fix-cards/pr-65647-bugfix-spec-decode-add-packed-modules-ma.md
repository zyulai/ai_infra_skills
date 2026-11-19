# PR #65647: [Bugfix][Spec Decode] Add packed_modules_mapping to DeepSeek-V4 DSpark models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#65647](https://github.com/vllm-project/vllm/pull/65647) |
| 作者 | peterzhang987723 |
| 合并日期 | 2026-11-19 |
| labels | bug, ready, ci/build, deepseek, DSv4, dflash, DSv4.1 |
| 热度 | 👍 0 · 💬 44 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
DSpark 投机解码 draft 模型(`DSparkDeepseekV4ForCausalLM`)漏掉了目标模型已有的 `packed_modules_mapping`,量化器无法把 draft 的融合层前缀展开回分片名,`quantization_config.ignored_layers` 里标记为 BF16 不量化的 draft 线性层仍被套上 `Fp8LinearMethod` 与未初始化的 `weight_scale_inv`,导致 0% draft token 接受率与无效输出;补上映射即修复。

## 问题现象
当 checkpoint 通过 `quantization_config.ignored_layers` 把 draft 的线性层保留为未量化 BF16 时,量化器用 `is_layer_skipped(prefix, ignored_layers, fused_mapping=quant_config.packed_modules_mapping)` 判断层是否跳过;由于 `DSparkDeepseekV4ForCausalLM` 缺少 `packed_modules_mapping`,draft 的融合层前缀(`fused_wqa_wkv`、`gate_up_proj`)无法展开为构成它们的分片名(`wq_a`/`wkv`、`w1`/`w3`),这些融合层仍然拿到 `Fp8LinearMethod` 和未初始化的 `weight_scale_inv`,最终表现为 0% draft token 接受率和无效输出。该问题横跨 AMD、NVIDIA、XPU 三个平台(fixes #65618)。

## 根因
正文披露根因:`DSparkDeepseekV4ForCausalLM` 通过 `stacked_params_mapping` 融合 `gate_up_proj`(分片 `w1`、`w3`)与 `fused_wqa_wkv`(分片 `wq_a`、`wkv`),但与目标模型 `DeepseekV4ForCausalLM` 不同,它没有定义 `packed_modules_mapping`。量化配置经 `configure_quant_config` 装配时拿不到该映射,`is_layer_skipped` 的 `fused_mapping` 缺失,ignored_layers 中的分片粒度记录无法匹配到融合前缀,本应跳过量化的 draft 融合层被错误量化。

## 修复方案
1. 为 5 个 DSpark 入口补上目标模型的映射(类属性直接引用):

```python
packed_modules_mapping = DeepseekV4ForCausalLM.packed_modules_mapping
```

   覆盖 `vllm/models/deepseek_v4/{amd,nvidia,xpu}/dspark.py` 与 `vllm/models/deepseek_v41/{amd,nvidia}/dspark.py`(v4.1 引用 `DeepseekV41LLMForCausalLM.packed_modules_mapping`)。
2. NVIDIA 平台的 `context_wkv_proj` 构造改为先检查 quantization config 存在且支持 ignored-layer 匹配模式,再用 `is_layer_skipped` 判断 `layers.{num_hidden_layers}.attn.wkv` 是否被忽略,被忽略时传 `quant_config=None`(不量化);同时消除先前 pre-commit 的 10 个 mypy 错误。
3. 新增 `tests/models/test_deepseek_dspark_packed_mapping.py`(166 行):断言各 DSpark 入口的 `packed_modules_mapping` 与目标模型一致、`configure_quant_config` 正确接线、融合前缀经 `is_layer_skipped` 可命中 ignored_layers;另测 NVIDIA `context_wkv_proj` 对 ignored_layers 的尊重与否两种行为。ROCm 用例仅在 ROCm 平台导入(避免在 x86 CPU 上二次注册 `_C` 扩展导致收集失败)。

## 影响与回归风险
- 修复对象:使用 DSv4/DSv4.1 DSpark draft 且 checkpoint 带 `ignored_layers` 的投机解码部署;接受率从 0% 恢复为正常。
- 行为变化:除补映射外,`context_wkv_proj` 在 `wkv` 被忽略时不再挂量化方法;无 ignored_layers 的部署路径不变。
- 平台覆盖:NVIDIA v4/v4.1、XPU 映射与 NVIDIA `context_wkv_proj` 用例在 CPU 测试通过(PyTorch 2.14.0,macOS arm64,7 例);AMD 两个映射用例由现有 MI355 Basic Models (Other) job 在 ROCm 上验证。ROCm 平台未在本地测试。
- CI 兼容性:测试不再全局 patch CUDA 设备发现,先前 CPU Buildkite job 因导入 ROCm 双注册 `_C` 扩展而 abort 的问题随之消除;`.buildkite/test_areas/models_basic.yaml` 登记新测试文件。

## 关联
- Fixes #65618(正文声明)。
- 涉及文件:`vllm/models/deepseek_v4/{amd,nvidia,xpu}/dspark.py`、`vllm/models/deepseek_v41/{amd,nvidia}/dspark.py`、`tests/models/test_deepseek_dspark_packed_mapping.py`、`.buildkite/test_areas/models_basic.yaml`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
