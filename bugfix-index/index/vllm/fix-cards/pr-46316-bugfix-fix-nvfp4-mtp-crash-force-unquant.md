# PR #46316: [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3Next

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#46316](https://github.com/vllm-project/vllm/pull/46316) |
| 作者 | WindChimeRan |
| 合并日期 | 2026-06-25 |
| labels | bug, quantization, qwen, ready |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
NVFP4 checkpoint 的 mtp.fc 是 bf16 但未被量化 ignore 规则覆盖,强制 mtp.fc 不量化修复加载崩溃。

## 问题现象
用 MTP 投机解码服务 NVFP4 Qwen3-Next checkpoint(如 `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`)时,draft 模型加载阶段崩溃:`load_column_parallel_weight` 中 `AssertionError` — 期望 param `[2048, 2048]` vs checkpoint `[2048, 4096]`。实测约 82% 的分片(shards)会触发。Fixes #35031。

## 根因
正文披露:`mtp.fc` 在 checkpoint 中以 **bf16** 存储,但量化 `ignore` 通配 `"mtp.layers.0*"` 匹配不到同级的 `mtp.fc`,于是 vLLM 把 `mtp.fc` 按 NVFP4 量化构建,打包后的量化参数无法加载 bf16 权重,形状不匹配报 AssertionError。

## 修复方案
`vllm/model_executor/models/qwen3_next_mtp.py`,对齐已合并的 #38832 的同一处理:
- 当量化方式为 `modelopt_fp4`(NVFP4)时,`mtp.fc` 强制传 `quant_config=None`(不量化),其余量化方式保持原样:
```python
fc_quant = None if (quant_config and quant_config.get_name() == "modelopt_fp4") else quant_config
self.fc = ColumnParallelLinear(..., quant_config=fc_quant, ...)
```
- 同时修正 `packed_modules_mapping["gate_up_proj"]`:`["up_proj","down_proj"]` 改为 `["gate_proj","up_proj"]`,与本文件的 `stacked_params_mapping` 及 base/sibling 模型保持一致。

**适用范围:** Qwen3-Next + MTP,仅 NVFP4;与硬件无关(权重加载 bug,不涉及 kernel)。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 `qwen3_next_mtp.py` 的 `mtp.fc` 构建与该文件的 packed_modules_mapping;影响所有以 NVFP4(modelopt_fp4)量化加载的 Qwen3-Next MTP 路径;gate_up_proj 映射修正影响该模型 MoE 层权重名映射,需注意与 checkpoint 命名匹配。

## 关联
- Fixes [#35031](https://github.com/vllm-project/vllm/issues/35031)
- [#35675](https://github.com/vllm-project/vllm/pull/35675)(正文称可关闭:已被 #38832 取代,本 PR 补齐 Qwen3-Next 剩余缺口)
- 参照实现 [#38832](https://github.com/vllm-project/vllm/pull/38832)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
