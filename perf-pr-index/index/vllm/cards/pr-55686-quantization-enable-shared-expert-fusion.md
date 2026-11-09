# PR #55686: [Quantization] Enable shared expert fusion compatibility with online `shared_expert` quantization (showcase: along Quark MXFP4 routed experts)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-10-08 |
| 分类 | Quantization(tier: major) |
| Label | qwen, deepseek, quantization, kimi, k3, glm, minimax |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(15 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/55686 |

## 一句话结论

让 online `shared_expert` 量化与 fused shared expert(FSE)优化兼容,使 ROCm AITER 共享专家融合在 online 量化场景下不再被禁用。

## 动机

开启 `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 时,模型代码(如 qwen3_next.py)不再实例化独立的 `self.shared_expert` 模块,而是把共享专家权重并入 routed experts。此前若用户通过 online 量化(`--quantization-config.targets '{"*shared_expert*": "mxfp4"}'`)量化 shared expert,融合路径与量化路径互相冲突,FSE 只能放弃,共享专家只能留在 BF16 单独执行。本 PR 打通两者,使 online 量化的 shared expert 也能进入 fused MoE kernel。

## 优化点

核心思路:当 shared_expert 模块因 FSE 不存在时,借 `RoutedExperts` 的 weight loading 捕获 shared expert 权重的加载过程,在加载处完成量化,再把量化后的 shared expert 权重拷入 fused expert 表中对应的额外 expert slot。实现上尽量与 quant method 无关:原有 routed experts 的 `quant_method`(如 `QuarkOCP_MX_MoEMethod`,showcase 另含 `INCMxfp4MoEMethod`)保持不变,新增 `_fused_shared_expert_quantizer` 挂在 layer 上,覆盖 `load_weights`、fused `gate_up_proj` 与 DeepSeek 风格 `weight_loader` 三条加载路径。同时扩展 FSE 的量化兼容性检查(`is_shared_expert_quant_fse_compatible`):online target 只覆盖部分 shared expert 投影、或 online 量化 key 与 routed expert key 不匹配时,回退为不融合并给出明确 warning,而不是静默出错。本 PR 不做 experts + shared_expert 的联合 online 量化,experts 需已量化。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/config.py: FusedMoEConfig 新增 shared_expert_prefix 字段
vllm/model_executor/layers/quantization/quark/quark_moe.py: QuarkOCP_MX_MoEMethod 支持捕获并量化 fused shared expert 权重
vllm/model_executor/layers/fused_moe/utils.py: is_shared_expert_quant_fse_compatible 扩展 online 量化兼容性判定
tests/quantization/test_online_shared_expert_fusion.py: 新增 637 行测试,覆盖 13 个模型架构的三条加载路径
```

## 性能声明

PR 正文未提供吞吐/延迟 benchmark 数字(perf claim regex 未检出,正文亦无)。其价值表述为能力打通:online 量化的 shared expert 现在可以进入 AITER FSE 融合路径;测试日志显示 80 层 shared expert(40 层 down_proj + 40 层 gate_up_proj)以 `uint8 + static per-group-32 scale` 成功量化进 fused slot。测试命令:`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 下 serve `amd/Qwen3.5-35B-A3B-MXFP4`(TP2)。

## 验证与回归风险

新增 `tests/quantization/test_online_shared_expert_fusion.py`(7 passed,10.31s),parametrize 覆盖 AXK1/DeepSeek V2/V3/V32/Glm4Moe/Glm4MoeLite/Glm5Next/GlmMoeDsa/MiniMaxM3/Qwen3Next/Qwen3_5Moe/Qwen4Exp 共 13 个架构;另改动了多个既有测试把 `Mock` 换成真实类型(`resolve_quant_method_cls` 签名变化)、`test_reload.py` 增加 `_fused_shared_expert_quantizer` 属性。风险:影响所有启用 FSE 的量化模型(shared_expert 兼容性检查逻辑变化);`load_model_without_vllm_runner` 测试工具对嵌套 `text_config` 的层数截断逻辑改动可能波及其他量化测试;AITER topK 全局 metadata 在测试间需重置。

## 关联

同区域:shared expert fusion 与 quant 兼容性体系(可扩展至 #53161 提到的新精度);同 batch 的 #51274(Kimi-K3 gfx942 int4)同属 ROCm AITER 量化路径;`INCMxfp4MoEMethod` showcase 实现 PR 自述"possibly to be removed"。
