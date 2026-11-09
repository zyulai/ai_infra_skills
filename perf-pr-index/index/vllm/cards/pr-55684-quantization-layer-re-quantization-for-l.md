# PR #55684: [Quantization] Layer re-quantization for linear layers through online quantization API (MXFP8 -> FP8 PTPC showcase)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-10-06 |
| 分类 | Quantization (tier: major) |
| Label | documentation, ready, quantization |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/55684 |

## 一句话结论

在线量化 API 支持把 checkpoint 已量化的 linear 层在加载期换精度(source 实现 `dequantize_weight` 即可),展示路径为 ModelOpt MXFP8 linear → FP8 PTPC。

## 动机

checkpoint 的量化精度此前在加载期不可更改:online quantization 遇到 pre-quantized layer 直接抛 ValueError。此前有多个特例化请求(#48427、#51274),sglang 侧已有对应能力(#28291/#29328);RFC 见 issue #59062(回应 #52167 第三项)。目标是用通用 API 完成 re-quantization,避免每种 source→target 组合写专用代码。

## 优化点

机制:`QuantizeMethodBase` 新增可选 `dequantize_weight(layer)`(默认抛 NotImplementedError),量化方法的实现可按 weight quant key 在不同 producer(quark/modelopt/compressed-tensors)间共享。`resolve_quant_method` 改写:checkpoint 已量化且 online target 命中 linear 层时,不再报错,而是把 checkpoint method 挂为 online method 的 `requantization_source`(`OnlineLinearBase.set_requantization_source`);MoE 层显式拒绝。`OnlineLinearBase.create_weights` 在有 source 时先委托 source method 创建其原生参数并登记;`get_weight_for_quantization` 返回 `dequantize_weight` 的物化权重;各 online 方法(per-tensor/per-block/per-channel FP8、mxfp4、mxfp8)的 `process_weights_after_loading` 改用该权重做在线量化,成功后 `release_requantization_source_weights` 释放 checkpoint 参数,FP8 PTPC 将 per-channel scale 写回 `weight_scale`。组合仍由 `resolve_quant_method` 完成,模型代码继续拿到 checkpoint 的 quant config 类型;未选中的 linear 与 MoE 仍由 checkpoint quant config 负责。另为 online 量化增加耗时统计与日志(`online_quantization_time`)。

## 关键改动

```text
vllm/model_executor/layers/quantization/base_config.py: dequantize_weight 默认实现 + resolve_quant_method requant 分支
vllm/model_executor/layers/quantization/online/fp8.py: OnlineLinearBase 挂 source/释放/取权重的通用逻辑
vllm/model_executor/layers/quantization/modelopt.py: ModelOptMxFp8LinearMethod.dequantize_weight(kMxfp8Static)
vllm/model_executor/model_loader/utils.py: online_quantization_timing_context 计时
docs/features/quantization/online.md: 支持组合表与用法文档
```

## 性能声明

无性能数字;PR 未注明测试条件(仅"Unit test passing, end-to-end eval to be added")。收益定性:MXFP8 linear 可在加载期转为 FP8 PTPC(per-channel weight/per-token activation),以利用目标硬件上更快的 FP8 kernel;加载期多一次反量化+重量化,已计入新增的耗时日志。

## 验证与回归风险

`tests/quantization/test_online.py` 扩展:requantization 兼容矩阵(quark/modelopt 报 NotImplementedError、modelopt_mxfp8 成功)、block-FP8 requant 后移除 `weight_scale` 换 `weight_scale_inv`、TP 分片 dequant+quant 与全量一致(`test_modelopt_mxfp8_requantized_ptpc_tp_shards_match_unsharded`)、e2e `mgoin/Qwen3-0.6B-MXFP8` → `Fp8PtpcOnlineLinearMethod` 且 source 权重被释放。风险面:改变此前"online 遇 pre-quantized 层报错"的既有契约;MoE 不支持(显式拒绝);checkpoint-exclusion 策略不由 online 量化执行(仅 `--quantization-config.targets/ignore` 生效);source 未实现 `dequantize_weight` 时运行期报错。

## 关联

同区域:#59062(RFC)、#52167、#48427(本 PR 由此改编,co-author @tanpinsiang)、#51274;sglang 侧 #28291/#29328(同类功能)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
