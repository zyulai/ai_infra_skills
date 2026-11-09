# PR #6226: enable auto-round quantization model

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @WeiweiZhang1 |
| 合并 | 2025-09-08 |
| 分类 | Quantization |
| Label | high priority, intel |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/6226 |

## 一句话结论

让 SGLang 能加载 Intel AutoRound 量化的模型(auto_gptq/auto_awq 打包),把 W4 及更低 bit 权重量化接入现有内核。

## 动机

AutoRound(Intel,arXiv 2309.05516)在极低 bit 宽(如 2-bit)下提供更高的量化精度,覆盖 LLM/VLM 与多种格式,已被 vLLM、pytorch/ao 和 HF Transformers 集成;SGLang 此前无法加载 AutoRound 产出的 checkpoint,用户只能转用其他推理引擎或 HF Transformers。

## 优化点

新增 `auto-round` 量化配置:AutoRoundConfig 读取 checkpoint 的 quantization_config.json,支持 bits {2,3,4,8}、int 数据类型、`auto_round:auto_gptq` 与 `auto_round:auto_awq` 两种打包格式,backend 可选 auto/gptq/gptq:marlin/awq/awq:marlin/marlin。按 packing_format/backend 分派到既有实现:GPTQ 路径按 (bits, sym) 判定是否可用 Marlin 内核(4-bit sym 用 uint4b8,8-bit 用 uint8b128),否则走 GPTQLinearMethod;AWQ 路径同理;FusedMoE 在 Marlin 可用时经 MoeWNA16Config 复用 MoE Marlin 方法。逐层配置 get_layer_config 支持 extra_config 按层覆盖 bits/group_size/sym、block_name_to_quantize 限定量化块,并对 fused MoE、fused QKV 等融合层要求子层配置一致(不一致报错),未量化层(如 lm_head,ParallelLMHead)回落 UnquantizedLinearMethod。`auto-round` 加入 server_args 与 model_config 的合法量化列表;FusedMoE 层为 GPTQMarlinMoEMethod 等 quant method 传入 intermediate_size_full。已知限制(正文明示):mixed bits 受限(vLLM 式层融合下同一融合层的不同 bit 宽不兼容)、量化 MoE 的 gate 层不支持(需跳过)、部分 VLM 有问题(Qwen2.5-VL-7B 的 auto_gptq 准确率接近零、gptq 格式报形状不对齐,auto_awq/awq 正常)。

## 关键改动

```text
python/sglang/srt/layers/quantization/auto_round.py: 新增 AutoRoundConfig,分派到 GPTQ/AWQ/Marlin/MoeWNA16
python/sglang/srt/layers/quantization/__init__.py: 注册 "auto-round" 映射
python/sglang/srt/server_args.py / configs/model_config.py: 合法量化方式列表加入 auto-round
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: MoE quant method 增加 intermediate_size_full 参数
docs/advanced_features/quantization.md: AutoRound 离线量化用法与已知问题
```

## 性能声明

PR 正文无吞吐/延迟数字。定性声明引自 AutoRound 上游:在极低 bit 宽(如 2-bit)下精度显著更高、兼容面更广;低 bit 权重量化可减少显存占用,但 PR 未给出 SGLang 侧的对比测量。

## 验证与回归风险

新增 test/srt/quant/test_autoround.py:三个 AutoRound 量化模型(OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc 与 OPEA Llama-3.2-11B-Vision 的 auto_gptq、Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound 的 auto_awq)各起 server 跑 MMLU 64 题,阈值 Llama>=0.6、其余>=0.26。回归面:实现完全复用既有 GPTQ/AWQ/Marlin 内核,不新写 kernel;触及 fused_moe_triton/layer.py 的公共构造路径(仅对特定 quant method 增参);mixed-bits 与 MoE gate 层量化是已知不可用面。

## 关联

同区域:GPTQModel、AWQ、Marlin 等既有量化路径(AutoRound 直接复用其内核);上游集成参考 vLLM/pytorch ao/Transformers,素材未给出 SGLang 内其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
