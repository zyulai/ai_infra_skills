# PR #14213: Add Mistral Large 3 support.

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @dcampora |
| 合并 | 2025-12-04 |
| 分类 | Model-specific Fast Path(major) |
| Label | high priority, quant, Multi-modal, deepseek, blackwell, run-ci, vlm, model-gateway |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数;diff 另有 7 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/14213 |

## 一句话结论

为 SGLang 接入 Mistral Large 3(MLA + MoE)与 Pixtral VLM,补齐 compressed-tensors fp8 MoE 量化支持。

## 动机

Mistral Large 3 尚未被 SGLang 支持;该模型为 MLA 架构 + MoE,且其 checkpoint 的量化格式与命名需要适配。同时 AutoConfig 暂不支持 ML3,PD 分离经 Dynamo 的 router manager 也有一个阻断问题(#12751)。

## 优化点

模型接入型 PR,含若干性能相关的量化路径扩展:新增 `MistralLarge3ForCausalLM`(继承 `DeepseekV3ForCausalLM`,带一整套 Mistral→SGLang 权重名 remapping,含 FP8 scale 项)与 `PixtralForConditionalGeneration`(VLM 支持);compressed-tensors 的 fp8 W8A8 MoE 从仅 per-tensor/per-channel 扩展出 BLOCK 策略(`weight_block_size`、`BlockQuantScaleParameter`、block-shaped w13/w2 scale,apply 走 `TritonMoeQuantInfo` 的 block 路径),linear 侧也支持 block fp8(`dispatch_w8a8_block_fp8_linear`、`validate_fp8_block_shape` 的 TP 分块校验);`model_config.py` 把两个新架构纳入 MLA/head_dim=256 分支与 `is_deepseek_nsa` 列表;`trtllm_mla_backend.py` 的 decode/extend 路径新增 `llama_4_scaling` 参数,`deepseek_v2.py` 按配置 `1 + beta*log(1+floor(pos/original_max_position_embeddings))` 计算并逐层施加;另有 AutoConfig 的 ML3 专用代码路径与 router manager 修复(PD 分离经 Dynamo 正常工作)。

## 关键改动

```text
python/sglang/srt/models/mistral_large_3.py: 新增 MistralLarge3ForCausalLM(权重 remapping 表)
python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py: fp8 W8A8 MoE 支持 BLOCK 策略
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py: block fp8 linear 的 weights/scale/apply
python/sglang/srt/models/deepseek_v2.py: _get_llama_4_scaling 及逐层透传
python/sglang/srt/configs/model_config.py: 新架构登记(MLA/head_dim/nsa 列表)
```

## 性能声明

无。PR 正文未提供任何 benchmark 或性能数字(Accuracy/Benchmarking 章节为空模板)。

## 验证与回归风险

PR 未提供精度或性能测试结果(相关 checkbox 未勾选)。风险面:模型接入触及共享基础设施——compressed-tensors fp8 MoE/linear 现在接受 BLOCK 策略,原先仅 per-tensor/channel 的判定与参数创建被重排;`is_deepseek_nsa` 把新架构纳入 index_topk 判定;`post_load_weights` 的 `linear_fp8_config` 回退逻辑与 fused qkv_a_proj 拼接处理了零维(empty)权重情形;trtllm MLA backend 新增可选参数为向后兼容(默认 None 不施 scaling)。

## 关联

关闭 issue #12751(router manager/PD 分离)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
