# PR #13167: [Model] Deepseek GGUF support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @SzymonOzog |
| 合并 | 2025-02-27 |
| 分类 | Quantization |
| Label | documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1 |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/13167 |

## 一句话结论

让 DeepSeek(含 MoE)GGUF 量化模型可在 vLLM 上加载运行,补齐量化格式支持而非提速。

## 动机

HuggingFace 不支持 DeepSeek 的 GGUF 元数据转 config,导致 unsloth 等 DeepSeek GGUF 量化模型无法直接在 vLLM 运行(Closes #12436)。作者加了 `--hf-config-path` 覆盖路径,可从原始 DeepSeek repo 读 config/tokenizer、从 GGUF repo 读权重。PR 自述目标是补功能:加载慢的问题已修,但 GGUF MoE 仍是"very naive implementation and is very slow"。

## 优化点

三层改动:其一,`ModelConfig`/`EngineArgs` 新增 `hf_config_path`,`get_config` 用它替代 model 路径解析 HF config,绕开 HF 对 GGUF 元数据的不支持。其二,新增 `GGUFMoEMethod` 挂到 `FusedMoE`:权重用 `GGUFUninitializedParameter` 惰性物化(按 loaded_weight shape 调整 TP 分片后 `materialize`),`apply` 里走 `FusedMoE.select_experts` 后逐 token、逐 expert 调 `_fuse_mul_mat`(GGUF 反量化 GEMM)+ SiluAndMul,是纯 Python 循环的朴素实现。其三,loader 的 `_get_gguf_weights_map` 对 `deepseek_v2/v3` 映射到 GGUF `deepseek2` arch,并手工补 `exp_probs_b`/`ffn_*_exps` 等专家权重名映射。

## 关键改动

```text
vllm/model_executor/layers/quantization/gguf.py: 新增 GGUFMoEMethod(create_weights 惰性参数 + 逐 token MoE apply)
vllm/config.py, vllm/engine/arg_utils.py: 新增 --hf-config-path,config 加载与 generation_config 读取走覆盖路径
vllm/model_executor/model_loader/loader.py: deepseek_v2/v3 -> deepseek2 的 GGUF 权重名映射
vllm/model_executor/layers/fused_moe/layer.py, linear.py: weight_loader 识别 is_gguf_weight 并物化 UninitializedParameter
```

## 性能声明

PR 未提供 benchmark 数字;并明确声明当前 GGUF MoE 实现很慢(逐 token Python 循环),作者计划后续优化。示例运行参数:`enforce_eager=True, tensor_parallel_size=4`(DeepSeek-R1-Q2_K GGUF)。本卡索引该 PR 是因其属量化支持版图,非性能优化。

## 验证与回归风险

无新增测试(测试计划为 CI)。回归面主要在通用权重加载路径:`FusedMoE.weight_loader` 与 `LinearBase.weight_loader` 加入了 GGUF 分支,影响所有模型的权重加载;`GGUFConfig.get_supported_act_dtypes` 从 `[half, bfloat16]` 收窄为 `[half]`。MoE 路径仅支持 SiLU 激活。

## 关联

Closes #12436。同属 GGUF 量化支持线;后续若有 GGUF MoE kernel 优化 PR 应挂在本卡之后。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
