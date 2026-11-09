# PR #33174: Add support for Mistral Large 3 inference with Flashinfer MoE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @dbari |
| 合并 | 2026-01-31 |
| 分类 | Speculative Decoding(classify 输出;正文主体为 MoE 后端支持) |
| Label | performance, ready, ci/build, deepseek, nvidia |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 全文可用) |
| 链接 | https://github.com/vllm-project/vllm/pull/33174 |

## 一句话结论

让 Mistral Large 3 在 Blackwell 上可用 Flashinfer TRTLLM MoE 后端:8×B200 上相对 Triton fused MoE,输出吞吐 +15%~17%、TTFT −14%~−18%。

## 动机

Mistral Large 3(675B MoE)此前在 vLLM 只能走通用 Triton fused MoE,Flashinfer TRTLLM 的 `latency` MoE 后端虽在 Blackwell 上更快,但存在两个阻断:其一,flashinfer 0.6.1 的 blockwise FP8 MoE kernel 有缺陷;其二,vLLM 侧 per-tensor FP8 的 Flashinfer MoE 调用硬绑 Llama4 路由(`RoutingMethodType.Llama4` + custom_routing_function 断言),Mistral 的 renormalize 路由无法进入。同时 Mistral Large 3 的形状(E=128/N=512、Eagle draft 模型 E=16/N=4096)缺少调优过的 Triton 配置,作后备路径时性能也欠佳。

## 优化点

三线并进:一是升级 Flashinfer 0.6.1 → 0.6.2(Dockerfile/versions.json/requirements 同步),引入修复后的 blockwise quantized FP8 MoE kernel;二是放开 Flashinfer TRTLLM per-tensor FP8 MoE 的路由限制——`_supports_routing_method` 在 (kFp8StaticTensorSym, kFp8StaticTensorSym) 下接受 Llama4、Renormalize、RenormalizeNaive,`apply_fi_trtllm_fp8_per_tensor_moe` 改用 `layer.routing_method_type` 并按路由类型分别校验(Llama4 要求非 renormalize + 专用 routing 函数,其余要求无 custom routing);三是在 `benchmark_moe.py` 中补 MistralLarge3ForCausalLM 参数解析、Pixtral 递归取 text_config、compressed-tensors `block_structure` 回退读取,并新增 6 份调优配置:B200/GB200/H200 的 E=128,N=512(fp8_w8a8 与 bf16)TP8 及含/不含 EP 的变体、B200/H200 的 E=16,N=4096 fp8 block_shape=[128,128](Eagle draft 模型)。附带修 deepseek_v2 架构下 n_group/topk_group 为 1 时不再走 grouped topk,使非分组路由模型可复用该路径。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py: FP8 per-tensor 路由方法白名单扩展
vllm/model_executor/layers/quantization/utils/flashinfer_utils.py: 按 routing_method_type 分支校验与调用
vllm/model_executor/layers/fused_moe/configs/: 6 份 Mistral Large 3/draft 模型调优配置(B200/GB200/H200)
benchmarks/kernels/benchmark_moe.py: 模型参数解析与 block_structure 支持
docker/versions.json、requirements/cuda.txt: flashinfer 0.6.1 → 0.6.2
vllm/model_executor/models/deepseek_v2.py: n_group/topk_group=1 时禁用 grouped topk
```

路由限制放开(节选):

```python
elif (weight_key, activation_key) == (kFp8StaticTensorSym, kFp8StaticTensorSym):
    return routing_method in [
        RoutingMethodType.Llama4,
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
    ]
```

## 性能声明

精度:lm-eval gsm8k(mistralai/Mistral-Large-3-675B-Instruct-2512,local-completions,temperature 0.01/top_p 0.9999999/max_gen_toks 8192,limit 1000,batch_size 1;表格 n-shot 列为 5):flexible-extract exact_match 0.940 ± 0.0075,strict-match 0.907 ± 0.0092。

性能(8×B200,Flashinfer MLA + FP8 KV-cache,仅切换 MoE 后端):

| MoE 后端 | Concurrency | TTFT (ms) | Output throughput (t/s) |
| --- | --- | --- | --- |
| Triton | 16 | 600.48 | 847.87 |
| Flashinfer | 16 | 515.33 | 978.03 |
| Triton | 64 | 858.36 | 2166.99 |
| Flashinfer | 64 | 700.76 | 2526.54 |

即 Flashinfer 相对 Triton:c16 吞吐 +15.3%、TTFT −14.2%;c64 吞吐 +16.6%、TTFT −18.4%。Triton 配置本身的收益未单独给出数字。

## 验证与回归风险

更新 `tests/kernels/moe/test_flashinfer.py`(8bit mock 层补 routing_method_type/renormalize 以匹配新接口)。回归风险:Flashinfer 版本号被 docker/requirements 强绑定,升级影响所有 Flashinfer 用户(0.6.2 为前提);per-tensor FP8 MoE 从 Llama4 独占变为多路由共享,原 Llama4 断言被分支校验替代,custom_routing_function 与路由类型的组合约束需依赖新 assert;deepseek_v2 的 grouped-topk 条件变化影响该架构家族的非分组模型行为。分类标签含 deepseek/Eagle draft,提示需覆盖 DeepSeek 系模型回归。

## 关联

Supersedes: #29884;依赖: #33008(Mistral Large 3 模型加载)。同区域:索引中 SM100 Flashinfer MoE 相关卡(pr-22357-nvidia-add-sm100-flashinfer-cutlass-moe 为本批次另一张)。Eagle draft 模型配置与本 PR 被归入 Speculative Decoding 分类的部分相关。
