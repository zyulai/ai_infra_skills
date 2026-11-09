# PR #56685: [Feature][Humming] Humming feature integration

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2026-09-20 |
| 分类 | Quantization(tier: major) |
| Label | ready, ci/build, nvidia, quantization |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(54 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/56685 |

## 一句话结论

扩展 vLLM 对 Humming 量化库的集成:在线量化支持 Hadamard 变换、SM120/121 的 NVFP4/MXFP4/MXFP8,以及把激活、Hadamard 与输入量化融合的统一 `process_input` kernel 覆盖 dense/MoE 各路径。

## 动机

Humming 是 NVIDIA 的量化 kernel 库,vLLM 已有基础接入;本 PR 一次补齐多块能力:在线量化缺 Hadamard 变换( rotations 类 W4A4/FP4 量化需要)、新硬件 SM120/SM121(含非公开的 `float4e0m3`/`float8e3m4` dtype,依赖 humming#41)不支持、输入量化模式不足(静态/动态组合缺三种)、以及 MoE 路径中 activation/量化/permute 多次读写带来的冗余访存。

## 优化点

三块机制:(1) 在线量化经 `VLLM_HUMMING_ONLINE_QUANT_CONFIG` 的 `hadamard_block_size` 启用 Hadamard 变换;(2) 统一 `process_input` kernel 把可选 activation、Hadamard 变换与输入量化融合为一次遍历,覆盖 dense、MoE gate/up(indexed / grouped_contiguous / grouped_mask)与 MoE down(同样三种)共 7 条路径,grouped 路径跳过 EP 的无效 padding token;自定义 activation 支持 unary / binary_split / binary_interleaved 三种模式,实现为 CUDA C++ 表达式。为此扩展底层 MoE 算子:`moe_permute`/`moe_align_block_size` 新增返回 `scatter_idx`(在 count/sort kernel 里顺带写有效路由的 scatter 映射,EP 下按 expert_map 与 token_mask 置 -1),新增 `moe_prepare_scatter` 入口,`computeExpertFirstTokenOffset` 合并为 `computeExpertOffsetsAndInverse`(一次 kernel 同时算 expert offset 与 inverse 索引);(3) 输入 schema:尊重 checkpoint 的输入量化设置,设备不支持配置的激活量化类型时自动回退更高精度 schema;`VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"bfloat16"}'` 可保留原 WNA16 行为;显式指定 schema 默认禁用 fallback,除非 `"allow_fallback": true`。另加 `HummingMxFp6LinearKernel`、`humming_utils.py` 重构(校验/转换逻辑移入 Humming 库)与类型标注。

## 关键改动

```text
csrc/libtorch_stable/moe/moe_align_sum_kernels.cu: count/sort kernel 写 scatter_idx,无效路由置 -1
csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu: 抽出 moe_sort_routing;新增 moe_prepare_scatter
csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu: computeExpertOffsetsAndInverse 融合 offset+inverse
vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py: process_input 接入 7 条 dense/MoE 路径
vllm/model_executor/layers/quantization/utils/humming/(原 humming_utils.py): 重构与校验下沉
```

## 性能声明

PR 正文无 benchmark 数字(性能动机仅以机制表述:activation+Hadamard+量化融合避免中间 buffer 的多次读写;grouped 路径跳过 EP 无效 padding)。新量化格式能力(NVFP4/MXFP4/MXFP8/MXFP6 on SM120/121)也未附加速或精度数据。PR 未注明测试条件下的性能对比。

## 验证与回归风险

新增 `tests/kernels/core/test_activation.py::test_humming_activation_matches_framework`(HUMMING_ACTIVATION_CASES:silu/gelu/situ/swigluoai 等 + relu2 + situ linear_beta 变体,含 clamp 边界值 0/0.001/2.99/3.01/6.99/7.01 等,与框架 `apply_moe_activation` 逐元素对比)与 `test_moe.py::test_humming_indexed_routing_matches_each_projection_tile`(w13/w2 独立 tuning tile 下路由行与专家身份保持、共享 sorted_ids 缓冲区);`test_moe_align_block_size.py::test_moe_align_scatter_masks_invalid_routes` 验证 scatter 掩蔽;grouped apply 测试从 `apply_activation` 改为 `process_input` 契约(`quanted_down_input` 替代 `activation_output`)。风险:改动触及通用 MoE CUDA 算子签名(`moe_align_block_size` 新增 scatter_idx 参数、torch library schema 变化),所有调用方需适配;Humming 路径缓冲区命名/契约变化影响下游测试;依赖 CUDA >= 12.0 与 humming 库版本(含 humming#41 的 dtype 支持)。

## 关联

同区域:Humming 相关量化卡与 NVIDIA FP4/MXFP 量化系列(pr-28892、pr-39510、pr-26107);上游依赖 vllm-project/humming#41。PR 正文未列出其他 vLLM 侧关联编号。
