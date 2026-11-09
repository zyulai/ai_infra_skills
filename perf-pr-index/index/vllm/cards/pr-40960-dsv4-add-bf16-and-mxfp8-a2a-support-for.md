# PR #40960: [DSV4] Add BF16 and MXFP8 A2A support for flashinfer a2a one sided

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-04-30 |
| 分类 | MoE |
| Label | performance, deepseek 等多个标签(文档/performance/new-model/v1/nvidia 等) |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),下方 diff 仅供内容参考 |
| 链接 | https://github.com/vllm-project/vllm/pull/40960 |

## 一句话结论

把 FlashInfer NVLink one-sided A2A MoE 的 dispatch 载荷从仅支持 nvfp4 扩展到 BF16 与 MXFP8,让 DeepSeek-V4 等非 FP4 模型也能走低开销 one-sided 通信路径。

## 动机

FlashInfer one-sided all-to-all(`flashinfer_nvlink_one_sided`)原本只支持 nvfp4 dispatch:workspace 大小按 nvfp4(hidden_size//2 字节 + hidden_size//16 scale)硬编码,非 nvfp4 量化配置会直接 raise ValueError,DeepSeek-V4(BF16/MXFP8)无法使用该 EP 通信后端,只能退回 two-sided 或 allgather_reducescatter。PR 原文:"Originally Flashinfer one sided a2a only supports nvfp4 dispatch. Add BF16 and MXFP8 dispatch."

## 优化点

不改动通信内核本身,而是把 dispatch 载荷尺寸参数化:新增 `dispatch_dtype_bytes_per_elem`(BF16=2、mxfp8=1、nvfp4=0 表示半字节打包)与 `dispatch_scale_bytes_per_token`(nvfp4 为 hidden//16,mxfp8 按 `mx_alignment`(trtllm 后端取 256)对齐 hidden 后 //32,BF16 无 scale)两个参数,`MoeAlltoAll` workspace 据此分配。`prepare` 侧新增 `defer_input_quant` 分支(BF16 时直接以未量化隐藏态作为载荷,`a1q_scale=None`),quantize 调用链全线透传 `mx_alignment`;mxfp8 的 scale 接收 reshape 改用 `scale_elems_per_token` 而非硬编码。配套把 trtllm mxfp4 experts 的 `expects_unquantized_inputs` 改为 False(输入已在外部量化)、输出按 `hidden_dim_unpadded` 分配以处理 padding。文档 `moe_kernel_features.md` 同步 one-sided 支持 nvfp4,bf16,mxfp8。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/all2all_utils.py: 按 quant_dtype 计算 dispatch 载荷字节数,放开 nvfp4-only 校验
vllm/distributed/device_communicators/all2all.py: MoeAlltoAll workspace 按参数化载荷尺寸初始化
vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py: defer_input_quant 分支与 scale reshape 参数化
vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py: 输入已量化时的 apply 路径与 unpadded 输出
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py: mxfp8_quantize 增加 alignment 参数
```

## 性能声明

PR 正文未给出吞吐/延迟 benchmark 数字;Test Result 仅为 DeepSeek-V4-Flash 上的 gsm8k 精度验证:5-shot exact_match flexible-extract 0.9583±0.0055,strict-match 0.9591±0.0055。PR 未注明性能测试条件。

## 验证与回归风险

- 验证:gsm8k on V4-Flash(见上);diff 中未见新增单元测试文件。
- 风险面较宽:`moe_kernel_quantize_input` 的 mxfp8 分支把 `is_sf_swizzled_layout` 固定为 False(原先透传 `is_fp4_scale_swizzled`),影响 trtllm/two_sided/naive_dp_ep/no_dp_ep 等所有调用方;`expects_unquantized_inputs` 语义反转可能影响其他 mxfp4 后端组合;非 nvfp4/mxfp8/BF16 的 quant_dtype 现在抛 NotImplementedError(行为从 ValueError 改变)。

## 关联

- 同区域:#49043(FlashInfer MNNVL workspace 校验修复)、PR 20037 系(FlashInfer CUTLASS fused MoE)、flashinfer_nvlink_two_sided 后端。
- 标题中 DSV4 指向 DeepSeek-V4 支持;相关量化 PR:#34664(MXFP8 Marlin)、#23696(B200 MXFP4 CUTLASS MoE)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
