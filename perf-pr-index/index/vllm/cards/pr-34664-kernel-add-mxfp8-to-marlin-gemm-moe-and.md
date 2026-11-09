# PR #34664: [Kernel] Add MXFP8 to Marlin GEMM/MoE and refactor Mxfp8LinearOp

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-04-01 |
| 分类 | MoE |
| Label | performance, ready, nvidia, quantization |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 覆盖 14 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/34664 |

## 一句话结论

把 MXFP8(FP8 权重 + e8m0 每 32 元素块 scale)接入 Marlin GEMM 与 MoE kernel,使 MXFP8 从仅 Blackwell 降到 SM80+ 硬件可跑。

## 动机

Marlin 已支持 FP8(per-channel/group scale)与 MXFP4(per-32-element e8m0 scale),MXFP8 是两者的自然组合:FP8 权重配 e8m0 microscaling 块 scale,"只需把内核积木接起来"。此前 MXFP8 线性层与 MoE 只能走 FlashInfer CUTLASS(要求 SM100 Blackwell)或 torch 仿真,老架构(Ampere/Ada/Hopper)无加速路径。PR 同时把 GEMM 后端选择逻辑收拢进 `Mxfp8LinearOp`。

## 优化点

kernel 层:在 dense `csrc/quantization/marlin` 与 MoE `csrc/moe/marlin_moe_wna16` 的生成配置里各加一组 MXFP8 类型(`a_type=kBFloat16, b_type=kFE4M3fn, s_type=kFE8M0fnu, group_blocks=2`);模板中新增 `is_8bit_scale`(scale 为 8 位类型)统一替换原先按 `kFE2M1f` 特判的 scale stride/加载路径,并加 `static_assert(b_type==kFE4M3fn && group_blocks==2)` 约束;`dequant_skip_flop` 对 MXFP8 关闭(FP8 权重 + e8m0 scale 需要乘 scale,不能跳过 FLOP);scale 量化反解路径按 `s_type == kFE4M3fn || kFE8M0fnu` 走 FP8 scale 分支。入口 `marlin_gemm`/`moe_wna16_marlin_gemm` 识别 "b_type=FP8 且 b_scales=float8_e8m0fnu" 组合。Python 层:新增 `select_mxfp8_linear_backend()`(SM100→FLASHINFER_CUTLASS,SM80+ 支持 fp8 marlin→MARLIN,否则 EMULATION),`Mxfp8LinearOp` 接管三种后端的 `process_weights`/`apply`;新增 `prepare_mxfp8_layer_for_marlin`/`prepare_mxfp8_moe_layer_for_marlin`(pack_fp8_to_int32 + gptq_marlin_repack + marlin_permute_scales + `mxfp8_marlin_process_scales` 的 [0,2,1,3] 重排)以及 torch 实现的 `_mxfp8_e4m3_quantize_torch` 供非 Blackwell 在线量化;MoE 后端选择把 MARLIN 加入 mxfp8 支持集并按可用性 auto-select。

## 关键改动

```text
csrc/quantization/marlin/{generate_kernels.py,marlin_template.h,marlin.cu}: dense GEMM 新增 MXFP8 类型组合
csrc/moe/marlin_moe_wna16/{generate_kernels.py,marlin_template.h,ops.cu}: MoE Marlin 新增 MXFP8
vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py: 新增 MXFP8 权重/scale repack 与 apply_mxfp8_marlin_linear
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py: 后端自动选择(SM100 CUTLASS / SM80+ MARLIN / EMULATION)
vllm/model_executor/layers/quantization/{modelopt.py,mxfp8.py}: Mxfp8LinearOp 统一接管,最小能力从 100 降到 80
```

## 性能声明

PR 正文未给出 benchmark 数字;仅列出验证用的三种 serve 命令(mgoin/Qwen3-0.6B-MXFP8、Qwen3-0.6B --quantization mxfp8、Qwen3-30B-A3B --quantization mxfp8)。PR 未注明测试条件与性能结果。

## 验证与回归风险

- 测试:`tests/models/quantization/test_mxfp8.py` 在 L4(SM89)CI 上线(MOE_MODEL 从 Qwen3-30B-A3B 换成单卡可跑的 OLMoE-1B-7B-0125-Instruct,dense 模型不变);`tests/kernels/moe/test_moe.py` 增加 MXFP8(a=bf16, b=fp8_e4m3, group_blocks=2)kernel 用例。
- 风险:e8m0 scale 的 [0,2,1,3] 重排与 group_blocks=2 布局是 kernel 约定,错误布局只会表现为数值错误而非报错;`get_min_capability` 100→80 使原本在 SM90 上报不支持的配置变为走 Marlin/仿真路径,行为变化;Marlin 后端是 W8A16(BF16 activation),与 CUTLASS W8A8 的数值分布不同;在线量化在非 Blackwell 上退到 torch 逐块实现,加载性能可能下降。

## 关联

- 同区域:#14447/#16850(Marlin MoE/dense 优化)、#23204(删除旧 marlin 格式)、#23696(B200 MXFP4 CUTLASS MoE)、#49043 系的 mxfp8 utils 演进(#40960 携带 alignment 参数)。
- MXFP8 量化方法本身(modelopt MXFP8 / mxfp8 config)由此 PR 获得老硬件支持。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
