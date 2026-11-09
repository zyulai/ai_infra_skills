# PR #44120: [MoE Refactor] Migrate MoeWNA16Method quantization method over to using the new MK oracle scheme.

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @bnellnm |
| 合并 | 2026-07-22 |
| 分类 | MoE(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 可用但另有 4 个文件因超长未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/44120 |

## 一句话结论

把 W4A16/W8A16 MoE 量化方法 MoeWNA16Method 迁移到新的 MK oracle 后端选择框架,统一 GPTQ/AWQ/压缩张量 MoE 的 kernel 分派与权重格式转换。

## 动机

vLLM 正在把 fused MoE 的后端选择收敛到"MK oracle"统一框架(按平台、量化方案、并行配置自动挑选 kernel)。`MoeWNA16Method`(覆盖 AutoGPTQ、AutoAWQ、compressed-tensors 的 INT W4/W8 × A16 专家量化)此前走独立路径,与 oracle 框架脱节:后端能力声明未接入(TRITON 专家类直接 raise NotImplementedError)、checkpoint 布局差异(AutoGPTQ vs MoeWNA16 的 N-first/K-first 打包、zero point dtype)缺少统一转换、以及 humming 模块未安装时量化导入路径损坏(早于 #42647 即已存在)。本 PR 是 #42647 的修复 + 更新版。

## 优化点

核心是把 WNA16 MoE 并入 oracle:`WNA16MoEBackend` 新增 `TRITON` 枚举并加入优先级链(FLASHINFER_TRTLLM → MARLIN → BATCHED_MARLIN → TRITON → HUMMING → EMULATION);`TritonWNA16Experts` 实现全套能力声明(支持 CUDA-alike/XPU、kInt4Static/kInt8Static/kInt4Static32 权重键、9 种激活、排除 FI NVL one/two-sided 并行)。新增 `_backend_incompatibility_reason()` 在后端选择阶段按 checkpoint 结构提前拒绝:TRITON 不支持 expert bias、AutoAWQ 权重布局、GPTQ/group activation ordering;FLASHINFER_TRTLLM 不支持 zero point/bias;MARLIN 系不支持 MoeWNA16 checkpoint 布局——显式请求不兼容后端时抛错,auto 模式则跳到下一后端。权重格式转换补齐三条:`convert_to_wna16_moe_kernel_format` 的 TRITON 分支把 AutoGPTQ/compressed-tensors 的 K-first int32 打包转置为 N-first 再 view 成 uint8(MoeWNA16 原生 N-first 则 no-op);`_convert_moe_wna16_humming_tensors`/`_MoeWNA16HummingWeightSchema` 适配器把 MoeWNA16 的 uint8 N-first 装箱转成 Humming 的 int32(含 qzeros 双重转置);Marlin GPTQ 路径修复 zero point dtype(params_dtype → int32)与 auto_gptq 来的 ZP 转换。另支持 checkpoint 自带 expert bias:`w13_bias`/`w2_bias` 零初始化、`RoutedExperts.weight_loader` 增加 bias 分支、未加载时卸载以保持与无 bias 路径等价,并修 `fused_moe.get_default_config` 的 `BLOCK_SIZE_M=min(16, next_power_of_2(M))`。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/int_wna16.py: TRITON 后端接入、不兼容判定、三种权重格式转换
vllm/model_executor/layers/fused_moe/experts/triton_moe.py: TritonWNA16Experts 能力声明实现
vllm/model_executor/layers/quantization/auto_gptq.py: qzeros dtype 修复、expert bias 参数与卸载逻辑
vllm/model_executor/layers/fused_moe/routed_experts.py: weight_loader 增加 expert bias 加载分支
vllm/model_executor/layers/fused_moe/fused_moe.py: moe_wna16_cuda 默认配置微调
tests/quantization/test_moe_wna16.py、test_auto_gptq.py: oracle 单测重写/新增
```

TRITON 权重转换核心(节选):

```python
if isinstance(quant_config, (AutoGPTQConfig, QuantizationArgs)):
    # K-first int32 → transpose to N-first, view as uint8
    w13_uint8 = w13.transpose(1, 2).contiguous().view(torch.uint8)
    w2_uint8 = w2.transpose(1, 2).contiguous().view(torch.uint8)
    w13_scale = w13_scale.transpose(1, 2).contiguous()
    w2_scale = w2_scale.transpose(1, 2).contiguous()
else:  # MoeWNA16 uses N-first uint8 weights and scales.
    w13_uint8 = w13.view(torch.uint8)
```

## 性能声明

PR 未声明吞吐/延迟提升(regex 检测为 none);性能意图在于后端选择统一后各量化 MoE 可按平台分派到最优 kernel。给出 gsm8k 5-shot lm-eval 精度对齐(Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 与 TheBloke/Mixtral_7Bx2_MoE-AWQ,batch_size 2048):Qwen baseline flexible-extract/strict-match 为 0.4640/0.3283,PR 为 0.4549/0.3237(略降,±0.0137/0.0129 stderr 内);Mixtral baseline 0.6929/0.6899,PR 0.6952/0.6922(略升)。测试条件未注明显卡型号。pytest 测试通过。

## 验证与回归风险

`tests/quantization/test_moe_wna16.py` 重写为 oracle 单测:后端映射、五组不兼容结构参数化(断言拒绝原因文案)、compressed-tensors 权重转置正确性、`_setup_kernel` 透传所选后端、Humming 张量 repack 与 quant config 获取;`test_auto_gptq.py` 新增零初始化 expert bias 与 bias 加载正确性测试。Test Plan 另含 `tests/compile/h100/test_startup.py::test_model_startup[kimi_k2.5]` 与两模型 gsm8k eval。回归风险:W4A16/W8A16 MoE(GPTQ/AWQ/MoeWNA16 checkpoints)全量走新分派,AutoGPTQ desc_act、AWQ 布局、带 bias checkpoint 等此前可能静默走错的组合现在会被显式拒绝或换后端;qzeros dtype 变更影响显存与加载布局;humming 未安装时导入不再损坏(修复项)。

## 关联

前序: #42647(本 PR 为其修复+更新版);humming 量化导入损坏问题早于 #42647 存在。同分类本批次无其他 MoE oracle 卡。
