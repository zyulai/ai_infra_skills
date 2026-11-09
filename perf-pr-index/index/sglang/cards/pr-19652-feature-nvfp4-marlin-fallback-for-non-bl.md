# PR #19652: [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Godmook |
| 合并 | 2026-04-03 |
| 分类 | Quantization (tier: major) |
| Label | documentation, high priority, quant, sgl-kernel, blackwell, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);6 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/19652 |

## 一句话结论

让 NVFP4 模型在 SM75+ 非 Blackwell GPU 上经 Marlin weight-only FP4 内核运行,并修复 FP4 scale stride 错误导致的 NaN/乱码输出。

## 动机

NVFP4 量化模型(如 `nvidia/Llama-3.1-8B-Instruct-NVFP4`、`nvidia/DeepSeek-V3-0324-FP4`、`mistralai/Minimax-M2.5-NVFP4`)此前在非 Blackwell GPU 上因 `get_min_capability()` 返回 100 直接 crash,迫使 A100/A40/H100/RTX 3090 用户退回精度更低的 AWQ/GPTQ 或改用 vLLM(其已有 Marlin fallback)。关联 issue #19491。

## 优化点

三块机制:(1) 新增 `marlin_utils_fp4.py`(自 vLLM 适配):SM>=75 能力检测、NVFP4 per-group scale 从 FP8-S1E4M3 转 Marlin 要求的 FP8-S0E5M3 格式、global scale 按 FP4/BF16 指数偏置预调整、权重经 `gptq_marlin_repack` 重排为 Marlin tile 布局,linear 与 MoE(逐 expert)各一套 prepare。(2) 修复 Marlin kernel 的 FP4 scale stride bug:原实现按 FP16 2 字节推 scale stride,而 NVFP4 scale 是 FP8 1 字节,导致隔行读 scale、scale-与-weight-group 错位、全量垃圾输出;`s_gl_stride`(prob_n/8→/16)、`s_sh_stride`、`s_tb_groups`、`s_gl_rd`、`s_sh_rd`、`cur_group_id` 共 6 处修正,全部以 `w_type == kFE2M1f` 编译期门控,对 INT4/INT8/FP8 路径零影响;linear(JIT 与 sgl-kernel 两份模板)与 MoE 模板同步对齐 vLLM。(3) MoE kernel 恢复被注释的 `dequant_fp8_scales()`(frag_s 持有 raw float8_e4m3fn 字节,不转会被当 BF16/FP16 解释出 NaN/Inf)并修正 global_scale 应用;FP4 kernel 实例化改由 `SGL_MOE_MARLIN_FP4` 宏 + 独立 JIT module 承载,避免非 FP4 场景编译时间回退。fallback 自动触发:非 Blackwell(或 `SGLANG_FORCE_NVFP4_MARLIN=1` 强制)且 SM>=75 时 repack;compressed-tensors W4A4 NVFP4 与 ModelOpt NVFP4 两套格式的 `get_min_capability` 从 100 降到 75,linear/MoE 均覆盖。

## 关键改动

```text
python/sglang/srt/layers/quantization/marlin_utils_fp4.py: 新文件,FP4 Marlin 工具(repack/scale 转换/linear apply)
python/sglang/jit_kernel/csrc/gemm/marlin/marlin_template.h: FP4 scale stride 6 处修正(kFE2M1f 门控)
python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h: 同步修复 + SGL_MOE_MARLIN_FP4 下启用 dequant_fp8_scales
python/sglang/srt/layers/quantization/modelopt_quant.py: min capability 100→75,linear+MoE 接 fallback
python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py: MoE 走 MARLIN runner
```

## 性能声明

作者定位为正确性修复,未提供与原生 Blackwell FP4 的吞吐对比(目标硬件不同)。E2E 数据:

- `nvidia/Llama-3.1-8B-Instruct-NVFP4` @ A100-40GB:GSM8K(CoT, 8-shot)74.1%(BF16 基线 84.5%,Meta 官方数据,评测协议不同);ARC-Challenge 25-shot acc/acc_norm 56.3%/60.8%;HellaSwag 10-shot 0.5849/0.7944(社区基线,作者注明非官方)。
- `nvidia/Qwen3-30B-A3B-NVFP4` @ A100-80GB:GSM8K 90.1%(invalid 0.000,latency 151.5 s,output throughput 1130.4 token/s);ARC-Challenge 25-shot acc/acc_norm 0.6689±0.0138 / 0.6894±0.0135。修复前同模型输出为乱码文本。
- 内存:权重保持 FP4 压缩(与原生 FP4 同 VRAM),相比 naive BF16 dequant 避免 4 倍权重内存。

## 验证与回归风险

新增 `test_nvfp4_marlin_fallback.py` 6 个单元测试(A40/SM86 全过):能力检测、min capability、scale 转换无 NaN、linear prepare+推理端到端、MoE Marlin GEMM 端到端(此前 NaN)、MoE repack 形状。风险:kernel 改动仅限 `kFE2M1f` 分支(声称对现有 INT4/INT8/FP8 零影响);非 Blackwell 上吞吐低于原生 FP4(启动 warning 注明);与 AWQ/GPTQ 的端到端吞吐对比留待后续 PR。

## 关联

关联 issue #19491(正文);同区域(卡片库中):#28354(FlashInfer CuTe DSL NVFP4 MoE)、#33115(ModelOpt FP4 online MoE weight quant)、#28291(AMD MXFP4 online requant)。
