# PR #47124: [Quantization][Autoround][XPU] Add W4A16(moe) / MXFP4(linear/moe) Support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @lkk12014402 |
| 合并 | 2026-07-30 |
| 分类 | MoE(tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, verified, mistral, rust |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 1 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/47124 |

## 一句话结论

让 XPU(Intel GPU)能部署 AutoRound 格式量化模型:新增 W4A16 MoE 路由与 MXFP4 linear/MoE scheme,dense 与 MoE 均支持,MoE kernel 按设备自动选 CUTLASS / XPU 原生 / Marlin。

## 动机

AutoRound 格式的量化模型(含 W4A16 与 MXFP4 方案,dense 与 MoE)此前无法在 XPU 上部署:vLLM 的 INC(auto-round)量化栈只有 MXFP8 与 WNA16 scheme,XPU 上 MoE 量化被直接跳过(`is_xpu() or is_cpu()` 一律走 UnquantizedFusedMoEMethod),Marlin 能力检查也在 XPU 上不可达。本 PR 补齐 XPU 的低比特权重部署路径,使 4-bit 模型显存占用与吞吐优化覆盖 Intel GPU。

## 优化点

机制上:(1) data_type 判定改为按 `"mx_fp"` 子串匹配(兼容 auto-round 的 `mx_fp4`/`mx_fp4e2m1` 等变体),新增 `INCMxfp4Scheme`,分发到 `INCMxfp4LinearMethod` 与 `INCMxfp4MoEMethod`。(2) MXFP4 linear:权重按 E2M1 每字节两值打包,scale 为 E8M0(group_size=32,无全局 scale),kernel 经 `init_mxfp4_linear_kernel` 按平台选择(CUDA 上 FlashInfer/Marlin,XPU 上 `fp4_gemm`)。(3) MXFP4 MoE:后端选择镜像 compressed-tensors 的做法——支持 CUTLASS 的 GPU 用 `CutlassExpertsMxfp4`(真 W4A4,加载后把 scale swizzle 成 CUTLASS tiled 布局),XPU 走共享 oracle 选择原生 kernel(packed 布局直通,无需 repack/transpose),其他设备回退 Marlin weight-only FP4。(4) W4A16 MoE:XPU 不再被排除,`_resolve_gptq_moe`/`_resolve_awq_moe` 去掉 `check_marlin_supported` 的能力门控(仅保留 `check_moe_marlin_supports_layer` 形状检查),由 AutoGPTQMoEMethod 经 WNA16 oracle 选择后端(CUDA 上 Marlin,XPU 上 `XPUExpertsWNA16`)。

## 关键改动

```text
vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp4_scheme.py: 新增 MXFP4 scheme 分发
vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp4_linear.py: MXFP4 linear method(E2M1 打包 + E8M0 scale)
vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp4_moe.py: MXFP4 MoE method(CUTLASS/XPU/Marlin 三后端)
vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py: W4A16 MoE 在 XPU 上启用,Marlin 门控改形状检查
vllm/model_executor/layers/quantization/inc/inc.py / config_parser.py: mx_fp 家族 data_type 识别与校验
```

关键片段(MoE 后端选择):

```python
self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
if self.use_cutlass_mxfp4:
    self.experts_cls = CutlassExpertsMxfp4
elif current_platform.is_xpu():
    self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
else:
    self.experts_cls = MarlinExperts  # weight-only FP4 fallback
```

## 性能声明

PR 正文未提供吞吐/延迟数字,只给出 lm-eval piqa 精度结果(XPU,tensor_parallel_size=4,max_model_len 8192,max_num_batched_tokens 32768,bfloat16,batch_size 8;MXFP4 两组 enforce_eager=True):

- W4A16 dense(Qwen3-8B-w4g128):piqa acc 0.7639 ± 0.0099,acc_norm 0.7737 ± 0.0098。
- W4A16 MoE(Qwen3-30B-A3B-w4g32):piqa acc 0.7987 ± 0.0094,acc_norm 0.7982 ± 0.0094。
- MXFP4 dense(Qwen3-8B-mxfp-w4g32):piqa acc 0.7203 ± 0.0105,acc_norm 0.7280 ± 0.0104。
- MXFP4 MoE(Qwen3-30B-A3B-mxfp-w4g32):piqa acc 0.7595 ± 0.0100,acc_norm 0.7688 ± 0.0098。

## 验证与回归风险

`tests/quantization/test_auto_round.py` 大幅扩展:新增 Qwen3-1.7B MXFP4、Qwen3-30B-A3B W4A16/MXFP4 三个模型参数(要求 CUDA/XPU),以及 config 解析、scheme 路由(W4A16→GPTQ MoE、MXFP4→MXFP4 MoE)、权重打包/加载、kernel 构建的单元测试。回归风险:Marlin 能力检查被移除后,原本依赖 `check_marlin_supported` 拒绝的层现在可能走 AutoGPTQMoEMethod 路径;MXFP4 MoE 在无原生 FP4 计算的设备上回退 Marlin weight-only(PR 注明对 compute-heavy workload 可能降速);mx_fp 子串匹配放宽了 data_type 校验。

## 关联

同族:compressed-tensors 的 `CompressedTensorsW4A4Mxfp4MoEMethod`(本 PR 后端选择逻辑对其镜像)。索引内同区域(MoE / Quantization)卡片:#14447(Marlin WNA16 kernel)、#25609(FBGEMM NVFP4 dense)、#26107(cuDNN FP4 GEMM)、#23809(ARM 4bit fused MoE)、#2453(DeepSeekMoE fused MoE)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
