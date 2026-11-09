# PR #23795: :sparkles: [llm][npu][quant] Add W4A4 MXFP4 quantization support for Qwen3 Dense on Ascend NPU

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @TallMessiWu |
| 合并 | 2026-07-17 |
| 分类 | Quantization(tier: major) |
| Label | documentation, quant, npu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 未提供 +add/-del 行数与文件数;diff 内容可用) |
| 链接 | https://github.com/sgl-project/sglang/pull/23795 |

## 一句话结论

Ascend NPU(950/A5)上为 Qwen3 dense 模型新增 W4A4 MXFP4 量化:吞吐较 BF16 +41%(离线 +49%),E2E 延迟约 −29%。

## 动机

继续 #21584 追踪的 Ascend NPU 量化工作,补齐 Qwen3/Qwen3.5 dense LLM 的真 W4A4(4-bit 权重 + 4-bit 激活)能力。前置的 #22352(W8A8 MXFP8)与 #23650(W4A8 MXFP4)已合并,本 PR 只含 W4A4 增量并复用其基础设施(`_NPULinearMethodBase`、`_get_float4_e2m1fn_x2_dtype()`、`npu_format_cast`、`ModelSlimLinearScheme` 等)。精度上,更早的单层 online RTN(UE8M0 2 的幂 block scale)损失过大,greedy 长文本解码会退化为永不输出 EOS 的循环,因此 online 路径改用 dual-level。

## 优化点

- Online(`--quantization mxfp4`,NPU 后端):新 `Mxfp4W4A4Config` 分发到 `NPUDualLevelMXFP4LinearMethod`。加载时 BF16/FP16 权重经 `npu_dynamic_dual_level_mx_quant` 量化为 packed FP4 + 细粒度 FP8(E4M3)L0 block scale + 粗粒度 L1 scale,权重 cast 成 FRACTAL_NZ;推理时激活同样动态量化,matmul 走 `npu_dual_level_quant_matmul`。dual-level 的细 L0 scale 显著改善 per-block dynamic range 捕获,消除单层 RTN 的退化。
- Offline(msmodelslim 预量化权重,`--quantization modelslim`):`W4A4_MXFP4` checkpoint 经 `quant_model_description.json` 自动检测,由 `ModelSlimMXFP4Scheme` 委托 `NPUSingleLevelMXFP4OfflineLinearMethod`:fp8-container FP4 权重重打包为 `float4_e2m1fn_x2` 并转置,UE8M0 scale 重排成 3D,matmul 走 `npu_quant_matmul(group_sizes=[1,1,32])`。checkpoint 只有单层 scale,故 offline 与 online 用不同 matmul kernel。
- 设备分发仅在 `layers/quantization/__init__.py` 的 `is_npu()` 块内注册 `"mxfp4" → Mxfp4W4A4Config`(镜像 `GPTQAscendConfig`),GPU/CPU 上同名 flag 仍解析到上游 OCP `Mxfp4Config`(MoE),无冲突。所有 NPU op 走 `torch.ops.npu.*`,无顶层 `torch_npu` import,模块在 CUDA/CPU/AMD/XPU CI 下可导入。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py: 新增 NPUDualLevelMXFP4LinearMethod(online)与 NPUSingleLevelMXFP4(Offline)LinearMethod
python/sglang/srt/layers/quantization/npu_mxfp4_w4a4.py: 新增 Mxfp4W4A4Config("mxfp4",NPU 分发,MoE 层回退 unquantized)
python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp4.py: 新增离线 ModelSlimMXFP4Scheme(fp8-container FP4 权重)
python/sglang/srt/layers/quantization/__init__.py: is_npu() 块内注册 "mxfp4" → Mxfp4W4A4Config
python/sglang/srt/layers/quantization/modelslim/modelslim.py: linear_quant_schemes 增加 ("W4A4_MXFP4", ModelSlimMXFP4Scheme)
```

## 性能声明

PR 注明在 Ascend A5(Ascend 950)硬件上端到端验证,对比 BF16 baseline;PR 未注明 benchmark 所用具体模型规模、数据集与请求参数。数字(均值,min–max 区间见原文):

- Online W4A4(dual-level):Request throughput 5.29→7.45 req/s(+40.8%),Total token throughput 3195.54→4507.91 tok/s(+41.1%),Output token throughput +41.1%;Mean E2E latency 5753.31→4104.58 ms(−28.7%),Mean TTFT 231.42→157.01 ms(−32.2%),Mean TPOT 26.62→18.88 ms(−29.1%)。
- Offline W4A4(single-level):Request throughput +48.8%(7.87 req/s),Total/Output token throughput 各 +48.9%;Mean E2E −32.3%(3893.88 ms),Mean TTFT −34.9%,Mean TPOT −33.0%。offline 略快是因为 dual-level 用部分吞吐换更细的 L0 scale。
- 精度(GSM8K,各 3 次运行):BF16 95.32;Online W4A4 93.76(−1.56);Offline W4A4 93.48(−1.84);online 略优于 offline。

## 验证与回归风险

PR 正文给出吞吐/延迟与 GSM8K 结果,但未列新增单元测试;pack 中的 CI 状态显示 Latest PR Test (Base)/(Extra) 均为失败(红叉)。回归风险:dual-level 依赖 `DualLevelQuantBatchMatmul` op,仅 Ascend 950 (A5) 可用,A2/A3 不支持;FusedMoE 层的 W4A4 未实现,直接回退 unquantized(BF16/FP16)并打 warning;非 NPU 设备上该 config 对 LinearBase 抛 `NotImplementedError`;packed FP4 dtype 必须取自 `torch_npu.float4_e2m1fn_x2` int enum(近期 op-plugin 构建会拒绝 torch dtype 对象);权重转置视图不能 `.contiguous()`,否则 block-scale 对齐被破坏。

## 关联

追踪:#21584(Ascend NPU MXFP8/MXFP4 for Qwen3 dense);前序:#22352(W8A8 MXFP8)、#23650(W4A8 MXFP4),均已合并。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
