# PR #28291: [AMD][MXFP4] Reland "Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs"

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @fxmarty-amd |
| 合并 | 2026-07-21 |
| 分类 | Quantization (tier: major) |
| Label | documentation, quant, run-ci, bypass-fastfail, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);6 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/28291 |

## 一句话结论

Reland:AMD GPU(CDNA4/MI355X)上把 FP8 checkpoint 在线重量化为 MXFP4,加载期渐进完成并以 meta device 避免额外显存开销。

## 动机

原 PR #18182 因 `model_runner.py` 错误的合并冲突(波及 modelopt 量化路径,未纳入 #27284 的修复)被 #28213 revert;当时 CI 的 fast-fail 机制跳过了后续 workflow,导致 `test_modelopt_fp8.py` 根本没跑,问题漏检。本 reland 与原 PR 唯一差异是移除 `model_runner.py` 中的在线量化 logging 逻辑。

## 优化点

机制上是一条加载期量化流水线:`--quantization quark_mxfp4` 作用在 FP8 checkpoint 上时,(1) `QuarkConfig.from_config` 识别 `quant_method=fp8 + requantization_method=quark_mxfp4 + dynamic activation` 组合,构造带 `dequantization_config=Fp8Config` 的 QuarkConfig;被 exclude 的层不再回退 BF16,而是保持原 FP8 精度(返回 `Fp8LinearMethod`),exclude 列表按 `amd/DeepSeek-V3.2-mxfp4` 官方配置适配 deepseek_v3/v32。(2) 权重加载时先在 meta device 上创建 FP8 weight 参数(避免加载期显存峰值,最终常驻的是更小的 MXFP4),自研 weight loader 负责:加载 FP8 weight 与 `weight_scale_inv` → 用 `CopyNumelCounter`(TorchDispatchMode)与锁跟踪所有 shard(q/k/v 等)到齐 → `block_quant_dequant` 反量化 FP8→BF16 → 重量化 BF16→MXFP4,渐进式逐层完成。(3) 为复用加载逻辑,`Fp8LinearMethod.create_fp8_weight_` 与 `Fp8MoEMethod.create_fp8_moe_weight_` 重构为 staticmethod;新增 `dequantization.py`(含 ue8m0 scale 逆变换的 Blackwell TODO)。

## 关键改动

```text
python/sglang/srt/layers/quantization/quark/quark.py: FP8→MXFP4 requant 配置接入与 exclude 层 FP8 保持
python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py: meta device FP8 权重 + 渐进 FP8→MXFP4 weight loader
python/sglang/srt/layers/quantization/fp8.py: create_fp8_weight_/create_fp8_moe_weight_ 重构为可复用 staticmethod
python/sglang/srt/layers/quantization/dequantization.py: 新文件,FP8 block 反量化(含 ue8m0 处理)
python/sglang/srt/layers/quantization/online_quantization.py: 新文件,CopyNumelCounter 跟踪 copy_ 元素数
```

## 性能声明

PR 正文未提供文字化 benchmark 数字;性能收益以两张 AMD 官方截图呈现,具体数值 PR 正文未注明。定性收益:MXFP4 GEMM 替代 BF16 GEMM(需 CDNA4 硬件 FP4 支持,如 MI355X),且权重常驻显存降为 MXFP4 尺寸。

## 验证与回归风险

MI355X:`test_quark_mxfp4.py` 的 TestOnlineQuantizationMemoryLoadDense/MOE、TestFP8ToMXFP4DenseTP1/TP2、TestFP8ToMXFP4MOETP1 通过;H100:`test_modelopt_fp8.py` 与 `test_modelopt_loader.py` 通过(验证上次 revert 的问题已修复);本 PR CI Base/Extra 均通过。风险:requant 可能带来精度损失(启动日志明确警告需重新评估);当前仅支持 Fp8Config 源、block_quant=True、dynamic activation;`block_quant_dequant` 的 `repeat_interleave` 实现被注释标记为内存低效(block 128x128 时 scale 需 *16384 内存)。

## 关联

前序与因果链(正文明确):原 PR #18182 → revert #28213;源头 #18005、其 bug 修复 #27284。同区域(卡片库中):#28354(FlashInfer CuTe DSL NVFP4 MoE)、#33115(ModelOpt FP4 online MoE quant)、#44865(ComfyUI online INT8)。
