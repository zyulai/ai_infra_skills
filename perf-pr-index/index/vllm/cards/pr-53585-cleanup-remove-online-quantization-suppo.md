# PR #53585: [Cleanup] Remove online quantization support in `fp8.py` in favor of online shorthands

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-09-24 |
| 分类 | Quantization(tier: major) |
| Label | documentation, intel-gpu, speculative-decoding, ready, torch.compile, ci/build, qwen, quantization |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 2 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/53585 |

## 一句话结论

废弃 `quantization="fp8"` 的在线量化路径:统一重定向到 online shorthand `fp8_per_tensor`,从 `fp8.py` 删除在线分支并移除与之重复的慢测试,收敛 FP8 在线量化实现到单一入口。

## 动机

vLLM 已有独立的在线量化模块 `vllm/model_executor/layers/quantization/online/fp8.py` 及 `ONLINE_QUANT_SHORTHAND_NAMES`(如 `fp8_per_tensor`,见在线量化文档),但旧的 `quantization="fp8"` 仍会在 `Fp8Config` 内部走 `is_checkpoint_fp8_serialized=False` 分支、实例化 `Fp8PerTensorOnlineLinearMethod/OnlineMoEMethod`,形成两套并存的在线 FP8 实现。本 PR 是 #32189 与 #45463 的后续,把旧入口废弃并重定向到 shorthand,同时清理 `test_fp8.py` 中与 `test_online.py` 重复且耗时的在线量化测试(在 (MI300) Quantization CI 组中单个测试长达 70.85s/55.51s/54.31s 等)。

## 优化点

机制上:(1) `Fp8Config` 的 `is_checkpoint_fp8_serialized` 默认改为 True,构造时若传 False 直接抛 ValueError 并提示改用 `--quantization fp8_per_tensor`;`get_quant_method` 删除 online 分支,Linear/RoutedExperts 一律返回离线 `Fp8LinearMethod`/`Fp8MoEMethod`。(2) 兼容层:`weight_utils.get_quant_config` 在找不到量化 config 文件且 `quantization=="fp8"` 时,`warning_once` 提示弃用,并把参数经 `resolve_quantization_config("fp8_per_tensor", online_args)` 解析后包成 `OnlineQuantizationConfig` 返回——旧显式 override(如 `ignore`、`linear="fp8_per_block"`)按 shorthand 同样的默认值保留。(3) 模型适配:`qianfan_ocr._patch_quant_config` 同时识别 `OnlineQuantizationConfig` 中三种静态 FP8 linear 量化类型,保留 vision/projector/head 层排除。(4) 测试与 CI:`test_fp8.py` 删除三个重复慢测试,`test_online.py` 扩展 legacy 别名与视觉层排除测试,多处测试与 Intel CI 命令改用 `fp8_per_tensor`;文档同步更新。

## 关键改动

```text
vllm/model_executor/layers/quantization/fp8.py: Fp8Config 移除在线量化分支(is_checkpoint_fp8_serialized=False 抛错)
vllm/model_executor/model_loader/weight_utils.py: legacy "fp8" 重定向到 fp8_per_tensor 的 OnlineQuantizationConfig
vllm/model_executor/models/qianfan_ocr.py: OnlineQuantizationConfig 下保留 vision 层忽略
tests/quantization/test_fp8.py: 删除重复慢测试(test_online_quantization / peak_mem / load_format_dummy)
tests/quantization/test_online.py: 新增 legacy 别名与 shorthand 选择测试
```

关键片段(重定向):

```python
if model_config.quantization == "fp8":
    logger.warning_once(
        "--quantization fp8 is deprecated for online quantization; "
        "use --quantization fp8_per_tensor instead.")
    fp8_args = resolve_quantization_config("fp8_per_tensor", online_args)
    return OnlineQuantizationConfig(args=fp8_args)
```

## 性能声明

PR 正文未提供推理性能 benchmark。文中数字均为 CI 测试耗时动机(被删除的旧在线量化测试在 (MI300) Quantization 组的 call 时间:test_online_quant_peak_mem 70.85s、test_online_quantization[True-False-auto] 55.51s、test_online_quant_load_format_dummy 54.31s 等);文档维持既有结论"online dynamic 模式下延迟改进有限"(动态 per-tensor activation scale,每次 forward 计算激活 min/max)。

## 验证与回归风险

新测试:`test_legacy_fp8_online_quantization_uses_per_tensor_shorthand`(验证 override 保留与弃用告警)、`test_qianfan_online_fp8_keeps_vision_layers_unquantized`(三种 FP8 shorthand 下 vision/lm_head/mlp1 层不量化)。验证记录:MI350 上 test_online.py 通过(5 passed/4 skipped);B300 上 legacy FP8 测试与 DeepSeek scale-loading 用例通过,4 个 online FP8 model-forward 用例因本地编译扩展不匹配(`moe_align_block_size` 参数个数)未能运行;CPU 目标下 44 passed;B300 上 `fp8` 与 `fp8_per_tensor` 的 Qwen3-0.6B 贪心 12-token 生成 token ID 完全一致(111 个量化层,含 27/28 attention o_proj)。回归风险:直接构造 `Fp8Config(is_checkpoint_fp8_serialized=False)` 的调用方会报错(PR 称 breaking changes 为 None,因 CLI/LLM 入口已重定向);Qianfan OCR 全量精度与 Intel XPU 实测未跑。

## 关联

前序:#32189、#45463(online shorthand 体系)、#53291(慢测试清理上下文)。索引内同区域(Quantization)卡片:#20358 不属同区;更相关为 #18844(FP8 quant 调优)、#22895(FlashInfer FP8 per-tensor GEMM)、#18778(SM100 FP8 CUTLASS kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
