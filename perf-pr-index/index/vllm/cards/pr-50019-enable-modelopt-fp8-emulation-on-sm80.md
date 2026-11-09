# PR #50019: Enable ModelOpt FP8 emulation on SM80

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mikekg |
| 合并 | 2026-07-31 |
| 分类 | Quantization \| tier: major |
| Label | ready, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未见「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/50019 |

## 一句话结论

把 ModelOpt FP8 最低算力从 SM89 降到 SM80，A100 上 FP8 emulation 比 BF16 快约 1.24x / 1.29x。

## 动机

ModelOpt FP8 checkpoint 目前在 NVIDIA A100 上被拒：`ValueError: The quantization method modelopt is not supported for the current GPU. Minimum capability: 89. Current capability: 80.`，vLLM 无法选择已有的 weight-only FP8 执行路径。本 PR 使序列化的 ModelOpt `quant_algo=FP8` checkpoint 能在 A100 加载而无需 backend override。正文解释：反直觉地，emulated FP8 比未量化 BF16 更快，因为 bandwidth-limited GEMMs 获得带宽下降；但对 dense/compute-limited GEMM 模型，FP8 emulation 可能因反量化开销落后于原生 BF16，故加了 warning。

## 优化点

把 `ModelOptFp8Config.get_min_capability()` 的最低 CUDA capability 从 SM89 降到 SM80。对 SM80–SM88 的 GPU 额外发一条模型级 warning，说明 FP8 非原生执行、vLLM 将用 emulation、某些模型可能更慢；SM89 及更新 GPU 保持原行为且不发 warning。改动仅放开能力门槛，让现有自动 FP8 fallback 对 dense 与 MoE 层生效，不新增 kernel 或量化算法。

## 关键改动

素材 diff 仅含 1 个文件：`vllm/model_executor/layers/quantization/modelopt.py`，`get_min_capability` 返回值 89 → 80。正文另述 SM80–SM88 会发模型级 warning（diff 可见片段中未见对应代码）。

```text
     @classmethod
     def get_min_capability(cls) -> int:
-        return 89
+        return 80
 
     @classmethod
     def override_quantization_method(
```

## 性能声明

条件：单张 NVIDIA A100 80GB、TP1、vLLM 0.26.0、BF16 activations、FP8 KV cache；模型 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`（revision `f8dc1c0afee92f44417695b4f5ddca9afc95ea58`），BF16 对照 `...-BF16`（revision `2d59de1cbd51c0adf384eb906b766d1aee0e0517`）；GSM8K（`NVIDIA/srt-slurm` commit `cb651682117df129162cd276cf567ec70dc57b36`，1319 题、8-shot、temp 0.6、seed 42、max-tokens 16000）。结果（BF16 → FP8）：Accuracy 61.1069% → 58.3776%（-2.7293 pp）；Invalid 0% → 0%；Evaluation wall time 539.667 s → 435.493 s（-19.30%）；Questions/s 2.4441 → 3.0288（+23.92%）；Total output tokens 588,906 → 611,214（+22,308）；Output tokens/s 1,091.24 → 1,403.50（+28.62%）；Full job wall time 13:31 → 11:56（-11.71%）。正文结论："approximately 1.24x faster by evaluation time and 1.29x faster by output-token throughput, with a 2.73 percentage-point accuracy reduction"，并强调这是单次全数据集运行的性能/精度权衡，不代表所有模型或 workload 都会加速。

## 验证与回归风险

验证：stock FP8 run 在 capability check 处失败、应用 PR 后成功加载并选出自动 FP8 fallback（dense 与 MoE 层），FP8 与 BF16 control 均完成 1319 题无 request failure。可能受影响路径：SM80–SM88 GPU 增加 warning、SM89+ 行为不变（正文明确）、dense/compute-limited GEMM 模型可能变慢（正文自述）、精度下降 2.73 pp。正文未提供新增单测命令。

## 关联

PR 正文未给出同分类其它 PR 编号、前序或后续 PR（Test Plan 明确 "no separate issue has been filed"），无明确线索。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
