# PR #23696: [Kernel][B200] `mxfp4` fused cutlass moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @djmmoss |
| 合并 | 2025-09-11 |
| 分类 | MoE \| tier: major |
| Label | documentation, performance, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾无 dropped from diff 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/23696 |

## 一句话结论

为 gpt-oss 加入 cutlass fused MoE，Blackwell 高并发下 mxfp8 x mxfp4 吞吐较 trtllm 高约 7%。

## 动机

PR 正文 Purpose：为 Hopper 上的 gpt-oss 增加 cutlass fused moe bf16xmxfp4、为 Blackwell 上的 gpt-oss 增加 mxfp8xmxfp4。正文明确指出 Hopper 相比 Triton 存在性能回退（团队正在修复，修好后会推到 flashinfer），并说明 Blackwell 的 mxfp8 x mxfp4 cutlass 在高并发下相比 trtllm 约有 7% 更高吞吐，但低并发下仍建议使用 trtllm backend。正文未描述改动前路径的具体瓶颈数据。

## 优化点

新增 cutlass/flashinfer 的 mxfp4 fused MoE 后端并接入现有选择逻辑：Hopper 走 bf16 x mxfp4、Blackwell 走 mxfp8 x mxfp4。启用由环境变量 `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1` 控制（见 Blackwell Accuracy 命令）。改动落在 `vllm/model_executor/layers/quantization/mxfp4.py`（mxfp4 量化与 cutlass MoE 路径）、`vllm/model_executor/layers/fused_moe/layer.py`（fused MoE backend 选择）、`vllm/envs.py`（开关定义）、`vllm/model_executor/warmup/kernel_warmup.py`（kernel warmup 配套）；`tests/kernels/moe/test_mxfp4_moe.py` 新增 sm90 + flashinfer 的可用性判定与 cutlass fused MoE 测试。正文未说明更细的 kernel 层面机制。

## 关键改动

```text
vllm/model_executor/layers/quantization/mxfp4.py: cutlass/flashinfer mxfp4 MoE 量化与 kernel 路径
vllm/model_executor/layers/fused_moe/layer.py: fused MoE backend 选择接入
vllm/envs.py: 新增 VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS 开关
vllm/model_executor/warmup/kernel_warmup.py: kernel warmup 配套
tests/kernels/moe/test_mxfp4_moe.py: 新增 HOPPER_MXFP4_BF16_AVAILABLE 与 test_flashinfer_cutlass_mxfp4_fused_moe
```

```text
+from vllm.utils.flashinfer import has_flashinfer
+HOPPER_MXFP4_BF16_AVAILABLE = (current_platform.is_cuda()
+                               and current_platform.is_device_capability(90)
+                               and has_flashinfer())
```

## 性能声明

Hopper（`vllm serve gpt-oss-120b --max-model-len 32768 -tp 8`，`vllm bench serve --dataset-name sharegpt`）：gpt-oss-120b FlashInfer Output token throughput 7404.72 tok/s、Total 15491.84、Mean TTFT 3449.17 ms，对应 Triton 10210.09 / 21307.23 / 4236.52 ms；gpt-oss-20b（-tp 2）FlashInfer 5831.00、Triton 12633.59 tok/s——即正文所述的 Hopper 相比 Triton 回退。Blackwell（random dataset，random-input-len 1024 / random-output-len 1024）：gpt-oss-20b 并发 1 时 trtllm 316 vs cutlass 197 OTPS（-60.3%），并发 512 时 22018 → 23330（+5.6%）、并发 1024 时 23195 → 24629（+5.8%）；gpt-oss-120b 并发 1024 时 15592 → 16221（+4.0%）、并发 1152 时 15369 → 16424（+6.9%），对应正文 "around 7% higher throughput compared to trtllm at high concurrency"。单元测试：Hopper 8 passed / 59 skipped，Blackwell 56 passed / 11 skipped（300.24s）。Accuracy（gpqa，reasoning effort low；参考值 gpt-oss-20b 0.58、gpt-oss-120b 0.68）：Hopper 0.5732323232323232 / 0.0.6609848484848485，Blackwell 0.6660353535353535 / 0.5612373737373737。

## 验证与回归风险

Test Plan 分 Hopper / Blackwell 两组给出 unit test（`pytest tests/kernels/moe/test_mxfp4_moe.py`）、gpqa evals 与 `vllm bench serve` 性能命令；Test Result 给出两组通过数、精度与吞吐表。回归风险：正文明确 Hopper 相比 Triton 有性能回退且待修复、Blackwell 低并发仍建议 trtllm；正文未说明对其他模型、精度或非 NVIDIA 硬件的影响，也未给出 backend 选择的最终结论（作者请 reviewer 定夺）。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
