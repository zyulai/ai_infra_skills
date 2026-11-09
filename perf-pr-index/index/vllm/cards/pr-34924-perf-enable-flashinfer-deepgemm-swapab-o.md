# PR #34924: [Perf] Enable FlashInfer DeepGEMM swapAB on SM90 by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2026-02-24 |
| 分类 | Attention / Kernel（tier: major） |
| Label | performance, ready, ready-run-all-tests |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 2 个文件：`vllm/envs.py`（默认值翻转）与 `tests/compile/fusions_e2e/test_tp1_quant.py`（跳过不兼容融合测试） |
| 链接 | https://github.com/vllm-project/vllm/pull/34924 |

## 一句话结论

把 `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` 默认打开，SM90 上 linear 层默认走 FlashInfer FP8 block-scale GEMM（swapAB）：MiniMax-M2.5 TP4/4xH100 输出吞吐 896→968 tok/s（+8.1%），TPOT 8.48→7.80ms。

## 动机

#29213 引入 FlashInfer 的 FP8 block-scale GEMM（TensorRT-LLM kernels，要求 SM90+ Hopper）swapAB 路径，但当时没有设为默认——而该路径在低 batch 下约有 +10% 提升，即使在大 batch 下也追平 DeepGEMM。作者决定默认启用并验证是否破坏现有功能（"Let's see if we break anything putting this on by default"）。

## 优化点

机制上是默认值/开关翻转而非新 kernel：`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER` 从 False 翻为 True（默认 "0"→"1"），使 SM90+ 上的 FP8 block-scale GEMM linear 层默认选择 FlashInfer/TRT-LLM 的 swapAB 实现。配套调整一处融合测试假设：FlashInfer block FP8 GEMM 内部自带量化，无法与其他 op 融合，`test_tp1_fp8_fusions` 在 `use_deepgemm` 且 `is_flashinfer_fp8_blockscale_gemm_supported()` 时直接 skip。

## 关键改动

```text
vllm/envs.py: VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER 默认 False→True（SM90+ FlashInfer FP8 block-scale GEMM）
tests/compile/fusions_e2e/test_tp1_quant.py: FlashInfer block FP8 GEMM 场景跳过（内部量化，不可融合）
```

```python
-    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = False
+    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True
```

## 性能声明

测试条件：4xH100，`MiniMaxAI/MiniMax-M2.5 -tp=4 --load-format=dummy`，`vllm bench serve` random 数据集 input 1024 / output 512、64 prompts、并发 8。Before→After：Output token throughput 896.22→968.47 tok/s（+8.1%）；Request throughput 1.75→1.89 req/s；Total token throughput 2688.66→2905.41 tok/s；Mean TPOT 8.48→7.80ms（−8.0%）；Mean ITL 8.49→7.79ms。Mean TTFT 略升 234.29→240.44ms（+2.6%）。PR 正文另引述：低 batch 下约 +10% 提升、大 batch 下与 DeepGEMM 持平（出自 #29213 的说法）。

## 验证与回归风险

测试：`tests/compile/fusions_e2e/test_tp1_quant.py` 增加 skip 分支；PR 标注 ready-run-all-tests（正文 Test Plan 为空，未列系统性测试矩阵）。风险面：SM90+ 全体用户的 FP8 block-scale linear 默认路径改变（可经环境变量回退为 0）；inductor 融合与该 kernel 不兼容（测试中显式 skip）；基准中 TTFT 有小幅上升；Blackwell（is_blackwell）分支另有 DeepGEMM 量化匹配 TODO 与本开关叠加的行为（测试中一并跳过）。

## 关联

前序：#29213（引入 swapAB 路径）。索引内同区域：#27284（SM100 CUTLASS FP8 GEMM 加 swapAB 优化）。PR 正文未列其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
