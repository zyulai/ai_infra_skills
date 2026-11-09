# PR #55528: [Bugfix][KV Cache][MLA] Align packed block strides for V3.2 sparse MLA

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @200lz |
| 合并 | 2026-09-25 |
| 分类 | Attention / Kernel (tier: major) |
| Label | bug, documentation, performance, new-model, rocm, frontend, intel-gpu, speculative-decoding, ready, ray, torch.compile, ci/build, multi-modality, llama, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia, quantization, vllm-ir, mistral, DSv4, dflash, rust, kimi, k3, scheduler, kv-cache-manager, glm, minimax, inkling, cohere, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供 "file(s) dropped from diff" 说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/55528 |

## 一句话结论

为 V3.2 sparse MLA 补齐 block-stride alignment 契约，CPU 258 passed / 41 skipped，未声称数值加速。

## 动机

V3.2 sparse MLA 没有公布其选定 kernel 所需的 physical block-stride alignment。本 PR 在 producer 侧设置现有的 `block_stride_alignment` 契约，把该字段提升到 `KVCacheSpec`，让 allocator 消费它而无需 MLA 类型检查；HiSparse hot-page 与 stride 约束合并进一个 LCM，避免顺序取整使前一个约束失效。正文说明采用 LucasWilkinson 在 200lz/vllm#1 的最小化建议：六个文件、`+187/-27`，含紧凑的 alignment/merge 测试以及既有 producer/allocator/binder 契约测试；layout gating/threading、admission/auto-fit 重写与 HiSparse PP capacity 工作不在本 PR 范围。

## 优化点

alignment 要求由 backend 与 cache format 选定：flattened-row 路径（含 FlashInfer sparse）与 warmup 共用其判定；SM100 FlashMLA paged FP8/NVFP4 额外要求独立的原生 TMA alignment（`fp8_ds_mla`: 656 bytes；`nvfp4_ds_mla`: 352 bytes），不依赖 flattened-row warmup；既有 BLHNC/BLNHC 支持保留。Scope and validation 说明本 PR 补充 #56254 已有的 alignment propagation 与 pool rounding，提供缺失的 V3.2 producer requirement；上游 #53781 已修复 fused-writer 问题，均不重复。正文明确：stu-cao 先前定性报告 GB200 NVLink NIXL transfer 改善，不声称数值加速；Production MTP/DFlash2 correctness、acceptance 与 long-context 行为仍需重测，不声称生产模型质量评测。

## 关键改动

```text
vllm/v1/kv_cache_interface.py: 把 block_stride_alignment 提升到 KVCacheSpec
vllm/model_executor/layers/attention/mla_attention.py: producer 侧按 backend/cache format 设置 block_stride_alignment 要求
vllm/v1/core/kv_cache_utils.py: allocator 消费该契约，HiSparse 约束用单一 LCM 合并
tests/v1/attention/test_sparse_mla_kv_cache_layout.py: allocation 与 warmup 物理寻址一致的 119 行新增测试
tests/v1/core/test_contiguous_kv_packing.py 与 tests/v1/attention/test_mla_backends.py: 打包与 fixture 修复
```

## 性能声明

PR 正文未给出定量性能数据。正文明确写 "no numerical speedup is claimed"，仅提到 stu-cao 先前对 GB200 NVLink NIXL transfer 的定性改善报告；`258 passed, 41 skipped`（exact suggestion `1d45dcafade7903237c627d941fda7df26341d76` 的 CPU 验证）、临时 merge 后的 `265 tests, 45 skipped`、以及 `88 passed, 310 skipped on CPU` 均为测试计数，非 benchmark。PR 未注明测试条件（无模型/硬件/精度/workload 的性能测量条件）。

## 验证与回归风险

CPU 验证：`258 passed, 41 skipped`；与 main `79468c20ef23227d4e051f807e10fd54fb24e24f` 的干净临时 merge 通过 `265 tests, 45 skipped`（PR 分支本身未与 main 合并）。命令含 `pytest` 七个文件、`ruff check`、`ruff format --check`、`git diff --check`、`pre-commit` 与 `mypy-3.12`，两棵树上全部静态检查通过（默认 mypy 3.10 与显式 3.12），CUDA cases 与 CUDA-only sparse-backend 模块在 CPU 主机被跳过。保留的 producer/native-TMA 逻辑与 sparse layout 测试不变自更早的 B200 验证修订（曾复现 FP8 TMA 断言、修正后通过 packed FP8/NVFP4 用例，早期 B200 bundle 182 passes），属历史结果而非当前 main 的新 GPU 运行；未做新 GPU run。CI 暴露既有 `test_mla_kv_cache_spec_uses_layer_cache_dtype` 的 `SimpleNamespace` fixture 不完整，仅测试侧修复、生产代码不变。风险面：生产 MTP/DFlash2 correctness、acceptance 与 long-context 行为仍需重测。

## 关联

#56254（已有 alignment propagation 与 pool rounding，本 PR 补 V3.2 producer requirement）、#53781（已修复 fused-writer 问题）、200lz/vllm#1（最小化的六文件建议来源）。均为正文明确出现的编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
