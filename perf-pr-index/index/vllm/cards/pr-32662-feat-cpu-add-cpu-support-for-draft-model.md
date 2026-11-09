# PR #32662: feat(cpu): add CPU support for draft model speculative decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ganeshr10 |
| 合并 | 2026-04-10 |
| 分类 | Speculative Decoding \| tier: major |
| Label | speculative-decoding, ready, ci/build, v1, cpu, verified |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 2 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/32662 |

## 一句话结论

为 CPU 上草稿模型投机解码补齐 Triton kernel 的 PyTorch fallback，使该路径可用。

## 动机

PR 之前 CPU 上 draft model speculative decoding 依赖的多个 Triton kernel 没有 CPU 实现，导致该能力无法在 CPU 运行。PR 目的即通过为这些 Triton kernel 增加 PyTorch fallbacks，启用 CPU 上的 draft model 投机解码。

## 优化点

在 `csrc/cpu/spec_decode_utils.cpp` 新增 CPU 版 eagle 系列 kernel（eagle_prepare_inputs_padded、eagle_prepare_next_token_padded、copy_and_expand_eagle_inputs、eagle_step_slot_mapping_metadata、rejection_greedy/random_sample、expand、sample_recovered_tokens、slot mapping 等）。这些 kernel 以 `#pragma omp parallel for` 按 request 并行遍历，并由 `csrc/cpu/torch_bindings.cpp` 通过 `TORCH_LIBRARY_EXPAND` 注册为 torch 自定义算子。`vllm/utils/cpu_triton_utils.py` 提供对应 PyTorch/torch.ops 包装并做 int64 dtype 归一化。`vllm/v1/worker/cpu_model_runner.py` 的 `_postprocess_triton` 在 CPU 启动时把这些 fallback 绑定到 `vllm.v1.spec_decode.eagle`、`vllm.v1.spec_decode.utils` 与 `vllm.v1.sample.rejection_sampler`。

## 关键改动

```text
csrc/cpu/spec_decode_utils.cpp: 新增 409 行 CPU 投机解码 kernel 实现
vllm/utils/cpu_triton_utils.py: PyTorch fallback 包装与 dtype 处理
vllm/v1/worker/cpu_model_runner.py: _postprocess_triton 把 fallback 绑定到 spec_decode / rejection_sampler
csrc/cpu/torch_bindings.cpp: 注册 torch 自定义算子
vllm/v1/spec_decode/eagle.py、vllm/v1/spec_decode/utils.py: 入口适配
```

```python
+        import vllm.v1.sample.rejection_sampler
+        import vllm.v1.spec_decode.eagle
+        import vllm.v1.spec_decode.utils
+
+        vllm.v1.spec_decode.eagle.eagle_prepare_inputs_padded_kernel = (
+            cpu_tl.eagle_prepare_inputs_padded_kernel
+        )
+        vllm.v1.spec_decode.eagle.eagle_prepare_next_token_padded_kernel = (
+            cpu_tl.eagle_prepare_next_token_padded_kernel
+        )
```

## 性能声明

测试条件：target Qwen/Qwen3-32B、draft Qwen/Qwen3-1.7B、dtype bfloat16、硬件 AMD EPYC 9654 96-Core Processor；workload 为 `vllm serve` + `vllm bench serve`（数据集 philschmid/mt-bench、80 prompts、max-concurrency 1/100、temperature 0.0）、speculative_config method=draft_model、num_speculative_tokens=3。PR 正文未给出定量性能数据（结果以两张截图呈现，仅给定性 Performance Highlights：除高 batch 外 TTFT 相当、SD 的 TPOT 优于自回归解码、各并发下 output/total token 吞吐更高）。

## 验证与回归风险

正文给出 serve/bench 命令与定性 Performance Highlights，未给出 pytest 通过数或具体性能数值；测试命令中 `vllm bench serve` 采用 1 与 100 两档并发分别对比，正文的 Essential Elements checklist 各项均未勾选。回归风险：结果以截图呈现、未量化；受影响路径为 CPU 后端上的 draft-model 投机解码（eagle 与 rejection sampler 相关算子）；正文未声明对其他硬件的影响。

## 关联

无

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
