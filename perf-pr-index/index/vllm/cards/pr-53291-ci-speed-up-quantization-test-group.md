# PR #53291: [CI] Speed up quantization test group

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-09-01 |
| 分类 | Quantization \| tier: major |
| Label | documentation, ready, quantization |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 9 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/53291 |

## 一句话结论

量化测试组二次运行在 local MI350 上由 0:56:09 降到 0:21:56。

## 动机

PR 正文 Purpose 指出本 PR 用于加速 quantization test group，修复 issue #51142。正文贴出 main + MI355 的 slowest 25 durations，显示 `tests/quantization/test_quark.py::test_nvfp4_wikitext_correctness[1]` 122.48s、`test_ocp_mx_wikitext_correctness` 103.76s / 97.75s、`test_cpu_offload.py::test_cpu_offload_fp8` 87.37s 等大量测试单调用耗时在 40s 以上，量化测试组整体过慢；原因是简单集成测试也去实例化 vllm runner，而实例化与清理 runner 对这类测试太慢。

## 优化点

正文列出 Common changes to many tests：一是在测试不需要时避免实例化 vllm runner，改用 `load_model_without_vllm_runner` 等辅助函数并配合 `set_current_vllm_config`、`set_forward_context`、`dist_init`、`workspace_init`；二是用 `hf_override` 把 `num_hidden_layers` 改小（e.g. 3）以减少加载与编译开销。三个 accuracy tests 被移动到 `.buildkite/test-amd.yaml` 的指定行；新增 gsm8k 配置 `Qwen-1.5-MOE-W-MXFP4-A-MXFP6.yaml`（accuracy_threshold 0.4405）、`Qwen-1.5-MOE-W-MXFP6-A-MXFP6.yaml`（0.5049）与 `Qwen3-30B-A3B-NVFP4-quark.yaml`，并加入 `models-gfx950-large.txt` / `models-mi3xx.txt`。正文 Remaining 列出 `test_cpu_offload_*`、logprobs tests、`test_online_quant_peak_mem` 留待其它 PR。

## 关键改动

```text
tests/quantization/test_auto_awq.py、test_auto_gptq.py、test_compressed_tensors.py: 用 load_model_without_vllm_runner 替代 vllm_runner，去掉不必要的 runner 实例化
tests/quantization/test_fp8.py、test_fp8_per_channel.py、test_gptq_dynamic.py、test_lm_head.py、test_modelopt.py: 同类改造并配合 hf_override 缩小 num_hidden_layers
tests/evals/gsm8k/configs/Qwen-1.5-MOE-W-MXFP4-A-MXFP6.yaml、Qwen-1.5-MOE-W-MXFP6-A-MXFP6.yaml: 新增 50 题、5-shot、server_args --enforce-eager --max-model-len 4096 的 gsm8k 配置
tests/evals/gsm8k/configs/models-gfx950-large.txt、models-mi3xx.txt: 把三个 accuracy 配置挂到对应机型列表
```

```diff
-from tests.quantization.utils import is_quant_method_supported
+from tests.quantization.utils import (
+    is_quant_method_supported,
+    load_model_without_vllm_runner,
+)
+from vllm.config import set_current_vllm_config
+from vllm.forward_context import set_forward_context
+from vllm.model_executor.layers.attention import Attention
+from vllm.platforms import current_platform
```

## 性能声明

PR 正文给出的是 CI 测试耗时（非推理吞吐/延迟）。条件：After all models are cached / compilation done (i.e., second run)，本地与 hosted 的 MI350 / MI300，命令 `VLLM_TEST_FORCE_LOAD_FORMAT=auto HIP_VISIBLE_DEVICES=1 nohup pytest -vvvvv -s --durations=50 tests/quantization/ --ignore tests/quantization/test_blackwell_moe.py`（1xMI350，after all model download）。原文数字：This branch (4ee2808) local MI350 579 (512 passed, 67 skipped) 0:21:56，main (9f295fe8cee4cbd2b21a5ce3066cec026e4bd2af) local MI350 586 (515 passed, 71 skipped) 0:56:09；This branch (c91679e660fc2cfee40f214e7fbb03edc9885441) local MI300 504 passed, 76 skipped 0:17:41，main local MI300 506 passed, 80 skipped 0:50:11；hosted MI350 this 513 passed, 67 skipped 0:26:17，main 515 passed, 71 skipped 0:57:52；hosted MI300 this 509 passed, 71 skipped 0:34:05，main 511 passed, 75 skipped 1:23:02。Test Results 另给出 1xMI350 上 `512 passed, 67 skipped, 28 warnings in 1317.00s (0:21:56)` 对 main `515 passed, 71 skipped, 23 warnings in 3369.39s (0:56:09)`。精度与推理 workload 条件正文未给出。

## 验证与回归风险

Test Plan：`.buildkite/test-amd.yaml` 中对应 shard 本地运行；Additional gsm8k tests 通过 `pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=tests/evals/gsm8k/configs/tests_refactor.txt`（含三个 Qwen/NVFP4 配置）跑精度；Test Results 给出 1xMI350 全量量化测试通过/跳过数。回归风险：测试被改为不实例化 vllm runner 且用 `hf_override` 缩小层数，覆盖路径与真实全量模型不同，可能掩盖只在完整 runner/完整层数下出现的问题；三个 accuracy tests 被移出到 `.buildkite/test-amd.yaml`，PR 未注明更多受影响路径。

## 关联

正文明确线索：修复 issue #51142；后续单独 PR #53585（likely deprecate/remove online quantization as part of `--quantization fp8`，涉及 `test_online_quant_peak_mem`）；三个 accuracy tests 移动到 `.buildkite/test-amd.yaml` 的指定行。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
