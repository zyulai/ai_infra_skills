# PR #8434: [Feature][kernel] tensor parallelism with bitsandbytes quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chenqianfzh |
| 合并 | 2024-09-17 |
| 分类 | Quantization \| tier: major |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/8434 |

## 一句话结论

为 bitsandbytes 量化补上张量并行（TP），llama2 / llama3 生成文本与无 TP 一致。

## 动机

正文说明本 PR "provides tensor parallelism to bitsandbytes quantization"，并在 llama2 与 llama3 模型上验证生成文本与无 TP 相同。从 diff 可见 `vllm/config.py` 原先在 `verify_with_parallel_config` 中显式抛出 "BitAndBytes quantization with TP or PP is not supported yet."（注释引用 bitsandbytes-foundation/bitsandbytes#1308），本 PR 移除该限制。

## 优化点

bitsandbytes 在 GPU 上按 TP 切分权重后再做 4bit 量化：`BitsAndBytesModelLoader._unquantized_generator` 按 `tp_size` / `tp_rank` 切分 `weight_tensor`——`down_proj` / `o_proj` 沿最后一维（`size(-1)`）切，其余沿第 0 维（`size(0)`）切，必要时 `contiguous()` 后送 `quantize_4bit`（注释引用 bitsandbytes#1342）。`vllm/model_executor/layers/linear.py` 的 `weight_loader` 在 `use_bitsandbytes_4bit` 时跳过 `narrow`（bnb 已加载对应分片）。pre-quantized bnb 模型 + TP 仍不支持：`quant_states` 无法与 split weight tensor 配合，loader 显式 `raise ValueError` 并建议使用 PP。新增测试 `test_load_tp_4bit_bnb_model` 以 `models_4bit_to_test` 参数化，要求至少 2 张 GPU 且 `is_quant_method_supported("bitsandbytes")` 为真，用 `hf_model_kwargs={"load_in_4bit": True}` 与 `vllm_tp_size=2` 跑 `generate_greedy(prompts, 8)` 后比对文本；`validate_generated_texts` 的 `vllm_tp_size` 默认仍为 1，`enforce_eager=True`、`gpu_memory_utilization=0.8` 保持不变。

## 关键改动

```text
vllm/config.py: 移除 bitsandbytes + TP/PP 的 ValueError 限制
vllm/model_executor/model_loader/loader.py: _unquantized_generator 按 TP 切分权重后再 4bit 量化，禁止 pre_quant + TP
vllm/model_executor/layers/linear.py: weight_loader 对 use_bitsandbytes_4bit 跳过 narrow
tests/quantization/test_bitsandbytes.py: 新增 test_load_tp_4bit_bnb_model（≥2 GPU），validate_generated_texts 增加 vllm_tp_size
```

TP 切分片段（摘自 loader.py 的 diff）：

```diff
+                if 'down_proj' in weight_name or 'o_proj' in weight_name:
+                    total_size = weight_tensor.size(-1)
+                    start_index = total_size // tp_size * tp_rank
+                    end_index = total_size // tp_size * (tp_rank + 1)
+                    weight_sub_tensor = weight_tensor[..., start_index:end_index]
+                else:
+                    total_size = weight_tensor.size(0)
```

## 性能声明

PR 正文未给出定量性能数据：正文只给出正确性验证（llama2 / llama3 生成文本与无 TP 相同），无 benchmark 数字，也没有模型 / 硬件 / 精度 / workload 等测试条件（涉及的模型为 llama2 与 llama3）。

## 验证与回归风险

正文：在 llama2 和 llama3 上验证生成文本与无 TP 相同；新增 `test_load_tp_4bit_bnb_model`（需 ≥2 GPU、需 `is_quant_method_supported("bitsandbytes")`），以 `vllm_tp_size=2` 对比 hf_runner 与 vllm_runner 的 greedy 生成文本。回归风险：pre-quantized BitsAndBytes 模型 + TP 不支持（loader 显式报错并建议改用 PP）；`default_target_modules` 注释为 "for Llama only"，TODO 支持其他模型；diff 中移除了两处 `torch.cuda.synchronize()`。其余路径 PR 未注明。正文同样没有给出任何性能数字或硬件型号，验证结论只到"生成文本一致"这一层。

## 关联

无（正文 / diff 仅引用 bitsandbytes 仓库 issue #1308、#1342，未给出 vLLM PR 编号）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
