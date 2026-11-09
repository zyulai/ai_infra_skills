# PR #13454: [Model][MiniMaxText01] Support MiniMaxText01 model inference

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZZBoom |
| 合并 | 2025-04-01 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明「5 file(s) dropped from diff for size」 |
| 链接 | https://github.com/vllm-project/vllm/pull/13454 |

## 一句话结论

新增 MiniMaxText01 推理与线性注意力专用 cache，8×H800 支持 200 万 token 上下文。

## 动机

正文 Purpose 说明本 PR 旨在支持 [MiniMaxText01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01) 模型推理：单机 8xH800 与 8xH20 可跑，单台 H800 机器最大可处理 200 万 token 上下文输入，单台 H20 机器最大 500 万 token。该模型为混合线性注意力结构，需要配套的独立 cache manager 与类 mamba 的输入适配。

## 优化点

(1) 添加 MiniMaxText01 模型推理实现，以及专用于 linear attention 的独立 cache manager（diff 中新增 `constant_size_cache.py` 的 `ConstantSizeCache` 抽象基类，管理 Mamba/Minimax 的 constant size cache、request→seq→index 映射与 CUDA graph 输入）。(2) 适配与 mamba 模型一致的输入，包括 `request_ids_to_seq_ids` 与 `finished_requests_ids`。(3) 为连续多 batch 推理中 `finished_requests_ids` 的问题加临时修复：`async_llm_engine.py` 仅在 `scheduler_outputs` 非空时才 reset，避免 mamba_cache/minimax_cache 无法释放上几步的 finished id。(4) `config.py` 的混合模型层数统计在 `layers_block_type`（Jamba/zamba2）之外新增 `hf_config.attn_type_list` 分支（统计 `t == 1`）。(5) 新增 651 行的 `lightning_attn.py` 线性注意力层，并登记模型到 registry 与 `supported_models.md`。

## 关键改动

```text
vllm/model_executor/layers/lightning_attn.py: 新增（651 行）linear attention 实现
vllm/model_executor/models/constant_size_cache.py: 新增（136 行）ConstantSizeCache 抽象基类与 cache 索引管理
vllm/config.py: get_num_layers_by_block_type 增加 attn_type_list 混合模型分支
vllm/engine/async_llm_engine.py: finished_requests_ids 的临时修复（非空 scheduler output 才 reset）
tests/models/registry.py、docs/source/models/supported_models.md、tests/kernels/test_lightning_attn.py: 模型登记与测试
```

关键片段（混合模型层数统计）：

```diff
+            # Hybrid model Minimax
+            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
+            if attn_type_list:
+                return sum(t == 1 for t in attn_type_list[start:end])
```

## 性能声明

PR 正文未给出定量性能数据（perf claims 检测为 none，无 benchmark 吞吐/延迟数字与测试条件）。正文给出的量化信息为部署容量：单机 8×H800（TP8、`experts_int8`、bfloat16、`--max_model_len 2048000`、`--gpu_memory_utilization 0.95`、`--max_num_seqs 1`）最大 200 万 token 上下文；8×H20（`--max_model_len 5120000`）最大 500 万 token；默认启动参数为 `--max_model_len 1000000`。以上为容量上限，非性能 benchmark。

## 验证与回归风险

PR 正文未注明测试与验证结果（无 Test Plan / Test Result 章节，仅有 Purpose / Modifications / Deployment）。diff 含 `tests/kernels/test_lightning_attn.py`、`tests/models/registry.py` 与文档改动，但正文未给运行结果。回归风险：`config.py` 的混合模型层数统计改动影响其他 hybrid model（Jamba/zamba2 分支保留，新增 `attn_type_list` 分支）；`async_llm_engine.py` 的 `finished_requests_ids` 门控影响 mamba_cache/minimax_cache 在多 batch 推理中的释放（正文自称临时方案）；新 cache manager 与 CUDA graph capture 的 `copy_inputs_before_cuda_graphs` 交互。pack 标注 5 file(s) dropped from diff，diff 不完整，部分文件未包含在素材中。

## 关联

无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
