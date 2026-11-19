# PR #65931: [Kernel][Perf] Add H100 BF16 batch-invariant matmul configs for gpt-oss

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @imthomp |
| 合并 | 2026-11-16 |
| 分类 | MoE（tier: major） |
| Label | ready, qwen, deepseek, gpt-oss, glm, batch-invariance |
| Diff | pack 标注 +None/-None, None files（实际 diff 仅 `batch_invariant_configs.py`，hopper 表新增 5 个形状配置）；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/65931 |

## 一句话结论

按 #62219 格式为 H100 添加 15 个 bf16 Linear 形状的 batch-invariant persistent-matmul 调优配置（覆盖 GLM-5.3、DeepSeek-V4.1、gpt-oss-20b/120b），kernel 比默认 config 快 1.08–3.04x，全部 438,030 行不变性校验零差异，gpt-oss-20b decode-heavy 吞吐 +4.3%。

## 动机

`VLLM_BATCH_INVARIANT=1` 下热门模型经 `linear_batch_invariant` 的 bf16 Linear 形状此前缺少 Hopper 调优配置（相关：#65569 与性能跟踪 issue #27433）。GLM-5.3 与 DeepSeek-V4.1 中其余部分为 FP8/FP4 用其他 kernel，bf16 层是 MoE router、sparse-attention indexer 投影、DeepSeek-V4.1 的 compressor 与 `lm_head`（TP=8/TP=4 每 rank）；gpt-oss-20b/120b 的未量化 attention、router 与 `lm_head` 自 #35404 起走该 kernel。Triton 路径不变，不把任何东西路由到 cuBLAS。

## 优化点

- 在 `vllm/model_executor/determinism/batch_invariant_configs.py` 的 `"hopper"` 表新增 5 个 `_MatmulShapeConfig`：(5120, 2880) gpt-oss-20b qkv_proj（BLOCK_K 128）、(2880, 4096) o_proj（128）、(201088, 2880) lm_head（64）、(32, 2880) MoE router（128）、(128, 2880) gpt-oss-120b MoE router（128），每个带 10 档 M bucket 的 (BLOCK_M, BLOCK_N, num_warps, num_stages)。加上 GLM-5.3 / DeepSeek-V4.1 的 router、indexer、compressor、lm_head 形状共 15 个形状。
- **不变性原则**：每个形状在所有 M bucket 用同一个 `BLOCK_K`，配合固定 bucket 配置保证 batch 尺寸不改变单行结果。
- **调优方法**：对每个形状与 `BLOCK_K ∈ {64, 128}`，在每 bucket 的最大 M 处计时选最快 (BLOCK_M, BLOCK_N, num_warps, num_stages)；保留与 cuBLAS 平均 log 时间比最低的 `BLOCK_K`。形状读自 GLM-5.3 checkpoint 与 DeepSeek-V4.1-Flash safetensors headers 并对照 vLLM 模型代码（batch invariance 下 `GateLinear` 回退 `ReplicatedLinear`；indexer/compressor 投影未量化；`lm_head` 为 vocab-parallel）。

## 关键改动

```python
# vllm/model_executor/determinism/batch_invariant_configs.py（"hopper" 表新增）
+        (5120, 2880): _MatmulShapeConfig(  # gpt-oss-20b qkv_proj
+            block_k=128,
+            m_buckets=(
+                (1, _MatmulMConfig(16, 64, 4, 5)),
+                ...
+                (2048, _MatmulMConfig(128, 128, 4, 3)),
+            ),
+        ),
+        (2880, 4096): _MatmulShapeConfig(  # gpt-oss-20b o_proj, block_k=128
+        (201088, 2880): _MatmulShapeConfig(  # gpt-oss-20b lm_head, block_k=64
+        (32, 2880): _MatmulShapeConfig(  # gpt-oss-20b MoE router, block_k=128
+        (128, 2880): _MatmulShapeConfig(  # gpt-oss-120b MoE router, block_k=128
# （GLM-5.3 / DeepSeek-V4.1 的形状配置在 pack 摘录中未展示, 共 15 个形状）
tests/v1/determinism/test_matmul_batch_invariant.py: + test_hopper_tuned_linear_batch_invariance
  （每个形状每个 bucket 边界两侧的首/末行精确相等校验）
```

## 性能声明

（环境：H100 80GB HBM3 (SXM)，torch 2.14.0+cu130，Triton 3.8.0，vLLM nightly 0.33.0rc1.dev454 + 本 diff。）

- **不变性**：每形状单一 `BLOCK_K` 跨所有 M bucket；单行独乘 vs 混入 M=1..1024 与每 16 档 M 至 8192 的 batch（取首 8、末 8 与 4 个中间行）：全部 15 个形状共 438,030 行中 0 行不同。
- **Kernel 速度**：调优配置在每个形状每个 bucket 比默认 config 快 1.08x–3.04x。M=1/32/256/2048 的代表值（提速 / 相对 cuBLAS）：GLM-5.3 MoE router (256,6144) 2.53x/1.84 → 1.99x/1.14；GLM-5.3 lm_head TP=8 (19360,6144) 2.57x/1.02 → 1.44x/0.98；DeepSeek-V4.1 compressor (1024,5120) 2.57x/1.42 → 1.20x/1.21；gpt-oss-20b o_proj (2880,4096) 3.04x/1.09 → 1.08x/1.50；gpt-oss-20b lm_head (201088,2880) 2.56x/0.96 → 1.26x/1.03；gpt-oss-20b MoE router (32,2880) 2.13x/1.27 → 2.51x/0.99。
- **端到端**（单 H100，`vllm bench serve`，3 交替轮，每轮 2,000 prompts，decode-heavy 512 in/256 out、prefill-heavy 1,024 in/4 out，--max-concurrency 32 --request-rate inf）：gpt-oss-20b decode-heavy output throughput 比 main +4.3%（3/3 轮更快；对 batch invariance off 的损失 22.8% → 19.5%）、prefill-heavy request throughput +2.4%（2/3 轮更快）；gpt-oss-120b decode-heavy +1.9%（3/3 轮更快；损失 22.1% → 20.6%）、prefill-heavy +0.9%（2/3 轮更快）。增益有限因 gpt-oss 的 MoE experts 用另一 kernel；GLM-5.3 与 DeepSeek-V4.1 预计同理。
- **未测**：GLM-5.3 与 DeepSeek-V4.1 端到端需整 8-GPU H200/H100 节点（排期中）；H200 使用同一 hopper 表但未基准测试。

## 验证与回归风险

新测试 `test_hopper_tuned_linear_batch_invariance` 检查 config 选择与每个形状每个 bucket 边界两侧的首/末行精确相等；`tests/v1/determinism/test_matmul_batch_invariant.py` 在 H100 上 50 passed（15 个新用例）。回归风险：配置表仅影响 `VLLM_BATCH_INVARIANT=1` 的 Hopper bf16 persistent-matmul 路径，Triton 路径不变；gpt-oss 的 lm_head (201088, 2880) 等超大 N 形状相对 cuBLAS 在大 M 时仍有 1.03–1.50x 开销（batch invariance 的固有代价）；相对 cuBLAS 个别 cell 大于 1（如 o_proj M=2048 1.50）意味着与关闭 batch invariance 相比仍有吞吐损失；H200 同表未测；调优脚本与原始数据在 https://github.com/imthomp/vllm-batch-invariance （src/tune_bi.py、results/）。PR checklist 中 Design Fit 等三项未勾选。

## 关联

- #65569、#27433（batch-invariance 性能跟踪 issue）；配置格式沿 #62219；gpt-oss 走该 kernel 起自 #35404。
- 调优脚本与原始输出：https://github.com/imthomp/vllm-batch-invariance 。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
