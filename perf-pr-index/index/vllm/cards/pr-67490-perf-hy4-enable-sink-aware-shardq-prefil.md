# PR #67490: [Perf][Hy4] Enable sink-aware ShardQ prefill

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @qianlihuang |
| 合并 | 2026-11-16 |
| 分类 | Other（tier: major） |
| Label | documentation, ready, ci/build |
| Diff | pack 标注 +None/-None, None files；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/67490 |

## 一句话结论

为 HY4 sparse-MLA 开启 sink-aware ShardQ prefill（TP 下按 head 顺序 all-gather checkpoint 加载的 attention sink），8×H200 上 prefill-only 吞吐在 ISL8192 +59.10%、ISL32768 +62.53%。

## 动机

ShardQ（token 分片 prefill）让每个 rank 对自己的 token slice 计算全部 query head，能显著提升长输入 prefill 吞吐。但 HY4 的 FlashMLA sparse 实现带 learnable attention sink，此前 sink 是按 head 与 TP 分片（`supports_shardq: ClassVar[bool] = False`），与 ShardQ「每 rank 跑所有 head」的布局冲突，因此 HY4 一直无法使用 ShardQ prefill。本 PR 让 HY4 的 sink 支持 full-head 布局，从而把共享 ShardQ 机制扩展到 HY4 的 model-local sink backend。

## 优化点

- `HYV4FlashMLASparseImpl.supports_shardq` 由 `False` 改为 `True`，复用既有 token split、attention kernel 与 eligibility checks。
- 新增 `process_weights_after_loading`：在 TP 执行设备上一次性 `tp_group.all_gather(self.sinks, dim=0)` 得到 full-head sink（`_sinks_full`）；reload 时用 `copy_` 原地更新以保住被捕获 CUDA graph 持有的指针。
- `_sinks_for_query` 中当 sink 数与 runtime query heads 不匹配时，改用 `_sinks_full`；不匹配且无法回退时报错（错误信息带 `full_sinks` 形状）。
- `HYV4MLAAttention` 构造 ShardQ 包装时传 `shardq_eligible=not is_mtp_layer`（MTP 层不参与）。
- decode 继续使用本地 sink 布局，不受影响。

## 关键改动

```python
# vllm/models/hy_v4/nvidia/flashmla_sparse.py
-    # The sinks are sharded by head with TP, and ShardQ runs every head on
-    # each rank.
-    supports_shardq: ClassVar[bool] = False
+    # ShardQ computes every head for each rank's token slice.
+    supports_shardq: ClassVar[bool] = True
...
+    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
+        super().process_weights_after_loading(act_dtype)
+        if self.sinks is None or not self.use_shardq:
+            return
+        tp_group = get_tp_group()
+        full = tp_group.all_gather(self.sinks.to(tp_group.device), dim=0)
+        # Reload in place to preserve pointers held by captured CUDA graphs.
+        if self._sinks_full is None:
+            self._sinks_full = torch.empty_like(full)
+        self._sinks_full.copy_(full)
```

其他：`vllm/models/hy_v4/nvidia/attention.py` 传入 `shardq_eligible=not is_mtp_layer`；新增 `tests/models/hy_v4/test_shardq_sinks.py`（138 行，TP1/TP2/TP8 三组 `mp.spawn` 多进程用例）；Buildkite `deepseek-shardq-sequence-parallel` 步骤改名 "ShardQ Sequence Parallel" 并纳入 `vllm/models/hy_v4/nvidia/*` 与新测试文件；删除 `docs/models/hy4.md`（supported_models.md 链接改为纯文本 HY4）。

## 性能声明

- **环境**：PR source `25227c86327a0acf3e9d502286028df5db57000c`，frozen main `329f93e1909efdd4548524b1500fa2c54ef2cbb0`，重建 CUDA extensions；`tencent/Hy4-preview-FP8`，TP8/DP1/EP8，ModelOpt MXFP8，默认 Humming MoE，`fp8_ds_mla` KV，自然 MTP3，默认 graph mode，`max_model_len=262144`。8×H200。测试包：torch 2.14.0、triton 3.8.0、flashinfer-python 0.7.3、humming-kernels 0.1.19。两臂均设 `VLLM_ENABLE_HPC_OPS=1`（SM90 iHC gate 保留 Triton fallback）。
- **方法**：同 GPU 串行 A/B/A，每臂 3 seeds（102/104/106），warmup 剔除；ISL2048/8192/32768 分别用 128/32/16 个请求，OSL1、RPS∞。
- **吞吐**：ISL2048：3.46993 → 3.46887 req/s（−0.03%）；ISL8192：0.85273 → 1.35666 req/s（+59.10%）；ISL32768：0.20758 → 0.33738 req/s（+62.53%）。ISL8192 seed102 明细：Mean TTFT 19362.15 → 12179.45 ms，Total token throughput 6989.21 → 11119.20 tok/s；ISL32768 seed102：Mean TTFT 41028.54 → 25244.17 ms，Total token throughput 6801.18 → 11060.41 tok/s。
- **Profile**（rank 0 FlashMLA 每次 native 调用中位 GPU 时间，含关联 metadata/attention/combine kernel）：query 形状 `[1,8192,64,576]` → `[1,1024,64,576]`；ISL8192：6.706 → 0.858 ms；ISL32768：6.734 → 0.879 ms。
- **GSM8K**（stock evaluator，20-shot，1319 题，C32，temp 0，seed 42，默认 256 输出 token 上限；tokenizer 实测输入 3554–3717 tokens）：Main Accuracy 0.924 / 12.265 questions per second；PR Accuracy 0.934 / 12.287 questions per second；Invalid responses 0.001 / 0.002。单独 greedy 单请求检查（3596 输入 tokens）产生相同 next token 与 top-five token 集合，五个 token 最大绝对 logprob 差 0.5782。

## 验证与回归风险

回归测试 `pytest tests/models/hy_v4/test_shardq_sinks.py -v -s`：main 上 TP2 用例在 `assert layer.mla_attn.use_shardq == (world > 1)` 失败；本 PR 后 TP1/TP2/TP8 全过（72.49 s）。TP2 用例用 32 总 heads 覆盖两 GPU CI，含 CPU-staged finalization、精确 local/full sink 值、padding、weight reload 与 graph replay；TP8 用 64 总 heads。Pre-commit 通过全部改动文件；既有两 H100 ShardQ Buildkite 步选 TP1/TP2，TP8 为 8×H200 手动运行。吞吐测量在 loading-only CPU-staging 修复与注释清理之前完成，最终文件用真实 checkpoint、20-shot GSM8K、匹配 profile 与 sink 回归套件复验。回归风险：`_sinks_full` 的 all-gather 发生在权重加载后（TP 通信路径新增一处）；CUDA graph 捕获依赖 `_sinks_full.copy_` 原地 reload 语义（测试已覆盖 replay 后 sink 值）；若 sink 数与 query heads 不匹配且 `_sinks_full` 缺失会启动/前向报错；正文未列出其他已知回归。

## 关联

- 前序扩展：#66549（norms/A-projection 的 ShardQ 扩展，覆盖 DeepSeek-V3.2 / GLM-5.3 模型包）。
- 正文提及（未给编号）：open DSv4.1 sink/token-sharding 扩展与 shared route-selection refactor 为独立改动；对 HY4/Hy4 ShardQ 的 open-PR 检索未发现其他实现。
- pack 未提供其他关联 PR/issue 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
