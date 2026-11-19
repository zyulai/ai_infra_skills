# PR #63085: [ROCm][Perf] Muse Spark Open Shard residual stream across TP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kliuae |
| 合并 | 2026-11-17 |
| 分类 | Speculative Decoding（tier: major） |
| Label | rocm, ready, ci/build |
| Diff | pack 标注 +None/-None, None files；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/63085 |

## 一句话结论

ROCm 下 Muse Spark TP 时按 token 切分 residual stream（大 forward 时 all-reduce 变 reduce-scatter+all-gather，每 rank 只对自己分片的 token 做逐 token 工作），TP8 吞吐 +5.9～15.7%、TP8+EP8 +2.0～8.7%、TTFT 降 15～26%。

## 动机

Muse Spark 在 ROCm TP 下每个 rank 保存完整 residual stream，attention 与 MoE collectives 之间的逐 token 工作（RMSNorm、gated residual update、router、squeeze、expand）每个 rank 都对全部 T tokens 重复执行。TP8 时 7/8 是重复工作，且其他 rank 其实已有这些数据。all-reduce 传输的字节量与 reduce-scatter+all-gather 组合相同，差异在 collective launch 开销：EP 下每层多三次 launch，8k tokens 时约 0.12 ms，而按 token 分片省约 0.84 ms；低于约 1k tokens 时额外 launch 延迟与分片节省相当，故默认阈值 1024 tokens，更小的 forward（包括 decode 与所有 CUDA-graph 捕获 batch）保持现行路径。

## 优化点

- **`TokenShard`**：rank r 持有行 `[r*R, (r+1)*R)`（R = ceil(T/tp)），`local()` 本地切片（不足补零）、`gather()` all-gather 截回 num_tokens、`reduce_scatter()` 先 pad 到 tp 倍数再 reduce-scatter。
- **collective 改造**：attention 前 all-gather；o_proj 后 all-reduce → reduce-scatter（`o_proj` 以 `reduce_results=not defer_reduce` 构建，不分片的 forward 在 `forward` 里做同样的 all-reduce）；experts 前 all-gather projected input 与 router logits；experts 后 TP+EP all-reduce → reduce-scatter；TP-sharded experts 的 route rows reduce-scatter + all-gather（每 chunk 8 次）合并为一次全 route rows 的 reduce-scatter。
- **MoE 侧**：EP 时 `defer_reduce`，TP-sharded 时经 `token_sharded_moe_output()` 上下文让 `MuseSparkTopKWeightAndReduce.apply` 走 `_token_sharded_finalize(gather=False)`，只写当前 rank 的行（要求 fused post-expert norm 且 bf16 reduce dtype，否则 NotImplementedError）；aux hidden states 与最终输出按需 `shard.gather` 给 drafter（读目标中间 hidden states 的投机 drafter 需要）。
- **门控**：`token_sharded_stream_supported` 要求 TP>1、DP=PP=PCP=DCP=1、无 sequence_parallel_moe、全部 `mlp_layer_types == "sparse"`、非 DSpark config，且 EP 或 bf16 MoE reduction；`VLLM_ROCM_MUSE_SP_MIN_TOKENS`（默认 1024，=0 关闭）控制最小 token 数，且要求 `num_tokens % token_shard_divisor == 0`（TP-sharded experts 下为 TP，避免 padding 拷贝）与非 graph 捕获流。

## 关键改动

```python
# vllm/models/muse_spark/amd/model.py
+class TokenShard:
+    """Current TP rank's rows of a token-sharded residual stream."""
+    def __init__(self, num_tokens):
+        self.world = get_tensor_model_parallel_world_size()
+        self.rows = -(-num_tokens // self.world)
+        self.start = get_tensor_model_parallel_rank() * self.rows
+    def local(self, x): ...   # 切片 + 补零
+    def gather(self, x): return tensor_model_parallel_all_gather(x.contiguous(), 0)[: self.num_tokens]
+    def reduce_scatter(self, x): ... tensor_model_parallel_reduce_scatter(x.contiguous(), 0)
+def token_sharded_stream_supported(vllm_config): ...  # TP>1, EP 或 bf16 MoE reduction, DP=PP=PCP=DCP=1, all-sparse MLP
# model forward:
+        shard = None
+        if (mono_out is None and self.token_shard_min_tokens
+                and num_tokens >= self.token_shard_min_tokens
+                and num_tokens % self.token_shard_divisor == 0
+                and not torch.cuda.is_current_stream_capturing()):
+            shard = TokenShard(num_tokens); x = shard.local(x)
# o_proj: reduce_results=not defer_reduce; back(): shard is not None -> shard.reduce_scatter(output)
# MoE: EP defer_reduce / TP-sharded experts_out = shard.reduce_scatter(experts_out)
vllm/models/muse_spark/amd/muse_moe_runner.py:
+@contextmanager
+def token_sharded_moe_output(): ...  # 全局 _token_sharded_output 标志
+        if _token_sharded_output and self.pre_norm_all_reduce:
+            # 需要 fused norm 与 bf16 reduce dtype, 否则 NotImplementedError
+            self._token_sharded_finalize(routed, topk_weights, output, tp_size, gather=False)
vllm/envs.py: + VLLM_ROCM_MUSE_SP_MIN_TOKENS: int = 1024
tests/distributed/test_muse_spark_token_shard.py（新, 2-GPU round trip: local/gather/reduce_scatter, num_tokens 1/5/16/1027）
tests/kernels/moe/test_muse_spark_rocm_bf16_moe.py / tests/models/test_muse_spark_rocm_loading.py: 新增 token-sharded 用例与 fail-closed 校验
```

## 性能声明

（模型 amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4，MI355X；bench：8192/1024，CONC 8/16/32/64，main 为 ROCm enablement branch。）

- **TP8**：Total tokens/s c8 5,629.11 → 5,963.56（+5.94%）、c16 8,252.63 → 8,951.72（+8.47%）、c32 11,397.82 → 12,764.28（+11.99%）、c64 14,579.79 → 16,867.00（+15.69%）；mean TTFT c8 −23.20%、c16 −23.65%、c32 −26.03%、c64 −22.41%；mean TPOT −3.34%～−12.61%。加 DSpark：吞吐 +7.52%～+10.00%，TTFT −20.81%～−23.49%（c16 output tokens/s +7.88%，c32 +7.52%）。
- **TP8+EP8**：Total tokens/s c8 5,139.37 → 5,243.25（+2.02%）、c16 8,453.32 → 8,792.39（+4.01%）、c32 12,400.49 → 12,995.48（+4.80%）、c64 16,633.40 → 18,087.38（+8.74%）；mean TTFT −15.42%～−15.82%。加 DSpark：c8 +4.00%、c16 +6.99%、c32 12,880.73 → 14,791.92（+14.84%），TTFT −10.94%～−15.10%。
- **Claims 汇总（PR 原文口径）**：TP8+EP8 约 2.0%–8.7% 吞吐、TP8 约 5.9%–15.7%；TTFT TP8+EP8 约 −15%、TP8 约 −22–26%；DSpark TTFT −14% 到 −22%、output throughput TP8 c16/c32 +5.2%/+9.7%、TP8+EP8 c32 +4.2%；端到端精度不变。
- **GSM8K 5-shot（MI355X）**：TP8+EP8 flexible 0.9454 → 0.9431、strict 0.9462 → 0.9439；TP8 flexible 0.9484 → 0.9447、strict 0.9492 → 0.9447；TP8+EP8+DSpark flexible 0.9553 → 0.9515、strict 0.9545 → 0.9507；TP8+DSpark flexible 0.9439 → 0.9515、strict 0.9431 → 0.9507（±0.006 级别，精度不变口径）。

## 验证与回归风险

新增 `tests/distributed/test_muse_spark_token_shard.py`（`@multi_gpu_test(num_gpus=2)`，2 GPU 上验证 TokenShard 的 local 行归属（owned 行 0 容差、padding 行为零）、gather 恢复原张量、reduce_scatter 求和结果，num_tokens 1/5/16/1027）并纳入 Buildkite distributed 步骤。`test_muse_spark_rocm_bf16_moe.py` 新增 token-sharded stream 只写当前 rank 行的用例（monkeypatch reduce-scatter/all-gather，断言 `scattered == [num_tokens]` 且非本 rank 行为 NaN）与 EP 下可 skip final all-reduce 的 fail-closed 校验；`test_muse_spark_rocm_loading.py` 新增 `token_sharded_stream_supported` 的 EP/bf16 条件参数化校验。回归风险：改动限 ROCm Muse Spark 路径，需 TP>1 且 EP 或 bf16 MoE reduction、DP=PP=PCP=DCP=1；token-sharded MoE 输出要求 fused post-expert norm + bf16 reduce dtype 否则抛 NotImplementedError；TP-sharded experts 要求 token 数整除 TP（否则不分片）；graph 捕获流与 <1024-token forward 保持旧路径；`skip_final_all_reduce` 校验由一律拒绝放宽为 tp_size>1 才拒绝；DSpark config 被排除在支持范围外；PR checklist 四项未勾选。

## 关联

- 依赖同一 rocmmuse 基础设施：`MuseSparkTopKWeightAndReduce._token_sharded_finalize`（本 PR 为其加 `gather=False` 模式）、`VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE`。
- 姊妹 PR：#63254（MoE finalize 单次 reduce-scatter，二者均改 muse_moe_runner.py）；正文表格中 "8 of each chunk" 指向该路径。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
