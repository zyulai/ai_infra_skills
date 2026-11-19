# PR #63254: [ROCm][Perf] Muse Spark: finalize the token-sharded MoE reduction in one reduce-scatter

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fanxingran |
| 合并 | 2026-11-18 |
| 分类 | Speculative Decoding（tier: major） |
| Label | rocm, ready, verified |
| Diff | pack 标注 +None/-None, None files（正文自述仅改 `muse_moe_runner.py`，+42/−20 行）；pack 末尾无「N file(s) dropped」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/63254 |

## 一句话结论

ROCm TP 下 Muse Spark MoE finalize 在 bf16 reduce dtype（无拷贝）时去掉 1024-token 分块，整批一次 reduce-scatter + 一次 in-place all-gather 完成，finalize 单测最快 −29%，TP8+DSpark serving 单请求 TTFT 降 7–8%、c32 吞吐 +6.0–6.9%。

## 动机

`MuseSparkTopKWeightAndReduce` 对 ≥256 tokens 的 MoE 输出 finalize 走三步：reduce-scatter、每 rank `fused_post_expert_finalize`、all-gather，并按 `_FINALIZE_CHUNK_ELEMS`（8192 tokens 时 1 GiB fp32）切成 1024-token chunk。但 recipe 的每条 AMD 命令都设 `VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16`，此时 `.to()` 不产生拷贝，chunk 便失去了限制内存的意义——反而把一对 reduce-scatter/all-gather 拆成 8 对（8192 tokens 时），且在 62 个 MoE 层各重复一次。因此当 per-route 输出已是 reduce dtype 时应整批一个 chunk。

## 优化点

- **整批单 chunk**：`token_sharded and routed.dtype == self.reduce_dtype` 时 `chunk_size = max(num_tokens, 1)`，整批一次 `_token_sharded_finalize`；fp32（有真实拷贝）时保持分块不变。
- **非 TP 整除批次**：超过一个默认 chunk 的批次只对 TP 整除前缀做 reduce-scatter，剩余 (<TP) tokens 走既有的 all-reduce + fused finalize（256 tokens 以下已用的路径）；对整批 padding 会拷贝全部数据（8191 tokens 时 512 MiB），故放弃。一个默认 chunk 以内的批次照旧 padding（至多 64 MiB，比多一次 all-reduce launch 便宜），≤1024 tokens 输出不变。
- **In-place all-gather**：chunk 无需 padding 时每 rank 直接 finalize 到自己输出行，然后 `dist.all_gather_into_tensor(out, out_local, group=get_tp_group().device_group)` 原地收集（NCCL/RCCL 支持的 in-place 布局），替代原先收集到新 `[T, H]` buffer 再拷贝；调用位于 MoE custom op 内，与 `sparse_attn_indexer`、`flashmla_sparse` 直接调用 `all_gather_into_tensor` 的方式一致。
- **范围**：仅改 `vllm/models/muse_spark/amd/muse_moe_runner.py`（+42/−20），无新 helper/flag/env var；fp32 reduce dtype、unfused-norm 路径、<256 tokens 批次与 EP（MoE tp_size==1）行为不变。与 #64461 的关系：MXFP4 (AITER) experts 在 TP prefill（≥1024 tokens eager batch）走 #64461 的 column-sharded stage 2，不再进本路径；两者改不同分支、任意顺序干净合并；Triton experts 上 column-sharded stage 2 不划算（BF16 intermediate 与 reduce-scatter 等宽，第二个 BF16 w2 放不下）。

## 关键改动

```python
# vllm/models/muse_spark/amd/muse_moe_runner.py
-        chunk_size = self._chunk_size(top_k, hidden_dim)
+        token_sharded = (
+            self.pre_norm_all_reduce
+            and self.fused_norm is not None
+            and (tp_size := get_tensor_model_parallel_world_size()) > 1
+        )
+        chunk_size = max_padded_tokens = self._chunk_size(top_k, hidden_dim)
+        if token_sharded and routed.dtype == self.reduce_dtype:
+            # The chunks only bound the reduce-dtype copy of ``routed``; with no
+            # copy, one reduce-scatter and all-gather beat one pair per chunk.
+            chunk_size = max(num_tokens, 1)
         for start in range(0, num_tokens, chunk_size):
...
+            tail = start
+            if token_sharded and end - start >= self.rs_min_tokens:
+                tail = end
+                if end - start > max_padded_tokens:
+                    # Padding copies the whole chunk; past one default chunk
+                    # the few tokens beyond a TP multiple take the all-reduce.
+                    tail -= (end - start) % tp_size
                 self._token_sharded_finalize(
-                    rows.to(self.reduce_dtype),
-                    topk_weights[start:end],
-                    output[start:end],
+                    routed[start:tail].to(self.reduce_dtype),
+                    topk_weights[start:tail],
+                    output[start:tail],
                     tp_size,
                 )
...
+        gather_in_place = gather and not pad and out.is_contiguous()
+        if gather_in_place or (not gather and owned == per_rank):
             out_local = out[rank * per_rank : (rank + 1) * per_rank]
...
+        if gather_in_place:
+            dist.all_gather_into_tensor(
+                out, out_local, group=get_tp_group().device_group
+            )
+            return
         gathered = tensor_model_parallel_all_gather(out_local, dim=0)
         out.copy_(gathered[:num_tokens])
```

## 性能声明

- **环境**：8×MI355X，`vllm/vllm-openai-rocm:muse-spark-open`（Python tree 替换），`amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` + `Inferact/Muse-Spark-Open-DSpark`（7 speculative tokens），env `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_KV_CACHE_LAYOUT=LBHNC AITER_UNIFIED_ATTN_SHORT_QUERY_3D=1`；base 与 PR 臂同节点背靠背、每对在两节点以交替臂序测。
- **TTFT on main（TP8+DSpark，base 0f6b261910，四臂 base/PR/PR/base 取两臂中位均值）**：2k 单请求 116.5 → 108.3 ms（−7.0%）；8k 356.1 → 328.0（−7.9%）；16k 720.4 → 665.2（−7.7%）；4×8k burst 1397.5 → 1289.8（−7.7%）。更早 base `6c2549be42` 上为 9–13%（节点 A：2k −13.2%、8k −10.0%、16k −9.6%、burst −10.1%；节点 B：−8.6%/−11.0%/−10.1%/−9.9%）。KV cache 6,846,396 tokens 四臂不变。
- **并发 sweep**（random 8k in/1k out，earlier base 0626d646d0，各 2 次取均值）：输出吞吐 c16 +2.0～+4.0%（如 1058.7 → 1097.9 +3.7%、1037.3 → 1057.6 +2.0%）、c32 +6.0～+6.9%（1299.1 → 1388.2 +6.9%、1315.8 → 1394.7 +6.0%）；mean TTFT 降 7–10%；mean TPOT 降 2–5%。仅覆盖 c16/c32；c1/c4 decode 步 8–32 tokens 低于 256-token 阈值，不受影响。
- **Finalize 单独**（8 GPU，`[T, 8, 4096]` bf16 partials，20 次 eager 调用中位，正文摘要"14–29% faster above 1024 tokens"）：T=2047：599.2 → 493.3 µs（−17.7%）；4100：1089.0 → 850.1（−21.9%）；8191：2148.2 → 1561.0（−27.3%）；8192：2104.5 → 1540.9（−26.8%）；16384：4156.2 → 2961.3（−28.7%）；T≤1024 bitwise equal；peak memory（profile run 8192 tokens）64 MiB/GPU，KV cache 尺寸不变；fp32 reduce dtype 时间低 0.5–1.7%、peak 153 → 144 MiB（无 padding 时）。
- **精度**：GSM8K strict/flexible（5-shot，1319）：节点 A 94.92 → 94.69，节点 B 94.84 → 94.69；McNemar paired p=0.74 / 0.86，无显著差异。TP8+EP8 输出 bitwise identical（EP 不走此路径）。bf16 reduce dtype、>1024 tokens 时与 base 非 bitwise 一致（RCCL 归约顺序不同，对 fp32 参考平均误差同为 1.44e-3）。

## 验证与回归风险

`pytest tests/kernels/moe/test_muse_spark_rocm_bf16_moe.py` 及其他 Muse Spark kernel/loading 测试：main（544f182d9f）115 passed；rebase 到 0f6b261910 后 nightly image 下 90 passed（其余 CUDA-only skip）。finalize 单独 A/B 在 main 上复跑与表内一致（<1%）。`VLLM_GPU_SYNC_CHECK=error` 下 TP8+DSpark 服务启动并服务 19 个 ≤8193-token prompt 的请求（覆盖整批 reduce-scatter + in-place all-gather 与 reduce-scatter + all-reduce tail），无 sync 报告。@mawong-amd 在 MI300X TP2/TP4 实测 oversized/tail、padded 与 FP32-reduce 控制路径（TP2 BF16 bitwise identical、TP4 在容差内）；CI build 101405 通过；padded 路径由既有测试覆盖；pre-commit 通过。回归风险：>1024 tokens + bf16 reduce dtype 下 TP 输出与 base 非 bitwise 一致（不同归约顺序）；in-place all-gather 要求 `out.is_contiguous()` 且无 padding，否则回退新 buffer 拷贝；greedy logprob 在 4096–8193 token prompts 上差 0.0002–0.33 nats（归约顺序）；AITER 自定义 reduce-scatter/all-gather 用于 256–1031 tokens（快 7–32%）留作后续。

## 关联

- #64461：MXFP4 (AITER) experts 的 column-sharded stage 2（TP prefill ≥1024 tokens），与本 PR 触不同 finalize 分支、可任意顺序合并。
- CI：Buildkite build 101405；base main `0f6b261910`、`544f182d9f`、`6c2549be42`、`0626d646d0`。
- pack 未提供其他关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
