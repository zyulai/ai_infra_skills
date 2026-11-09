# PR #57214: [Perf][Pooling] Avoid blocking seq_lens GPU-to-CPU copy for pooling in FlashInfer metadata builder

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @frankwang28 |
| 合并 | 2026-09-27 |
| 分类 | Attention / Kernel |
| Label | performance, ready, nvidia, mrv2, verified |
| Diff | diffstat 缺失(pack 未提供);diff 可用,涉及 `vllm/v1/attention/backends/flashinfer.py` 与新增测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/57214 |

## 一句话结论

Pooling 模型在 FlashInfer metadata builder 里复用精确的 CPU seq_lens 上界,避免每步一次阻塞式 GPU→CPU 同步拷贝,L40S 端到端提升约 1.35%。

## 动机

FlashInfer 的 metadata builder 需要 CPU 侧 per-request 序列长度来规划 paged KV metadata。V2 model runner 下 `seq_lens` 在 GPU 上,builder 用 `common_attn_metadata.seq_lens.cpu()` 取回——这是一次与设备同步的 D2H 拷贝。对 pooling 模型该拷贝可以避免:pooling 没有 speculative tokens,model runner 维护的 `seq_lens_cpu_upper_bound` 是精确值而非乐观上界(pooling_runner.py 已注明 "Pooling has no speculative tokens, so this CPU upper bound is exact")。

## 优化点

`FlashInferMetadataBuilder.build` 中,当 `needs_seq_lens_cpu` 成立时:若 `_num_speculative_tokens == 0` 且 `seq_lens_cpu_upper_bound is not None`,直接把该 CPU 上界当作 `seq_lens_cpu` 使用,跳过 `seq_lens.cpu()` 的同步拷贝;否则保留原路径(`gpu_sync_allowed()` 下的 `.cpu()`)。判定条件覆盖安全面:一旦有 spec-decode draft 或 dLLM canvas token 这类投机 token 记账,被拒绝的 token 仍在途、上界变乐观,就回退到真实拷贝。builder 构造时缓存 `vllm_config.num_speculative_tokens`。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: build() 在无投机 token 时复用精确 CPU 上界
tests/v1/attention/test_attention_backends.py: 新增 5 场景拷贝计数测试
```

```python
         if needs_seq_lens_cpu:
-            with gpu_sync_allowed():
-                seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
+            if (
+                self._num_speculative_tokens == 0
+                and common_attn_metadata.seq_lens_cpu_upper_bound is not None
+            ):
+                # No speculative token accounting, so the upper bound is exact.
+                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
+            else:
+                with gpu_sync_allowed():
+                    seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
```

## 性能声明

L40S,e2e:Qwen/Qwen3-Embedding-0.6B,`--runner pooling --max-num-seqs 512 --max-num-batched-tokens 4096 --enable-chunked-prefill --async-scheduling --attention-backend FLASHINFER`,random input 1300 / output 1、8192 prompts、request-rate inf:

- Before:Request throughput 105.72 req/s,Total token throughput 137437.46 tok/s,Mean E2EL 45470.91 ms
- After:Request throughput 107.15 req/s,Total token throughput 139298.69 tok/s,Mean E2EL 44672.40 ms

即约 1.35% 提升(PR 原文 "performance boost of about 1.35%")。另做了 embeddings 数值几乎一致的人工核对。

## 验证与回归风险

新增单测 `test_flashinfer_avoids_seq_lens_copy_without_spec_tokens`,用 mock 统计 `.cpu()` 调用次数,覆盖 5 种场景:generate/pooling 无投机(0 次拷贝)、无 upper_bound(1 次)、spec_decode(1 次)、diffusion(1 次),并验证 metadata 结构与 last_page_len 正确。风险面:只在 FlashInfer backend、非 DCP/非 cascade、全 trtllm 之外的路径生效;正确性依赖"无投机 token 时上界精确"这一 runner 不变量,若未来 runner 引入新的在途 token 机制而未接入该判定,可能出现元数据偏差(测试把 spec_decode/diffusion 显式排除在外)。

## 关联

同批无其他 pooling/FlashInfer metadata 卡;pooling runner 对上界精确性的既有注释(pooling_runner.py)是本 PR 依据。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
