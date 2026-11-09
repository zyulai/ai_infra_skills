# PR #15526: Optimize Bailing-MoE with FlashInfer Fused All-Reduce

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-12-21 |
| 分类 | MoE (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/15526 |

## 一句话结论

让 Bailing-MoE(以 Ling-plus 验证)通过 `--enable-flashinfer-allreduce-fusion` 把 allreduce+rmsnorm+residual_add 融合为一,B200 TP8 上 E2E TTFT 降 7.2%、E2E 延迟降 10%。

## 动机

TP>1 的 Bailing-MoE 每层 MLP 输出都要做 tensor-parallel allreduce,随后又是独立的 RMSNorm 与 residual add;这三个操作在解码/预填充热路径上逐层重复,通信与 norm 各自产生 kernel launch 与显存往返。FlashInfer 的 fused_allreduce 已能把 allreduce+rmsnorm+residual_add 融合为单个 kernel,但 Bailing-MoE 模型尚未接入该路径。

## 优化点

在 `bailing_moe.py` 中接入 communicator 的 allreduce-fusion 机制:每层 forward 先询问 `layer_communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)`;命中时 MLP 的 `down_proj` 以 `skip_all_reduce=True` 跳过常规 allreduce,并把输出张量标记 `_sglang_needs_allreduce_fusion`,由下一层的 communicator 在 fused kernel 中一并完成 allreduce、rmsnorm 与 residual add;未命中时走原 `postprocess_layer` 路径。同时接入了既有的 `prepare_attn_and_capture_last_layer_outputs`(为 EAGLE 类辅助隐藏状态捕获)与 `should_use_flashinfer_cutlass_moe_fp4_allgather` 条件跳过 allreduce,并给最后一层加 `is_last_layer` 标记、attention 前加 `shape[0] != 0` 守卫。开启方式为 server 参数 `--enable-flashinfer-allreduce-fusion`,默认关闭。

## 关键改动

```text
python/sglang/srt/models/bailing_moe.py: MLP 跳过常规 allreduce、层间 fission 标记与 communicator 融合判定接线
```

```python
+        if should_allreduce_fusion:
+            hidden_states._sglang_needs_allreduce_fusion = True
+        else:
+            hidden_states, residual = self.layer_communicator.postprocess_layer(
+                hidden_states, residual, forward_batch
+            )
```

## 性能声明

B200,inclusionAI/Ling-plus,`--tp-size 8 --attention-backend triton --disable-radix-cache`,bench_serving(500 prompts,random 1024 输入/20 输出,range-ratio 1.0,max-concurrency 32):

- Mean TTFT 841.51 → 780.02 ms(-7.2%);P99 TTFT 1320.95 → 994.32 ms。
- Mean E2E Latency 1401.09 → 1260.14 ms(-10%)。
- Mean TPOT 29.45 → 25.27 ms;Mean ITL 29.45 → 25.27 ms;Max ITL 1256.89 → 645.99 ms。
- 总吞吐:total token throughput 23302.85 → 25845.37 tok/s;request throughput 22.32 → 24.76 req/s。
- GSM8K(200 题、8-shot、parallel 128):双方 Accuracy 0.900、Invalid 0.000;Latency 25.799 → 23.976 s,output throughput 1130.878 → 1215.039 tok/s。

## 验证与回归风险

- 精度验证为改前/改后 GSM8K 对照(同分 0.900),无逐 token 等价性检查;未新增单元测试(checklist 各项未勾选)。
- 回归面:仅 Bailing-MoE 模型文件改动,默认关闭、需显式开启 fusion flag;非 fused 路径的 `postprocess_layer` 保持原行为。风险集中在 fusion 判定与 reduce-scatter/fp4-allgather 等互斥条件的组合正确性(代码中已加互斥判断)。CI 结论 pack 中不可见。

## 关联

- 同区域:FlashInfer allreduce fusion 在其他模型(deepseek_v2 等)的接入;communicator 层的 allreduce/reduce-scatter 选择逻辑。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
