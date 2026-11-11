# PR #63096: [ROCm][Spec Decode] Enable fused multi-step draft decode for ROCM_AITER_UNIFIED_ATTN

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fengshaoyi-amd |
| 合并 | 2026-11-09 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, verified |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数为 2 个文件(diff 完整,未见截断) |
| 链接 | https://github.com/vllm-project/vllm/pull/63096 |

## 一句话结论

让 ROCM_AITER_UNIFIED_ATTN 声明支持 draft metadata 原地更新,EAGLE/MTP 多步 draft decode 融合进单个 CUDA graph:隔离计时下 draft 循环 k=3 快 16–19%、k=7 快 6–8%;端到端吞吐持平(不宣称增益)。

## 动机

#46849 的 fused multi-step draft decode 在每个 draft attention backend 都设置 `supports_draft_decode_metadata_update` 时,一次 metadata 构建 + 一个 CUDA graph 跑完所有自回归 draft 步。`ROCM_AITER_UNIFIED_ATTN` 未支持,导致其上的 EAGLE 与 MTP drafter 每步都打日志回退:在 host 上重建 attention metadata,并对 k−1 个 draft decode 步各 replay 一次 draft graph。审计发现该 backend 的 metadata 只引用 fused 循环会保持最新的持久 buffer 视图(`seq_lens` 原地推进,另有 `block_table`、`slot_mapping`、`query_start_loc`)加跨步不变的标量(`max_seq_len` 每次 `propose()` 定死一次,`max_query_len` 恒为 1),无 host plan(`scheduler_metadata` 为 None),也不写 builder 自有 tensor——满足原地更新的全部前提。

## 优化点

机制上是一行级改动:给 `RocmAiterUnifiedAttentionMetadataBuilder` 加 `supports_draft_decode_metadata_update = True` 与 no-op 的 `update_draft_decode_metadata()`,与 Triton backend、FlashInfer trtllm-gen(#58371)的做法一致。效果是 `ROCM_AITER_UNIFIED_ATTN` 上的 multi-step drafter 走 #46849 的 fused draft decode 循环,回退日志消失,省去每步的 host metadata 重建与逐 graph replay;profile 显示 draft 步之间的 idle gaps(4.4µs)归零。范围刻意收窄:仅 `ROCM_AITER_UNIFIED_ATTN` 改动(MI300/MI355 经 `--attention-backend` opt-in,RDNA+AITER 下默认);`ROCM_ATTN` 共享 `build()` 但 decode kernel 不同且未验证,保持回退;`method=draft_model` 推测不走此循环;#62749 为目标模型 decode build 加的独立 flag 与本改动可组合。

## 关键改动

```text
vllm/v1/attention/backends/rocm_aiter_unified_attn.py: 声明原地更新支持 + no-op update
tests/v1/attention/test_triton_attn_metadata_builder.py: 两个 draft decode 测试纳入 unified builder(仅 ROCm 收集)
```

```python
class RocmAiterUnifiedAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

    def update_draft_decode_metadata(
        self, _metadata: RocmAttentionMetadata
    ) -> None:
        pass
```

## 性能声明

MI355X(gfx950),`rocm/vllm-dev:ci_base`(torch 2.13.0+rocm10.0.0, HIP 7.15),`VLLM_ROCM_USE_AITER=1`。Draft 循环隔离计时(`_multi_step_decode` vs `_fused_multi_step_decode`,同步前后,1024-token prompt,1151 calls/arm):k=3 为 571→463µs(batch 1)、697→585µs(batch 8)、678→559µs(batch 32),即 16–19%;k=7 为 1197→1108µs、1623→1530µs、1559→1464µs,即 6–8%;每循环节省 90–120µs,不随 k 增长。端到端:Llama-3.1-8B-Instruct + EAGLE3 speculator,synthetic acceptance 2.78,`--num-gpu-blocks-override 6144`(KV 1.5 GiB/层,避开 ROCm/aiter#6661 的 2–4 GiB 错误区);k=3 serving +0.1%/+0.2%(c1/c8)、−0.6%(c32);k=7 为 +1.9%/−0.1%/−0.2%,在 server 实例间波动内(main 三个实例 c1 即差 3%)。c32 torch profiler(两臂交错、顺序平衡):step 中位 7432 vs 7431µs,draft 段 588 vs 583µs,draft 步间 idle gap 4.4→0µs。正确性:GSM8K 5-shot k=3 strict-match 0.7028(main 0.7020/0.7043),acceptance 2.344 vs 2.343/2.345;k=7 zero-shot 0.8749(main 0.8681/0.8787);GPQA(gpt-oss-20b,k=4)0.4792/0.4729 vs main 0.4735/0.4855;`VLLM_GPU_SYNC_CHECK=warn` 两臂均无同步调用。

## 验证与回归风险

新增/扩展的 draft decode 单测断言两件事:`build()` 不写任何 builder 自有 tensor(字节不变),且原地 update 后 metadata 与对 replayed batch 重新 build 完全一致;参数化把 unified builder 加进 `tests/v1/attention/test_triton_attn_metadata_builder.py`,仅 ROCm 平台收集,CI 由 `:amd: (MI300) V1 Attention` 执行。作者还在 main 上跑过阴性对照(该测试在 main 上 2 failed)。风险很小但有两个已知边界:`--num-gpu-blocks-override` 需把 KV 控制在 2–4 GiB 以下(unified attention 在 gfx950 上此区间有错误,上游 ROCm/aiter#6661);gpt-oss-20b 在 k≥5 时 greedy 输出即偏离非投机解码(#65611,与本 PR 无关),故 GPQA 只测 k=4。

## 关联

前序:#46849(fused multi-step draft decode 循环本体)。同模式 backend opt-in:#58371(FlashInfer trtllm-gen)、#54790、#58449;相关:#62749(目标模型 decode build 的独立 flag)、#65611(gpt-oss-20b 高 k 输出正确性)。上游限制:ROCm/aiter#6661。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
