# PR #63956: [Perf][DSA] Token-sharded sparse-MLA prefill under TP; remove indexer row sharding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-11-05 |
| 分类 | Attention / Kernel \| tier: major |
| Label | rocm, ready, ci/build, deepseek, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 11 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/63956 |

## 一句话结论

ShardQ 按 token 分片 TP 下 sparse-MLA prefill 替代行分片，GLM-5.3 128K TTFT 提速 1.40x。

## 动机

TP 下每 rank 只持有 H/tp 个 heads，但 indexer 与 sparse attention kernel 仍要处理全部 prefill tokens。PR 正文指出 #54951 引入的 indexer TP row sharding 需单步 ≥65536 个 prefill rows 才生效，高于默认 max_num_batched_tokens，默认配置下从不启用。本 PR 用按 token 分片的方案取代它，并一并覆盖 GLM-5.3-Flash 的 k-pool indexer。

## 优化点

ShardQ（Ulysses 风格）把一步 tokens 按 zigzag 切成 2*tp 个 chunk，rank r 拥有 chunk r 与 2*tp-1-r 以平衡因果 indexer 开销；indexer 只为本 rank tokens 打分并紧凑写 top-k。随后一次 all-to-all 把 MQA query 从 [T, H/tp, D] 变为 [T/tp, H, D]，sparse attention kernel 在复制的 KV cache 上跨全部 heads 计算，再用第二次 all-to-all 还原 [T, H/tp, 512] 供 W_UV 投影。交换用一个 Triton kernel 直接写入对端 symmetric-memory buffer，TP group 无 symmetric memory 时回退 NCCL all_to_all_single。默认对解码后 ≥1024 tokens 的 step 开启（sparse_mla_shard_q_min_tokens），并覆盖混合 prefill+decode step（decode tokens 仍按 head 分片）。

## 关键改动

```text
vllm/model_executor/layers/attention/mla_attention.py: 分配 ShardQ buffer，在 forward_impl 调用 shard_q / unshard_out
vllm/model_executor/layers/attention/sparse_mla_attention.py: 稀疏 MLA 公共实现与 metadata builder 适配 ShardQ
vllm/model_executor/layers/sparse_attn_indexer.py: 移除 indexer 的 TP row sharding
vllm/config/attention.py: 新增 sparse_mla_shard_q 等配置项
tests/v1/attention/test_shardq.py 等: ShardQ 切分、交换与后端回归测试
```

```python
            if shardq is not None:
                decode_q, mqa_q = shard_q(mqa_q, shardq)
                decode_out = None
                if decode_q.shape[0]:
                    # Decode tokens attend head-sharded.
                    decode_out, _ = self.impl.forward_mqa(
                        decode_q, kv_cache, attn_metadata, self
                    )
                mqa_metadata = shardq.attn_metadata
            attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, mqa_metadata, self)
            if shardq is not None:
                attn_out = unshard_out(attn_out, shardq, decode_out)
```

## 性能声明

GLM-5.3 NVFP4、TP4、GB200、默认 FlashInfer sparse MLA、concurrency 1 下 TTFT：32K 1.24x、64K 1.29x、128K 1.40x。B300 serving A/B（GLM-5.3 NVFP4、TP4、concurrency 1、256 output tokens、四对重启进程）：server TTFT 8K 1.19x、32K 1.29x、128K 1.46x、512K 1.99x、1M 2.62x（162.5 s → 62.0 s），1K-4K 低于阈值不变。Dummy weights 下 per-step prefill forward 比 main 快 1.11x-1.70x（16K @128K：926 ms → 544 ms），比被移除的 row sharding 快 1.18x-1.25x；混合 prefill+decode step 在 FlashInfer 上快 1.16x-2.90x、FlashMLA sparse + fp8_ds_mla 上 2.04x-2.97x。PR 正文注明性能与精度数字测于 ZJY0516/vllm#24 重构之前、且用更早的 8192-token 阈值，未在当前 head 重跑。

## 验证与回归风险

GB200 上 test_shardq.py 等 229 passed；test_sparse_mla_backends.py 与 test_flashmla_sparse.py 547 passed、1384 skipped；PCP/DCP 回归与 merge base 结果一致。gsm8k（5-shot、~10K-token 45-shot）与 passkey retrieval 精度基本不变，132/132 匹配探针生成 token 完全一致。风险：正文自述数字未在重构后重测；GLM-5.3-Flash 端到端 serving 未测；仅 TP>1、无 DCP/PCP/HiSparse、混合 batch 不用 FULL CUDA graphs 时启用；TP1/DCP 等非 ShardQ 配置仍会触发 FlashMLA fp8 的 int32 stride 上限；TP4 默认 16384 budget 下 KV cache 每 GPU 缩小 0.47 GiB。

## 关联

取代 #54951 的 indexer 行分片，并沿用其 serving A/B 协议与检索探针；两棵对比树都包含 #64122 的 FlashInfer sparse MLA 修复；正文说明数字测于 ZJY0516/vllm#24 重构之前。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
