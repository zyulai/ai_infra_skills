# PR #60695: [Bugfix][Mooncake] Save the EAGLE attention proof with hybrid Mamba partial tails

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Dao007forever |
| 合并 | 2026-10-10 |
| 分类 | Speculative Decoding（tier: major） |
| Label | bug, ready, ci/build, kv-connector, kimi, k3, mooncake |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失）；pack 中 diff 截断，省略 8 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/60695 |

## 一句话结论

修复 Mooncake partial-tail 保存缺 EAGLE attention proof 的问题：hybrid 模型 GPU cache 重置后外部前缀命中率从 0 恢复，Kimi K3 端到端命中率 90.47%→94.04%。

## 动机

hybrid attention + Mamba 模型（Kimi K3）在 EAGLE 投机解码下，GPU prefix-cache 重置后 Mooncake 外部 store 命中为 0 token。根因（P=7,449、prefix-match unit U=128、EAGLE block drop 开启，Mamba checkpoint H=7,296）：普通 Mooncake 保存只存 attention 到最后一个 scheduler-block 边界（7,168）且从不存 Mamba 状态；partial-tail 保存虽然存了 7,296 的 Mamba 状态，但把 attention 裁剪到同一位置（`hash@7296`），位于 `hash@7424`（H+U）的 proof 从未入库。重置后 lookup 把 attention 7,296 经 EAGLE drop 降到 7,168，Mamba 在 7,168 以下无状态，命中 0。

## 优化点

- `ReqMeta.boundary_puts`：把 job 显式按前缀 hash 存的每个块列为 `BoundaryPut(group_id, block_id, num_tokens)`，scheduler 负责解析并 pin，worker 写到 `block_hashes[num_tokens // U - 1]`。两类条目：Mamba 状态（每次 core hand-off，block-aligned snapshot 或 H 处 sub-block tail 各成一个 put；DCP 下 Mamba 块小于 LCM 时只存 boundary 块）；non-Mamba tail（`_partial_tail_non_mamba_puts` 覆盖从最后 scheduler-block 边界到 H 加该组 `eagle_proof_margin` 的每段，键取各块末端并裁剪）。
- 发布时机：tail 在首次覆盖整个 prompt 的 save（`ReqMeta.publish_partial_tail`）或 Mamba hand-off 到达时发布一次；boundary 由 prompt 长度推导，无 Mamba hand-off 的 job 也能发布 attention proof；后续 save 不再重发 tail。
- Pinning：store job 精确 pin 其 boundary puts 涉及的块。此前 finish-time tail 只 pin Mamba 块，worker 却从已释放的表读 attention 块。
- 语义拆分：`ReqMeta` 区分 `num_prompt_tokens`（决定 checkpoint）与 `prefill_end_tokens`（含抢占后重放的输出 token，用于 store mask），并新增 `completed_token_len`。
- 对齐 core：外部 lookup 对 EAGLE 组扫过 hit limit（与 #60533 后的 core 一致）；Mooncake 不再自行判定 `enable_partial_hash_hits`，改读 core 解析后的 `kv_cache_config.cache_hit_alignment_tokens`（`engine/core.py` 一行使该字段在 scheduler 的 KVCacheConfig 上也可用）。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py: boundary_puts 解析/pin/_partial_tail_non_mamba_puts
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py: 按 boundary_puts 写块、publish 语义
vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py: ReqMeta/BoundaryPut 字段与语义
vllm/v1/engine/core.py: scheduler KVCacheConfig 透传 cache_hit_alignment_tokens/hash_block_size
tests/models/kimi_k3/test_prefix_cache.py: Mooncake master fixture + resend 端到端用例
```

## 性能声明

本 PR 自身只测了正确性；端到端收益由父 PR #59826 报告（含 core 修复 #60533，未对本 PR 单独重测）：Kimi K3，1P1D DCP8 + DCP8，DSpark k4，prefix-match unit 128；并发 128 下 prefix hit rate 90.47%→**94.04%**，吞吐 11,371→**15,754 TPS/GPU**。修复前行为：GPU prefix-cache 重置后 Mooncake 外部命中 0 token。

## 验证与回归风险

- CPU 单测 5 个套件（coordinator/scheduler/worker/HMA e2e/partial prefix hits）：本分支 480 passed；将测试文件拷到 `main`（`22948c5`）则为 29 failed / 233 passed，失败项包括 `test_eagle_attention_proof_published_after_checkpoint_handoff` 等，说明测试有效区分修复。ruff 干净，mypy 无新增错误。
- Kimi K3 端到端用例（`test_mooncake_resend_reuses_prompt_checkpoint`）本 PR 未运行：需 GPU、RDMA 与 `mooncake_master`，fixture 启动 master，缺二进制时 skip。
- 已知限制（PR 自述）：prompt-completion tail 只发送一次，若该 save 在 store 压力下被跳过或 put 失败则不重试（此前每次后续 save 会重建 tail）；行为对齐 core 的 partial-hit 判定可能改变 Mooncake 侧原有部分命中语义。

## 关联

- 父 PR：#59826（core + Mooncake 全部改动，评审时要求拆分）；姊妹 PR：#60533（core 部分，已合并，本 PR 依赖其 lookup 语义）。
- 同区域：本批 Speculative Decoding / KV 相关 PR #65874（fused GDN MTP）、#44594（kvcache watermark）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
