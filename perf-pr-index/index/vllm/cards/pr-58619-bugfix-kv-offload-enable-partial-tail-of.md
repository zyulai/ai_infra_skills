# PR #58619: [Bugfix][KV Offload] Enable partial-tail offload for hybrid models under DCP

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @nvyutwu |
| 合并 | 2026-11-12 |
| 分类 | KV Cache / Memory(tier: major) |
| Label | bug, documentation, ready, ci/build, kv-connector, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/58619 |

## 一句话结论

Bugfix:旧 `supports_partial_tail` 要求单一 block size 且 `dcp == 1`,导致所有 hybrid 模型在 DCP 下 partial-tail offload 全部关闭;修复门控与 per-group 寻址后,Kimi-Linear-48B 在 GB300 TP2/DCP2 上每个 prompt 都能恢复到最后 hash boundary(此前部分 prompt 从 CPU tier 恢复 0 token)。

## 动机

#57216 拆出的第二部分(per @orozery 的 review)。`supports_partial_tail` 同时要求 `len(group_block_sizes) == 1` 与 `decode_context_parallel_size == 1`;DCP 下 full-attention offload block 是 `dcp ×` attention block 而 recurrent block 不分片(DCP 8 时 12288 vs 1536;attention group 在 `build_offloading_config` / `resolve_dcp_kv_block_size` 缩放,`MambaManager.__init__` 对 recurrent state 撤销缩放)。两个条件都不成立,该功能对 DCP 下的每个 hybrid 模型都是关的。store 路径还用同一个共享物理 block index(`boundary // self._partial_tail_block_size`)寻址所有 group——block size 不同即错;且 block size 取自 `kv_group_configs[0]`,它未必是 full-attention group。

## 优化点

1. 门控改为 core 已解析的 hit alignment:`cache_hit_alignment_tokens < alignment_tokens`,取代 `len(group_block_sizes) == 1` 与 `dcp == 1`;满足"每个 recurrent block 整除 full-attention block 且 hash-aligned、hash boundary 在 DCP rank 间均匀切分"即允许混合 block size。tail 可处的窗口即 full-attention offload block,`SchedulerOffloadConfig.alignment_tokens` 已持有——不加新字段。
2. 每个 group 按自己的 block size 寻址"结束于该 boundary 的 block":`(boundary - 1) // tokens_per_block`;旧的 `boundary // tokens_per_block` 在 boundary 落在 group block 边缘时指向下一个 block(今天 inert,因为 `blocks_per_chunk == 1` 时 worker 不用 `block_indices`,但前提一破即错)。
3. tail 存到 core 的 prompt checkpoint:`get_mamba_prefill_checkpoint_position(...)`(与 Mamba manager 同参数),且只存该 boundary 上的 hand-off entries。DCP 下一次 hand-off 可携带多个 boundary(如 prompt tail 旁的最后一个 full recurrent block),旧 `assert len(boundaries) == 1` 会变成 engine-core crash;存什么现在不依赖 hand-off 的批处理方式。
4. EAGLE/MTP partial tails 在 DCP 下保持禁用(proof position 未在 DCP 下验证);`dcp == 1` 时与 main(#61019)一致。
5. `dcp == 1` 时 store 路径解析出与之前相同的单 block index(无回归)。
6. 文档:`kv_offloading_usage.md` 新增 "Hybrid attention and recurrent models" 小节(DCP 下要求每个 recurrent block 整除 full-attention block、hash unit 可被 DCP size 整除)。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: supports_partial_tail 门控重写 + 每 group (boundary-1)//tokens_per_block 寻址 + get_mamba_prefill_checkpoint_position 边界
docs/features/kv_offloading_usage.md: 新增 hybrid/DCP 使用条件小节
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py: DCP-shaped hybrid fixture、multi-boundary hand-off、per-group 寻址测试
```

```python
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        cache_hit_alignment = kv_cache_config.cache_hit_alignment_tokens
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and alignment_tokens is not None
            and cache_hit_alignment is not None
            and cache_hit_alignment < alignment_tokens
            and has_partial_recurrent_group
            and spec.tokens_per_hash % dcp_world_size == 0
            and (
                dcp_world_size == 1
                or not any(config.is_eagle_group for config in kv_group_configs)
            )
        )
```

## 性能声明

- **Unit tests**(`pytest tests/v1/kv_connector/unit/offloading_connector/`,GB300,同镜像;base `main` @ e318ac0979):main 690 passed, 3 failed;本 PR 704 passed, 3 failed;本 PR + #58618 721 passed, 3 failed。3 个失败在 main 上相同(`test_partial_tail_store_uses_attention_and_recurrent_cow_sources` 与 `test_eagle_partial_tail_round_trip_serves_mamba_checkpoint[True/False]` 期望 recurrent group CPU `BlockStored.block_size` 为 0,而 #62613 起是 4)。`pre-commit`(ruff、mypy)通过。
- **Hardware(1 × GB300 node,`vllm/vllm-openai:nightly-2b009fc6`;"fix" 覆盖 branch 的 `vllm/`,"main" 为 stock nightly)**:Kimi-Linear-48B-A3B-Instruct(KDA + MLA),TP2,`--mamba-cache-mode align --prefix-match-unit 128`,OffloadingConnector + 16 GiB CPU pool;"fix" = 本 PR + #58618。每 prompt 流程:清空所有 tier、cold run(断言 0 cached)、只重置 GPU prefix cache、replay;下表为从 CPU tier 恢复的 token 数(`vllm:prompt_tokens_by_source{source="external_kv_transfer"}` 的 replay delta):

| prompt tokens | DCP1: main | DCP1: fix | DCP2: main | DCP2: fix |
|---|---|---|---|---|
| 1541 | 1536 | 1536 | **0** | **1536** |
| 12549 | 12544 | 12544 | 12288 | **12544** |
| 24069 | 24064 | 24064 | **0** | **24064** |
| 39685 | 39680 | 39680 | 38912 | **39680** |

  DCP1 与 main 完全一致(无回归);DCP2 每个 prompt 都恢复到最后 hash boundary;全部 trial 的 replay greedy 输出(32 tokens)与 cold run 及 GPU-prefix-cache replay 一致,0 allocation failures;同一数字在第二个 GB300 cluster 的上一 rebase 上复现。
- **早期版本(TP4/DCP4,review 改动前)**:GB300,`nightly-29468dde8b` 仅替换 `offloading/scheduler.py`;Kimi-Linear TP4/DCP4,recurrent block 512、full-attention offload block 2048、hash 128(K3 形状 1536/12288/128 缩小版):main 下只恢复整 2048-token chunk 且仅当最后一个 full recurrent block 落在 chunk grid 上(645/2565/2949/3717/17029 等恢复 0);本 PR 单独:tail 在 complete-chunk hit 的一个窗口内可存可查;加 #58618 后每 prompt 恢复到最后 hash boundary(640/2304/2560/2944/3712/4096/4480/17024)。replay greedy 输出与 cold recompute byte-identical,0 allocation failures、server log 无断言/错误。Caveat:该模型上每个 retained boundary 各自到达,多 boundary crash 路径仅由 unit reproducer 覆盖。
- 组合改动在 2-node GB300 K3 部署(TP8/DCP8)上的更早 run 见 #57216。

## 验证与回归风险

- Standalone 本身正确,但 CPU tier 只有在 #58618(lookup anchor)也落地后才能在 complete-chunk hit 一个窗口之外给出 hit;与它干净合并(已验证,合并树通过 suite)。
- `tokens_per_hash % dcp == 0` 是保守条件:store/load 每 rank 拷贝整物理块,copy 路径并不需要它,但这是硬件上验证过的唯一形状。
- EAGLE/MTP partial tails 在 DCP 下仍禁用;`prefix_match_unit` 不满足整除时 upstream 直接 ValueError。
- 3 个既有 unit 失败与 #62613 的期望值有关,与本 PR 无关。

## 关联

- #57216(拆出源)、#58618(part 1,lookup anchor)、#53889(互补的 config 侧 DCP 缩放)、#52397(`max_offload_tokens` cap 修复,留给该 PR)、#61019(dcp==1 的 EAGLE/MTP 行为)、#62613(既有测试失败来源)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
