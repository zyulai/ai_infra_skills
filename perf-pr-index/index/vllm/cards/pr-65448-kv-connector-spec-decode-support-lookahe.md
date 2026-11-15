# PR #65448: [KV Connector][Spec Decode] Support lookahead block hashes in remaining KV connectors

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andakai |
| 合并 | 2026-11-12 |
| 分类 | Speculative Decoding(tier: major) |
| Label | multi-modality, kv-connector, kimi, k3, kv-cache-manager, mooncake, hisparse, nixl |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/65448 |

## 一句话结论

完成 #64096 的 lookahead block hash 迁移:删除 `supports_lookahead_block_hashes` 能力检查,UMBP/HiSparse 等剩余 connector 全部支持 EAGLE/EAGLE3/MTP 的 lookahead hash(block 边界 draft KV 可复用);UMBP 上 LongBench v2 准确率 48.5%→50.9%(Qwen3.6-27B MTP)。

## 动机

#64096 让 block hash 覆盖后续 `num_hash_lookahead_tokens`(L)个 token(EAGLE/EAGLE3/MTP),使 prefix hit 同时证明 block 边界的 draft KV 可复用;但未声明 `supports_lookahead_block_hashes` 的 connector 仍回退 L=0 与 legacy EAGLE trailing-block drop。本 PR 收尾:适配自身 reuse 路径仍丢 block 的 UMBP 与 HiSparse,然后移除能力检查;另以单独 commit 修复两个忽略 L>1(multi-module MTP)的 cache bound。

## 优化点

1. **所有 connector 获得 lookahead**:`supports_lookahead_block_hashes` 从基类、各 override 与 MultiConnector 移除(默认 True);只要配置了 KV connector 即解析 L,即使 core prefix caching 关闭;`disable_lookahead_block_hashes` 仍选 legacy drop。
2. **UMBP**:lookup、hit trimming、store mask 改用 core 的 L 并把 lookup 上限截到 hashed blocks;request 结束时对"调度过 compute"的请求保存并 pin 住最后采样才拿到 hash 的 KV(pending load 的请求不保存);KV events 带 lookahead token IDs。
3. **HiSparse**:自建 host block pool 接收 device pool 的 L;finish/cancel 时 publication 止步于有 hash 的页(与 core 的 `cache_blocks` 同界)。
4. **MoRIIO**:hybrid producer 截断最后一个 prompt token 时同步截断依赖该 token 的 hashes(同 NIXL)。
5. **Builtin LMCache MP fallback**:store/retrieve bound 从按 hash 数改为按 full token blocks(修复 `N % chunk_size < L` 时 assert)。
6. **Multi-module MTP bounds**:`replay_boundaries()` 接受 L(core、Mooncake store、UMBP、Offloading 传入);draft groups 只缓存 finalized tokens,target attention 与 Mamba state 仍缓存到 chunk 末尾。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/base.py: supports_lookahead_block_hashes 默认 True
vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py: lookup/store 改用 core 的 L + _register_finished_lookahead_save(结束 pin 保存)
vllm/v1/hisparse/coordinator.py: publish_when_ready 按 hash 页数截断 num_computed_tokens
vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py: mamba 截断时同步 truncate_block_hashes
vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py: bound 改按 len(all_token_ids)//vllm_block_size
vllm/v1/core/kv_cache_coordinator.py: replay_boundaries() 增加 num_hash_lookahead_tokens
vllm/distributed/kv_transfer/kv_connector/v1/{mooncake,mooncake_store,nixl,offloading,simple_cpu_offload}_*.py: 删除各自 override
tests/...: LMCache MP lookahead、UMBP finish-save、sparse retention replay boundary 等新测试
```

(注:pack 的 diff 截断,有 1 个文件未展示,以下仅覆盖已展示部分。)

```python
     def supports_lookahead_block_hashes(
         cls, kv_transfer_config: "KVTransferConfig"
     ) -> bool:
-        """Whether the connector supports core lookahead hashing."""
-        return False
+        """Whether the connector supports core lookahead hashing.
+
+        If False, `num_hash_lookahead_tokens` is 0 and block hashes cover only
+        each block's own tokens.
+        """
+        return True
```

## 性能声明

(验证为 LongBench v2 `short` 子集约 15k–60k tokens、temperature 0;每批 16 条以 `max_tokens=1` 预热后重置 local prefix cache 再评分;每 run 对比 main `ffdf267` 与本 PR 同一 precompiled build。)

- **UMBP,MI350X,Qwen3.6-27B MTP k=4,165 prompts,a=832**:main(drop)→PR(lookahead):accuracy 48.5%→**50.9%**,acceptance 67.1%→67.4%,restored tokens 4,569,344→4,706,624,boundary match 165/165;baseline A1/A2:47.9%/47.9%(acceptance 66.9%/67.0%)。
- **UMBP,MI350X,Qwen3-8B EAGLE3 k=4,103 prompts,a=64**:accuracy 39.8%→**41.7%**,acceptance 23.8%→24.1%,restored tokens 2,081,216→2,087,808,boundary match 103/103;baseline A1/A2:36.9%/38.8%(acceptance 23.9%/24.0%)。
- **HiSparse,GB300,GLM-5.3-NVFP4 TP4 MTP k=4,170 prompts,a=64**:accuracy 49.4%→**50.0%**,acceptance 68.8%→68.4%,restored tokens 4,693,760→4,704,512,boundary match 170/170;DSA without HiSparse:44.1%(acceptance 69.5%)。
- 正文结论:lookahead 每 prompt 恢复一个 alignment unit;accuracy 与 acceptance 在噪声范围内(UMBP 在 A/A spread 内)。
- **LMCache MP**(lmcache 0.5.6,Llama-3.1-8B EAGLE3,48 prompts):仅验证本 PR 不改变 LMCache 行为,46/48 请求恢复相同 tokens;accuracy main 37.5% vs PR 39.6%。

## 验证与回归风险

- Multi-module MTP 修复:两个新 unit test 在 main 上失败(resend 命中 0 而非 48;draft group 缓存 32 而非 16 tokens)、本 PR 通过;MI350X 上 UMBP、MoRIIO 与 core prefix caching 测试共 534 passed。
- 未覆盖:MoRIIO 仅 unit tests;HF3FS/Example/FlexKV 未重跑(从未支持 drop,代码未变)。
- 兼容性:out-of-tree connector 现在会收到 lookahead hashes,`request.block_hashes` 条目数可能少于 full token blocks;token-keyed connector(Example、HF3FS、LMCacheV1、LMCacheMP、FlexKV)行为不变,其边界处 draft KV 可能来自不同 continuation,降低 acceptance 但不影响输出质量(正文原述)。

## 关联

- 前置:#64096(引入 lookahead block hashes 与 `supports_lookahead_block_hashes`)。
- 涉及 connector:Mooncake、MooncakeStore、Nixl、MoRIIO、UMBP、HiSparse、Offloading、SimpleCPUOffload、LMCache MP、MultiConnector。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
