# PR #27770: [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ishandhanani |
| 合并 | 2026-08-21 |
| 分类 | KV Cache / Memory |
| Label | amd, hicache, run-ci, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,可见部分涉及 10 个文件,另有 1 个文件因超长未展示 |
| 链接 | https://github.com/sgl-project/sglang/pull/27770 |

## 一句话结论

把 P/D 分离的 decode 侧 radix cache 扩展到 SWA hybrid 模型(gpt-oss 等),复用 full-attention KV、只从 prefill 传 SWA 窗口增量。

## 动机

decode 侧 radix cache(PR #19746)此前对 SWA hybrid 模型(如 gpt-oss)启动即拒绝。这类模型有两套生命周期不同的 KV:full-attention 层全量保留(大而持久),sliding-window 层只保留最近 W 个 token(旧槽位随窗口推进回收)。因此在 decode worker 上,缓存前缀的 full-attention KV 可以跨请求复用,但其 SWA KV 已失效(成为 tombstone:full KV 在、SWA 已回收)。本 PR 让 decode worker 复用缓存前缀的 full-attention KV,只向 prefill worker 请求增量,减少重复的 P/D KV 传输。功能限定在实验性 unified radix tree(`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`);默认 `SWARadixCache`、Mamba/SSM、DSA(DeepSeek-V4)、SWA-compress(Gemma/MiMo)均不支持。取代 #26218(该版本走默认 SWARadixCache,现已放弃并统一到 unified tree)。

## 优化点

`kv_cache_builder.py`:SWA + `--disaggregation-decode-enable-radix-cache` 仅在 unified tree 激活时放行,否则报错;hierarchical cache、DSA、SWA-compress 在 unified tree 下也仍拒绝。`decode.py`:decode 准入时把 full-attention 前缀复用 cap 在 sliding-window start,使窗口整体落在 fresh delta 中(保持 `alloc_extend_swa_tail` 的 tail→full 映射在界内,只损失最后约 window_size 个 full token 的复用);radix 命中时也可走 `alloc_extend_swa_tail` 只分配 SWA 尾部(此前 prefix_len>0 时回退到按整页分配 SWA);非窗口前缀标记 `swa_evicted_seqlen`(SWA tombstone);预算/驱逐改为 SWA-aware(full_* accessor,SWA 可驱逐量纳入 tail 预算);`swa_uuid_for_lock` 贯穿 prefix lock/release,准入拒绝路径也正确释放原匹配节点锁。`unified_radix_cache.py`/`swa_component.py` 修两个阻塞 bug:(1) SWA pool 泄漏——`prepare_for_caching_req` 只在 is_finished=True 转发 `swa_evicted_seqlen`,未完成路径把窗口外 token 当 live SWA 计入 swa_evictable,空闲时 available+evictable 超出池容量;(2) `cache_unfinished_req` repoint——去重/repoint 按 SWA-window-safe match 定长,但操作的是 full-attention cards,tombstoned 前缀下 match 塌缩为 0 触发 `assert new_prefix_len <= len(new_indices)`;新增 `MatchPrefixParams.return_full_match` 返回最深 device-resident 节点的完整 cards 用于 repoint(仅此调用点启用,准入与复用判定仍要求 SWA 有效匹配)。

## 关键改动

```text
python/sglang/srt/mem_cache/kv_cache_builder.py: SWA decode-radix 仅限 unified tree,拒绝层列出
python/sglang/srt/disaggregation/decode.py: SWA 前缀 cap、tail 预分配、SWA-aware 预算与锁
python/sglang/srt/mem_cache/unified_radix_cache.py: return_full_match 支持 repoint
python/sglang/srt/mem_cache/swa_component.py: 未完成路径转发 swa_evicted_seqlen 修 SWA 泄漏
test/registered/disaggregation/test_disaggregation_decode_radix_cache_swa.py: 新增 gpt-oss-20b 覆盖
```

## 性能声明

PR 正文无吞吐/加速比数字(regex 未检出,正文亦未给出端到端 benchmark)。验证数字(gpt-oss-20b,1P1D,2xL40S,`--page-size 64 --attention-backend triton`):multi-turn cache hit 下 cached tokens 逐轮增长 0 → 1792 → 3584;gsm8k 500 题 2 遍 0.532 / 0.572(第二遍命中缓存);SWA-pool 泄漏 / `cache_unfinished_req` assert 出现 0 / 0。属功能验证,不构成性能加速声明。

## 验证与回归风险

单测:`test_decode_radix_lock_ref.py`(6 passed,新增 SWA tail 长度/可驱逐预算/页对齐回收/拒绝释放锁场景)、`test_unified_radix_cache_unittest.py`(1818 ran,758 skipped)、mocked `build_kv_cache` smoke;集成:`/rerun-group disaggregation` SWA decode-radix 通过 8xH20、hybrid-attention 通过 8xH200;新增 registered 测试在 8-gpu-h200 CI 池(gsm8k 阈值 0.45 + 两遍非回归)。风险:预算计算与锁生命周期改动覆盖所有 decode radix cache 用户(非 SWA 路径也改了 accessor 调用);unified tree 为实验特性;CI 状态显示 Base/AMD 未跑。

## 关联

前序:PR #19746(decode-side radix cache);取代:PR #26218(SWA 支持的旧方案);同区域:hicache / unified-radix-cache 相关工作(编号素材未注明)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
