# PR #52771: [Bugfix] OffloadingConnector: stop zeroing offload hits under MTP/EAGLE spec decode

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @kamb-code |
| 合并 | 2026-09-07 |
| 分类 | Speculative Decoding (tier: major) |
| Label | bug, ready, kv-connector |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/52771 |

## 一句话结论

修复三个叠加缺陷:混合 KV 配置下开 MTP/EAGLE 后 OffloadingConnector 把 KV 存入 CPU 层却查出 0 个可复用 token,KV offload 命中被清零。

## 动机

issue #52735:受影响的 hybrid 配置中,启用 MTP/EAGLE trailing-block 处理后 `OffloadingConnector` 正常向 CPU 层存储 KV,但 lookup 返回零可复用 token——prefix cache offload 在 spec decode 下形同虚设。根因是三个缺陷叠加:(1) drafter 组回退过宽;(2) volatile-tail pop 未对非滑动窗口组加宽查询;(3) decode 期尾块排除造成永久存储空洞。独立硬件报告复现了该缺陷(H100 NVL 在 stock v0.27.1 上、MTP-off 对照组正常)。

## 优化点

三处修复:(1) `SchedulerOffloadConfig.from_spec`——trailing-block dropping 启用但无 `is_eagle_group` 注解时(shared-group MTP 模型的常态,如 Qwen3.5/3.8 式 drafter 是合并进目标 full-attention 组的普通 decoder 层),原代码把所有组都当 drafter 组,volatile-tail pop 作用于全部组,任一组可服务窗口只有一个 chunk 时整个请求的 offload hit 清零。修复为保持 drafter 组集合为空并 `info_once` 记日志;drafter KV 服务慢一个 chunk 只会降低接受率(每个 draft 都被 target 验证),即 "fail toward serving";`use_eagle_block_drop()`(#53388)能力仍被尊重。(2) `_lookup_complete_chunks`——旧注释称加宽仅需滑动窗口组,但 un-widened pop 会把 `max_hit_size_tokens` 缩到低于查询量,使确认边界落到更粗粒度 sibling 组(如 mamba-align)的 chunk 粒度之下而返回 0;加宽现对所有 drafter 组生效,pop 落在被多查的一个 chunk 上,边界保持。(3) `storable_chunks`——decode 期对 drafter 组尾块的排除在 `next_stored_chunk_idx` 派生不一致时会永久跳过最后一个可存 chunk;修复为请求 `is_finished()` 后(不再有 spec 拒绝可改写尾部)解除排除,finish 时补存尾块;lookup 侧 legacy pop 保持不变。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py: from_spec 移除 all-groups 回退;storable_chunks 完成时解除尾块排除;_lookup_complete_chunks 对所有 drafter 组加宽查询
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py: TestSharedGroupMTPOffload 与 TestMambaHybridOffloadServing 回归套件
tests/v1/kv_connector/unit/offloading_connector/utils.py: RequestRunner 支持 speculative_config 注入
```

关键摘录(from_spec 的修复):

```python
if use_eagle_block_drop and not eagle_groups:
    logger.info_once(
        "KV offloading: speculative decoding is enabled but no "
        "KV-cache group is annotated as a drafter group; treating "
        "all groups as non-draft for offloading.")
```

## 性能声明

修复前后数据来自 issue 线程的独立硬件报告(非提交者运行):B70 eviction 后 18.9 s → 2.62 s(7.2x);GB200 5.43 s → 0.95 s;H200 重置 local APC 后 15.1 s → 0.6 s。另有一套 4x DGX Spark + NVMe offload 层的部署独立实现了相同修正并测得:每 100K prompt 的存储块从 339→389(共 390),eviction/重启后的 reload 从 335 s 降到 3–8 s。B70 报告无输出变化但未做受控接受率 A/B;H200 报告 greedy 补全 byte-identical;GB200 报告聚焦性能。测试条件与 workload 细节以 issue #52735 原帖为准。

## 验证与回归风险

测试:offloading connector 单元套件(CPU-only 主机,`support_hybrid_kv_cache` stub 为 True)318 passed / 2 skipped;在相同 base 的隔离副本中仅回退三处生产改动即可让受影响的 8 个 sync/async 用例按预期断言失败;`ruff-check`/`ruff-format` 通过。安全依赖:#52807(独立 partial-hit 断言修复,合并后套件 319 passed)与 #54288(finish-time offload 不把末尾采样 token 当已写 KV,改动同一 finish watermark 与本 PR 扩展的三处期望,组合后 318 passed)——二者建议先于或随本 PR 合入。回归风险:(1) 若确有未注解的独立 drafter 组,可能改变 draft proposal 与接受率(被接受 token 仍由 target 验证);(2) 加宽查询增加少量 lookup 开销;(3) finish 时补存尾块略增 offload 存储量;(4) 明确非以下 PR 的重复:#47890/#47891/#48459、#52047、#41640、#50897、#53388、#54637。

## 关联

Fixes #52735;安全依赖 #52807、#54288;同区域:#53388(可配置 drop)、#50897(successor-aware hashes)、#52047(Kimi-K3 DSpark 注解)、#47891/#48459(DSpark ephemeral 组)。与 PR #54442 同属 spec decode 下 KV/缓存的正确性区域。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
