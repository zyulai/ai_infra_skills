# PR #52771: [Bugfix] OffloadingConnector: stop zeroing offload hits under MTP/EAGLE spec decode

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#52771](https://github.com/vllm-project/vllm/pull/52771) |
| 作者 | kamb-code |
| 合并日期 | 2026-09-07 |
| labels | bug, ready, kv-connector |
| 热度 | 👍 3 · 💬 25 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
混合配置开 MTP/EAGLE 后 OffloadingConnector 存了 CPU 端 KV 却返回 0 复用 token;修三处缺陷恢复 offload 命中。

## 问题现象
在受影响的 hybrid 配置下,启用 MTP/EAGLE trailing-block 处理后,`OffloadingConnector` 会把 KV 正常存入 CPU/offload 层,但 prefix 复用查找返回 0 个可复用 token——存进去却完全取不出来。社区独立复现:B70 在 eviction 后首查 18.9s→2.62s(7.2×)、GB200 5.43s→0.95s、H200 15.1s→0.6s(重置本地 APC 后);H100 NVL 在 stock `v0.27.1` 镜像上带 MTP-off 对照组复现该缺陷(#52735)。

## 根因
正文披露三个缺陷叠加:
1. **All-groups drafter fallback**(`SchedulerOffloadConfig.from_spec`):开启 trailing-block drop 且无 `is_eagle_group` 标注时(shared-group MTP 模型的常态,Qwen3.5/3.8 风格 drafter 是并入目标组的普通 `full_attention` 层),所有组都被当成 drafter 组。
2. **Volatile-tail pop 未对非滑窗组扩查询**(`_lookup_complete_chunks`):旧注释称扩查询只需用于滑窗组,对组自身成立,但对其兄弟组不成立——未扩宽的 pop 使 `max_hit_size_tokens` 缩到低于实际查询量,把 confirmed boundary 压到更粗粒度组(如 mamba-align 组)的 chunk 粒度之下,整个请求返回 0。
3. **永久尾块存储空洞**(`storable_chunks`):decode 期排除逻辑可能永久留下最后一个本可存储的完整 drafter chunk 未发布。

## 修复方案
三处对应修复,均在 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:
1. 删除 fallback:drafter 组集合留空并 `logger.info_once` 提示一次;
2. 扩查询应用于所有 drafter 组:`if is_eagle_unverified and sliding_window_size_in_chunks is not None:` 改为 `if is_eagle_unverified:`;
3. 请求结束后解除 decode 期排除:

```python
# Finished requests have no pending speculation.
if group_config.is_eagle_group and is_decoding and not self.req.is_finished():
    num_chunks = max(0, num_chunks - 1)
```

测试侧新增 `TestSharedGroupMTPOffload`、`TestMambaHybridOffloadServing` 回归用例;反转三处生产改动后同样的 8 个用例按预期失败。

## 影响与回归风险
- 影响窗口:正文提及 H100 NVL 在 stock `v0.27.1` 镜像上可复现缺陷。
- 回归面:若确实存在独立 drafter 组但缺标注,移除 fallback 可能改变 draft 提案与接受率(accepted token 仍经 target 验证);legacy lookup 侧 pop 未改动,abort 场景的尾块发布未单独断言。与 #52807、#54288 有交互,#54288 应先合或与本 PR 组合重跑。CPU-only 主机套件 318 passed / 2 skipped;B70/H200 报告未见输出变化,但未做受控接受率 A/B,重 base 后未重跑 GPU kernel 路径。

## 关联
- Fixes [#52735](https://github.com/vllm-project/vllm/issues/52735)(根因链:[comment](https://github.com/vllm-project/vllm/issues/52735#issuecomment-5328641909))
- [#52807](https://github.com/vllm-project/vllm/pull/52807)、[#54288](https://github.com/vllm-project/vllm/pull/54288)(安全依赖)
- [#53388](https://github.com/vllm-project/vllm/pull/53388)、[#50897](https://github.com/vllm-project/vllm/pull/50897)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
