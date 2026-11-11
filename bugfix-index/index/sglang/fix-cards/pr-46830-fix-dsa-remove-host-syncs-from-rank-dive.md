# PR #46830: [Fix] dsa: Remove host syncs from rank-divergent eager paths

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#46830](https://github.com/sgl-project/sglang/pull/46830) |
| 作者 | dorado269 |
| 合并日期 | 2026-11-11 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 19 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DSA eager 回退在无 host mirror 时读 GPU 长度,只有部分 rank 阻塞 D2H 同步,集合通信不匹配可挂死通信组;改用静态宽度并去掉多余 host mirror。

## 问题现象
DSA 声明 `needs_cpu_seq_lens=False`,但其 eager 回退在 host mirror 缺失时仍会在主机侧读 GPU 上的长度。在 PD + DP-attention + EAGLE/MTP 下,忙与空闲的 rank 可能走不同的 host-mirror 分支;只在部分 rank 上执行阻塞的 `.max().item()` 或 `.cpu()` 会使这些 rank 无法发出与之匹配的 collective,从而挂死整个通信组。原始报告观察到的是首个被路由请求的死锁;本 PR 的最新证据是 CPU 契约回归测试,而非新的分布式死锁复现。两条问题路径在 main `8a455f1` 上仍存在:无 host mirror 时的 page-table 宽度选择,以及 DRAFT_EXTEND_V2 的 host mirror 物化。

## 根因
正文披露:两条 eager 路径在 host mirror 缺失时直接读 GPU 张量(对 `seq_lens` 做 `.max().item()`、对 `extend_prefix_lens`/`seq_lens` 做 `.cpu()`),在 DP-attention 下与 host-mirror 分支形成 rank 间分歧,阻塞 D2H 同步的那部分 rank 不再参与匹配的 collective,导致组挂死。正文指出历史证据与现状分开报告:生产改动行与原 #33973 的 diff 完全一致,本 PR 是按 main 重新放置并补注册的 CPU 回归测试;原始调查在 8×MI355X + GLM-5.2-FP8(PD + DP attention + EAGLE MTP)上确认首个请求死锁清除。

## 修复方案
`python/sglang/srt/layers/attention/dsa_backend.py` 的 `init_forward_metadata`:
- 无 host 长度 mirror 时改用静态请求表宽度,不再读 GPU 长度;per-row 边界仍用真实 GPU sequence lengths,top-k 用 `cache_seqlens` 掩掉未使用的列;host mirror 已存在时保留较窄宽度;
- DRAFT_EXTEND_V2 直接使用 `prepare_for_draft_extend` 提供的 extend 长度 metadata,不再构造未使用的 host mirror(删除 `extend_prefix_lens.cpu().tolist()` 与 `seq_lens.cpu()`),断言改为只要求 `extend_seq_lens{,_cpu}`;
- 已有 graph 行为不变。新增 `test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py`,跑真实 metadata 入口、拒绝对 device 长度输入的 host 读取、检查 decode/draft page table,并保留 host-mirror 对照;rebase 后 CPU 结果 3 passed + 2 subtests,未打补丁的 main 为 2 expected failures + 1 passed + 2 subtests。

关键摘录:

```diff
-            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num
+            # Use the static width to avoid rank-divergent D2H syncs.
+            # Top-k masks unused columns by cache_seqlens.
+            max_seqlen_k = self.req_to_token.shape[1]
```

## 影响与回归风险
- 影响窗口:正文未给出明确版本范围,仅说明两条问题路径在 main `8a455f1` 上仍存在;历史部署验证(死锁清除)来自被替换的 #33973。
- 回归面:无 host mirror 时 decode/draft eager 的 page-table 宽度由「max 长度」变为静态表宽,历史 MI300X 对比记录了两种宽度下 top-k 真实选择与分数一致(per-row masking 下短于 top-k 的序列在两种宽度都有 surplus -inf slot);DRAFT_EXTEND_V2 的 host mirror 构造被移除;prefix 检查/ragged dispatch 仍仅限普通 EXTEND,`seq_lens_sum` 的自动 prefill 选择用途仍由 `ForwardMode.EXTEND` 保护。正文声明:未在本 head 做全新的 PD/DP/MTP 硬件复跑,broader AMD/CUDA 回归、长上下文内存验证与 maintainer-enabled CI 仍是后续覆盖,不声称新的吞吐或全栈精度结果。CI 状态节另提及仓库级 lint 在未改动的 main XPU graph runner 上因 `F821 logger` 失败(上游既有问题,与本 PR 无关)。

## 关联
- [#33973](https://github.com/sgl-project/sglang/pull/33973)(被本 PR 替换的 stale draft,原始调查与硬件验证)
- [#30980](https://github.com/sgl-project/sglang/pull/30980)(上游 lint 报告与复现评论)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
