# PR #24694: Fix NCCL deadlock in Ulysses SP when sequence length has remainder

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24694](https://github.com/sgl-project/sglang/pull/24694) |
| 作者 | storyicon |
| 合并日期 | 2026-05-09 |
| labels | run-ci, diffusion |
| 热度 | 👍 2 · 💬 5 |
| 分类 | Parallelism / Communication |
| bug 类型 | hang |

## 一句话结论
Ulysses SP 下音视频序列长度不能整除 sp_world_size 时,有 padding 的 rank 多调一次 `all_gather(mask)` 导致 NCCL 死锁;改为无 padding 也返回全 True mask,统一各 rank 集合操作路径。

## 问题现象
Ulysses sequence-parallel(SP)模式下,当音频/视频序列长度不能被 `sp_world_size` 整除时发生死锁(deadlock)。具体表现:有 padding 的 rank 阻塞在 `all_gather(mask)` 上等待其他 rank 加入集合操作,而那些无 padding 的 rank 永远不会调用它,进程挂起。

## 根因
正文披露:`_build_ltx2_sp_padding_mask` 在某 rank 没有 padding token(`valid >= seq_len`)时返回 `None`,而有 padding 的 rank 返回 mask Tensor。`USPAttention.forward` 以 `attn_mask is not None` 分支,导致不同 rank 执行不同的 NCCL 集合操作序列:无 padding rank(mask=None)跳过 `all_gather(mask)`,直接本地 attention 后做 all-to-all 输出;有 padding rank(mask=Tensor)先 `all_gather(mask)` 再 attend、再 all-to-all。集合操作参与者不一致(padding rank 卡在 all_gather 等别人、别人永远不来),造成 NCCL 死锁。

## 修复方案
移除 `_build_ltx2_sp_padding_mask` 中 `valid >= seq_len → None` 的提前返回:无 padding 时也返回一个全 1(全 True)的 mask Tensor,让所有 rank 走相同的 masked-attention 路径。全 1 mask 与 `None` 语义等价(没有任何 token 被 mask 掉),模型输出不受影响,只是保证各 rank 的 NCCL 集合操作一致。

修改两个文件(`python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py` 与 `.../layers/attention/layer.py`):

关键 diff 摘录(ltx_2_denoising.py):

```diff
-        if valid <= 0 or valid >= int(seq_len):
-            return None
-        mask = torch.ones(
-            (batch_size, int(seq_len)), device=device, dtype=torch.float32
-        )
-        mask[:, valid:] = 0.0
+        mask = torch.ones((batch_size, int(seq_len)), device=device, dtype=torch.bool)
+        if valid < int(seq_len):
+            mask[:, max(0, valid) :] = False
         return mask
```

layer.py 仅增加一段注释,提示:若此处出现 NCCL timeout/deadlock,应检查 attn_mask 是否在各 SP rank 间不一致(部分 None、部分 Tensor)导致 all_gather 参与者不匹配,上游 mask builder 必须保证所有 rank 产出相同类型的 mask。另注意 mask 类型从 float32(1/0)改为 bool(True/False)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响 multimodal_gen 中 LTX-2 去噪阶段的 SP padding mask 构造与 `USPAttention` 的 mask 路径:原先返回 None 的无 padding 情况现在返回 bool Tensor,所有 rank 走 all_gather(mask) 路径,SP 组内多一次 mask 的集合通信开销;mask dtype 由 float32 变 bool,下游 SDPA 后端对 bool mask 的语义(True=保留还是 True=屏蔽)需与实现一致。序列长度可整除 `sp_world_size` 的场景行为不变。

## 关联
正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
