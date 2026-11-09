# PR #27945: fix(moe): make FlashInfer A2A robust to collapsed global_num_tokens (moe_dense_tp_size NaN)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27945](https://github.com/sgl-project/sglang/pull/27945) |
| 作者 | YAMY1234 |
| 合并日期 | 2026-06-12 |
| labels | run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | MoE |
| bug 类型 | correctness |

## 一句话结论
`--moe-dense-tp-size 1` + dp-attention + flashinfer A2A 时 global_num_tokens 被折叠成本地计数,各 EP rank 的 A2A 几何不一致读写出错槽位产生 NaN;修复为该情形回退用静态 max_num_tokens。

## 问题现象
`--moe-a2a-backend flashinfer` 与 `--enable-dp-attention` 组合下,设置 `--moe-dense-tp-size 1` 会静默产出 NaN/inf logits:gsm8k 得分约 0,或采样时报 `probability tensor contains inf/nan` 的 device-side assert。FlashInfer CuteDSL/Cutlass MoE kernel 本身正确,损坏来自每 rank 的 A2A dispatch 尺寸不一致。正文给出复现命令(Qwen3.5-397B-A17B-NVFP4,4x Blackwell,tp4/dp4/ep4 + dp-attention + flashinfer_cutedsl + flashinfer A2A),全量 gsm8k(1319 例,8-shot)修复前 crash(`probability tensor contains inf/nan`)/ 0.0,修复后 **0.977**。

## 根因
正文披露:`moe_dense_tp_size` 被设置时 `require_mlp_tp_gather()` 返回 False,scheduler 在 `batch.global_num_tokens` 里只存本 rank 的 token 数而非 all-gather 后的列表(对应 `scheduler_components/dp_attn.py` 里已有的 `TODO: handle the case when moe_dense_tp_size != 1`)。随后 `FlashinferDispatcher.dispatch()` 判断 `len(dp_global) > 1 == False`,走进本为 `dp_size=1`/SP 准备的 else 分支,把 `runtime_max_tokens_per_rank` 设为 `x.shape[0]`。在 EP 下该本地计数各 rank 不同,固定几何的 `MoeAlltoAll`(combine 时 reshape 成 `[ep_size, runtime_max_tokens_per_rank, hidden]`)在各 rank 间约定不一致,读写到错位的槽位,产生 NaN。只有 FlashInfer dispatcher 读 `get_dp_global_num_tokens()`;DeepEP/Mooncake/MoRI/NIXL 用静态 `num_max_dispatch_tokens_per_rank` 定尺寸,不受影响(所以 `moe_dense_tp_size=1` 在它们上本来就能用,如 DeepSeek)。

## 修复方案
修改 `python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py` 的 `dispatch()`:当 `ep_size > 1` 且 `require_mlp_tp_gather(get_global_server_args())` 为 False(正是导致 global_num_tokens 被折叠的条件)时,回退使用 `self.max_num_tokens`(初始化即固定的每 rank dispatch 容量,所有 rank 相同且为 workspace 上界),而非每 rank 不一致的 `x.shape[0]`。CUDA-graph 安全(初始化时固定,无 device sync)。关键 diff:

```python
+        elif self.ep_size > 1 and not require_mlp_tp_gather(get_global_server_args()):
+            # require_mlp_tp_gather is False, so the scheduler collapsed
+            # global_num_tokens to the local count; x.shape[0] then differs
+            # across EP ranks and breaks the fixed-geometry MoeAlltoAll. Use
+            # the static all-rank capacity instead.
+            self.runtime_max_tokens_per_rank = self.max_num_tokens
```

正常的多 rank 路径(`max(dp_global)`)与 `dp_size=1`/SP 路径不变。diff 另含一处无关的 protocol.py 返回类型注解修正(`-> "ResponseTool"` 改 `-> ResponseTool`)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:仅 FlashInfer token dispatcher:仅在 `ep_size>1` 且 `require_mlp_tp_gather` 为 False(即 moe_dense_tp_size 已设置)时改用静态容量,恰为之前损坏的场景;该路径下 A2A workspace 按静态容量分配,若实际 token 数远小于容量会有冗余预留;其余路径不变。

## 关联
正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
