# PR #22312: Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#22312](https://github.com/sgl-project/sglang/pull/22312) |
| 作者 | cs-cat |
| 合并日期 | 2026-04-10 |
| labels | run-ci |
| 热度 | 👍 1 · 💬 5 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
Qwen3.5-27B 走 fallback BA 路径产生非连续 split 视图,GDN Triton 内核按连续布局硬编码 stride 读错内存,精度从 49/50 跌到 3/50;改为显式传 stride 修复。

## 问题现象
服务 Qwen3.5-27B 时出现严重精度回退:回归测试(cs-cat/sglang_is_blind)通过率从 49/50 跌至 3/50。触发条件是模型在 commit 5bdc07d974f6cf236fa765a685453ea5e587a838 之后走 fallback BA 路径,a、b 张量为 split 出来的非连续视图,受影响的是 GDN(gated delta net)Triton 内核的 gating 计算。修复后回归测试恢复到 49/50,与直接 revert 该 commit 的通过率一致。

## 根因
正文披露:fallback BA 路径对 a 和 b 返回非连续的 split 视图,而现有 GDN Triton 内核假设输入是连续布局、硬编码了 batch/token 方向的 stride,导致内核从错误的内存位置读取数据,造成观察到的精度回退。

## 修复方案
让两条 GDN 内核 stride 感知:
- `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`:kernel 新增 `stride_a`/`stride_b` 入参,加载由 `a + off`(硬编码)改为按 batch 维 stride 寻址。
- `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`:为 a 传入 token 轴 stride(取 `a.stride()[-2]`,兼容 GDN [T,HV] 与 KDA [T,HV*K]/[B,T,HV*K] 布局),循环内 `p_a += stride_a` 取代原先 IS_KDA 分支下 `+= HV*K` / `+= HV` 的固定步长。
- 新增 `test/registered/attention/test_gdn_noncontiguous_stride.py`,对比连续张量与非连续 split 视图,覆盖 Qwen3.5-27B TP1/TP2 形状与 KDA 路径。

关键 diff(`fused_gdn_gating_kernel`):
```diff
-    blk_a = tl.load(a + off, mask=mask)
-    blk_b = tl.load(b + off, mask=mask)
+    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
+    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本范围;回归自 commit 5bdc07d974f6cf236fa765a685453ea5e587a838(Qwen3.5 fallback BA 路径)引入。
- 回归面:所有走 `fused_gdn_gating` / `fused_sigmoid_gating_delta_rule_update` 的 GDN 与 KDA 路径。连续输入时新 stride 取值与原硬编码值一致,行为应等价;KDA 循环步长由固定 HV*K/HV 改为统一 stride_a,依赖 a 的实际布局。

## 关联
- 修复目标 issue:#22311
- 回归测试仓库:https://github.com/cs-cat/sglang_is_blind

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
