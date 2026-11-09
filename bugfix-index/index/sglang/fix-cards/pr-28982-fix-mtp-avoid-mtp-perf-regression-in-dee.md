# PR #28982: fix(mtp): avoid mtp perf regression in deepseek when enable eplb

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#28982](https://github.com/sgl-project/sglang/pull/28982) |
| 作者 | ICENacl |
| 合并日期 | 2026-07-09 |
| labels | deepseek, run-ci, bypass-fastfail |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | regression |

## 一句话结论

DeepSeek 同时开 EPLB 与 MTP 时 draft 复用 target 的 dispatch info 致 accept rate 降;is_nextn 下不再初始化。

## 问题现象

DeepSeek 同时启用 EPLB 与 MTP 时,MTP 出现性能回退:正文披露 draft model 复用了 target model 的 dispatch info,导致接受率(accept rate)下降。正文未提供具体的 accept rate / 吞吐数字。

## 根因

正文披露:MTP 的 draft(nextn)模型在执行时错误复用了 target model 的 expert location dispatch info(`ExpertLocationDispatchInfo`),导致 accept rate 下降;修复思路是引入 `is_nextn` 标志区分 draft 与 target,该做法与 Qwen3 MoE 在 `Qwen2MoeSparseMoeBlock._forward_deepep` 中的既有实现一致。

## 修复方案

改 `python/sglang/srt/models/deepseek_v2.py` 四处构造 `ExpertLocationDispatchInfo` 的位置,统一加上 `not self.is_nextn` 条件,使 draft(nextn)模型不再携带 dispatch info:

```diff
         dispatch_info = (
             ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
-            if server_args.enable_eplb
+            if server_args.enable_eplb and not self.is_nextn
             else None
         )
```

四处分别位于 `forward_normal_dual_stream`、`forward_normal`、`forward_deepep` 与 `op_select_expert`;后两处由原来的无条件传参改为三元表达式,`not self.is_nextn` 时才 `init_new`,否则传 `None`。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:DeepSeek(deepseek_v2.py)MoE 的 expert dispatch 路径;行为变化仅体现在 `is_nextn`(MTP draft)分支——EPLB 开启时 draft 不再使用 target 的 dispatch info;target 模型的 EPLB dispatch 行为不变。EPLB 与 MTP 不同时开启的场景不受影响。

## 关联

正文未列出。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
