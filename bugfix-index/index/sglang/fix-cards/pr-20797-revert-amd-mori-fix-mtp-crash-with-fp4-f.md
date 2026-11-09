# PR #20797: Revert "[AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars."

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#20797](https://github.com/sgl-project/sglang/pull/20797) |
| 作者 | Qiaolin-Yu |
| 合并日期 | 2026-03-18 |
| labels | documentation, deepseek |
| 热度 | 👍 1 · 💬 4 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论

#20647 的 MORI FP4/FP8 dispatch 修复导致 CI job 失败;整体 revert,移除 NEXTN dispatch 环境变量与 `is_nextn` 通路,恢复原 dispatch 量化逻辑。

## 问题现象

#20647([AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars)合入后造成 CI 失败,正文给出的证据是失败的 job 链接:[run 23083366420 job 67452685234?pr=20469](https://github.com/sgl-project/sglang/actions/runs/23083366420/job/67452685234?pr=20469)。正文未粘贴具体报错,受影响面为 AMD MORI EP dispatch 相关路径及 `deepseek_v2` 的 MTP/NextN 专家层构造。

## 根因

正文未披露 #20647 导致失败的具体机制,仅以 CI 失败为由整体 revert。(由 diff 推断)#20647 在 `run_moe_core` 中新增了 BF16 权重(`is_bf16_weights`)与 FP8/FP4 dispatch 组合的去量化分支,并把 `is_nextn` 参数沿 `MoeRunnerConfig` → `create_moe_dispatcher` → `MoriEPDispatcher` 一路传入,按 `SGLANG_MORI_NEXTN_FP8_DISP`/`SGLANG_MORI_NEXTN_FP4_DISP` 为 NextN 层单独选择 dispatch 量化类型;这套新分支在 CI 环境下引入了错误行为,故回退。

## 修复方案

revert #20647 全部改动:

- `python/sglang/srt/layers/moe/ep_moe/layer.py` `run_moe_core`:删除 `is_bf16_weights` 判断及 BF16 权重 + FP8/FP4 dispatch 的去量化分支,恢复仅 `is_quark_w4a4` / `is_fp8_quant` 的原逻辑;
- `fused_moe_triton/layer.py`、`moe_runner/base.py`、`token_dispatcher/moriep.py`、`models/deepseek_v2.py`:移除 `is_nextn` 参数链与 `MoeRunnerConfig.is_nextn` 字段;
- `moriep.py`:`_get_mori_dispatch_quant_flags` 恢复为无参版本,删除 NEXTN 专用环境变量分支,`init_mori_op` 恢复原签名并在内部自行读取量化 flags,`dispatch_a`/`dispatch_b` 恢复读实例属性;
- `docs/references/environment_variables.md`:删除 `SGLANG_MORI_NEXTN_FP8_DISP`、`SGLANG_MORI_NEXTN_FP4_DISP` 两行文档。

## 影响与回归风险

- 影响窗口:自 #20647 合并起至本 revert。
- 回归面:AMD MORI EP dispatch 的量化选择路径、MTP/NextN 层的 dispatcher 构造、相关环境变量与文档。NEXTN 独立 dispatch 量化配置能力随 revert 消失,需配置 NEXTN dispatch 的用户回到共享 `SGLANG_MORI_FP8_DISP`/`SGLANG_MORI_FP4_DISP` 的行为;#20647 标题所述的 MTP + FP4/FP8 dispatch 崩溃场景在其原触发条件下可能重新存在(依据被 revert PR 的标题,正文未展开)。

## 关联

- #20647(被 revert 的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
