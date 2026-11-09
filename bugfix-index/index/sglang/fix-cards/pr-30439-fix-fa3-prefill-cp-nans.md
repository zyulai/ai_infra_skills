# PR #30439: Fix FA3 prefill CP NaNs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#30439](https://github.com/sgl-project/sglang/pull/30439) |
| 作者 | Fridge003 |
| 合并日期 | 2026-07-08 |
| labels | (none) |
| 热度 | 👍 0 · 💬 30 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论
FA3 prefill-CP 在 Qwen3 MoE + zigzag CP + GQA 混合批次上突发 NaN/IMA:预计算 scheduler metadata 与跨 rank 实时 cache_seqlens/num_splits 不一致致 split-KV combine 越界;CP 下禁用预计算并清理陈旧 CP-v2 元数据。

## 问题现象
FA3 prefill-CP burst 复现:Qwen3 MoE、zigzag CP、GQA 风格的 prefix-hit 与 long-prefill 混合批次下,出现 NaN/IMA(illegal memory access)突发失败。验证记录:8-sequence burst repro 跑 96 轮,开/不开 `SGLANG_ENABLE_CP_V2=1` 均日志干净、无 NaN/assert/illegal/CUDA-error 签名(修复后);H200 devbox `baizhou-dev` 上 `test_cp_strategy_unit.py` 11 passed、`test_merge_state_v2.py` 150 passed。

## 根因
正文 Root Cause 披露:burst 复现失败于 FA3 split-KV combine——预计算(precompute)的 scheduler metadata 已陈旧,而实时 CP/decode 元数据使各 rank 的 `cache_seqlens`/`num_splits` 发生变化,二者不一致导致 combine kernel 越界访问(`flash_fwd_combine_launch_template.h:52` 的 OOB read,见 diff 注释)。让 `scheduler_metadata` 保持未设置,FA3 会退回逐层(per-layer)metadata 路径,避免 OOB。作者还排查了正文怀疑的 `merge_state` 路径与 #28421 的 zigzag padding 注释:merge-state 改动被回滚、H200 devbox 恢复修复前 `common_ops.abi3.so` 后复跑 burst,确认 merge_state 非本因;padding 疑虑是独立的 CP-v2/GQA 安全项,不需要用来修这个 burst。

## 修复方案
- `flashattention_backend.py`:抽出的 `_should_disable_scheduler_metadata_precompute(server_args)` 改为 `enable_prefill_cp or enable_dp_attention` 时返回 True,即 prefill CP(`attn_cp_size > 1`,见 Summary)与 DP attention 一样跳过 FA3 scheduler_metadata 预计算(PR #21104 引入),走 per-layer metadata 路径。
- `eager_runner.py::_execute_extend`:eager runner 回退到非 CP-v2 正常路径时,清掉残留的 `forward_batch.attn_cp_metadata = None`,避免使用陈旧 CP-v2 元数据。
- Summary 还列出:把 zigzag CP-v2 排除在不安全支持它的 `ForwardMode.MIXED` 之外,并为 FA3 metadata 守卫与 MIXED 拒绝新增 CP 单测(这两项在 pack 提供的 diff 中不可见)。

关键 diff(flashattention_backend.py):

```python
def _should_disable_scheduler_metadata_precompute(server_args) -> bool:
    return bool(server_args.enable_prefill_cp or server_args.enable_dp_attention)
```

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围。
- 回归面:开启 prefill CP 的 FA3 部署不再使用 scheduler_metadata 预计算,统一走逐层 metadata 路径(该路径已有,可能影响预计算带来的性能收益,正文未测性能);eager runner 非 CP-v2 回退路径会清空 attn_cp_metadata;DP attention 行为与修复前一致(条件逻辑等价改写)。

## 关联
- https://github.com/sgl-project/sglang/pull/28421
- https://github.com/sgl-project/sglang/pull/21104

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
