# PR #17574: [BugFix][Attention] Fix sliding window attention in V1 giving incorrect results

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#17574](https://github.com/vllm-project/vllm/pull/17574) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-05-02 |
| labels | ready, v1 |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
FA3 AOT 调度器未正确处理 sliding window,V1 滑窗注意力结果错误;AOT 调度补传 window_size,混合滑窗配置则退回非 AOT。

## 问题现象
正文说明:V1 的 sliding window attention 给出错误结果,根因是 FA3 改用 AOT 调度器的更新(#13111)没有正确处理 sliding window attention。修复 issue #17476。正文未给出更具体的复现细节,验证方式为 `tests/v1/e2e/test_correctness_sliding_window.py`。

## 根因
正文披露:FA3 AOT 调度器改造(#13111)未正确处理 sliding window attention。(由 diff 推断)AOT 调度路径构造 batch schedule 时根本没有传 `window_size`,滑动窗口被当作全量因果注意力计算,于是结果错误;且滑窗大小属于每层参数,与 AOT 调度要求"整批常量"的前提冲突。

## 修复方案
`vllm/v1/attention/backends/flash_attn.py`:
- 新增 `_get_sliding_window_configs(vllm_config)`,遍历模型所有 `Attention` 层收集 `FlashAttentionImpl.sliding_window` 配置集合。
- `FlashAttentionMetadataBuilder.__init__` 不再直接置 `aot_schedule`,新增 `self.aot_sliding_window`,延迟到首次 `build()` 时填充(此时层才构造完成):唯一滑窗配置则采用之,出现多于一种配置则 `aot_schedule = False` 退回非 AOT 调度。
- AOT 调度的 `schedule` 闭包补传窗口参数:

```diff
                     page_size=self.page_size,
                     cu_seqlens_q=cu_query_lens,
                     causal=causal,
+                    window_size=self.aot_sliding_window,
                 )
```

## 影响与回归风险
- 影响窗口:#13111(FA3 AOT 调度)合入起至本 PR 的 V1 FlashAttention 路径。
- 回归面:V1 FlashAttention(FA3)metadata 构建:使用滑窗的模型(单一滑窗配置)走 AOT 时结果改变(修复);多滑窗配置模型改为退回非 AOT 调度(可能影响该场景吞吐);无滑窗模型 `aot_sliding_window=(-1,-1)` 不变。

## 关联
- #17476(被修复的 issue)
- #13111(引入问题的 FA3 AOT 调度器 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
