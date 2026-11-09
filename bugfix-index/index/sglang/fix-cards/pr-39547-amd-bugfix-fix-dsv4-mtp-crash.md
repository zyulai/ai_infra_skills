# PR #39547: [AMD][bugfix] Fix DSV4 MTP crash

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#39547](https://github.com/sgl-project/sglang/pull/39547) |
| 作者 | RolaoDenthu |
| 合并日期 | 2026-09-16 |
| labels | deepseek |
| 热度 | 👍 3 · 💬 0 |
| 分类 | Speculative Decoding |
| bug 类型 | crash |

## 一句话结论
#39116 在 `repeat_interleave` 里去掉了 D2H 同步,AMD DSV4 开 MTP 即 GPU 访存 fault;把该跳过限制在 `is_dspark`,MTP 恢复同步。

## 问题现象
AMD 上运行 DeepSeek-V4(DSV4)并开启 MTP(speculative decoding)时,进程以 GPU memory access fault 崩溃。正文未给出完整复现脚本与日志,仅指出崩溃由 #39116 引入。

## 根因
正文披露:#39116 在 `repeat_interleave` 相关路径里删掉了一次 D2H(device-to-host)同步。diff 中新增注释进一步说明机制:"output_size drops the implicit sum()'s D2H in the prefill-meta build. EAGLE verify faults without that sync"——即 prefill 元数据构建时 `output_size` 依赖隐式 `sum()` 造成的 D2H 同步,跳过它之后 EAGLE verify(target verify)路径在没有同步的情况下运行而出访存错误。哪条具体 kernel 访问越界,正文未进一步披露。

## 修复方案
`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` 的 `init_forward_metadata_target_verify_old`:把"跳过 D2H 同步"的行为用 `is_dspark` 门控——只有 dspark 路径继续享受免同步,使用 MTP 时恢复该同步:

```python
        # output_size drops the implicit sum()'s D2H in the prefill-meta build.
        # EAGLE verify faults without that sync.
        exact_num_tokens = exact_num_tokens and self.is_dspark
```

正文注明改动意图:"gate the D2H sync skip behind `is_dspark` and keep it when using MTP"。

## 影响与回归风险
- 影响窗口:正文未给版本/提交范围,仅指明回归由 #39116 引入(其合并之后、本 PR 之前的 main)。
- 回归面:仅涉及 AMD HIP radix DSV4 后端的 target-verify 元数据构建路径;dspark 路径保持原跳过同步的快速路径不变,非 dspark(如 MTP/EAGLE verify)恢复 D2H 同步,该路径可能因此多一次同步开销(diff 所示改动的直接后果);模型输出不因此改变,正文附 GSM8K 1319 题 0.952 验证。

## 关联
- https://github.com/sgl-project/sglang/pull/39116

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
