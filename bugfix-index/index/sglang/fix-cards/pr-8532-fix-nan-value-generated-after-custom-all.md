# PR #8532: Fix nan value generated after custom all reduce

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#8532](https://github.com/sgl-project/sglang/pull/8532) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-07-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论

custom all reduce 输出/元数据缓冲未初始化产生 NaN,全部改用 zeros 初始化修复,gsm8k 精度恢复。

## 问题现象

作者在 custom all reduce 之后发现生成了 NaN 值,并导致 gsm8k accuracy 测试失败。正文未给出更详细的复现步骤或日志。

## 根因

正文未披露详细根因分析,但 Modifications 一节写明修复思路是「Change the empty_like and empty to zeros_like and zeros to ensure the tensor value is initialized to zero」,即相关张量此前未初始化为零。

从 diff 描述改动所暗示的机制(由 diff 推断):`custom_all_reduce.py` 中 `rank_data`、`buffer`(元数据缓冲)以及 `all_reduce_reg`/`all_reduce_unreg`/`all_reduce`/`custom_all_reduce` 的输出张量均用 `torch.empty`/`torch.empty_like` 分配,内容为未初始化内存;custom all reduce 的部分路径(如 warm-up 模拟分配、`out` 未显式传入时的输出缓冲)会把这些未清零的字节当作数据使用,从而产生 NaN。

## 修复方案

在 `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` 中,将所有 `torch.empty`/`torch.empty_like` 分配改为 `torch.zeros`/`torch.zeros_like`,共 7 处:

- 两处 `rank_data`(8MB uint8,分支与 MI200 meta buffer 分支)由 `torch.empty` 改为 `torch.zeros`;
- MI200 分支的 `self.buffer` 由 `torch.empty` 改为 `torch.zeros`;
- `all_reduce_reg`、`all_reduce_unreg`、`all_reduce` 中 `out is None` 时的输出张量由 `torch.empty_like(inp)` 改为 `torch.zeros_like(inp)`;
- `custom_all_reduce` warm-up 模拟分配路径的返回值由 `torch.empty_like(input)` 改为 `torch.zeros_like(input)`。

作者验证:gsm8k(`--num-questions 2000 --parallel 2000`)Accuracy 0.948、Invalid 0.000、Output throughput 2409.507 token/s。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:所有使用 custom all reduce 的分布式推理路径;零初始化只在分配时多一次写入,8MB 级缓冲的初始化开销很小,但理论上引入少量分配期开销;行为上消除对未初始化内存的依赖。涉及的输出张量在 `ops.all_reduce*` 之后被完全覆盖,零初始化主要影响未被覆盖即被读取的路径(warm-up 返回值等)。

## 关联

正文未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
