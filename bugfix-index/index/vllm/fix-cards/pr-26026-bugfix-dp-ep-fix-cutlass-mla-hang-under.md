# PR #26026: [BugFix][DP/EP] Fix CUTLASS MLA hang under load

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#26026](https://github.com/vllm-project/vllm/pull/26026) |
| 作者 | LucasWilkinson |
| 合并日期 | 2025-10-01 |
| labels | bug, deepseek, ready |
| 热度 | 👍 7 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
CUTLASS SM100 MLA kernel 的 compute() 早退路径只 `arrive()` epilogue named barrier 就进入下一轮循环,与 load warps 的 `arrive_and_wait()` 失配死锁;改为 `arrive_and_wait()`。

## 问题现象
DP/EP 负载下 CUTLASS MLA(DeepSeek 相关,SM100)挂死(hang)。正文未描述具体报错或复现步骤,症状为 kernel 死锁、请求无法完成。

## 根因
正文披露:`compute()` 的早退(early return)路径会执行
```cpp
cutlass::arch::NamedBarrier(
    (kNumComputeWarps + kNumLoadWarps) * NumThreadsPerWarp,
    kNamedBarrierEpilogue
).arrive();
```
但在绕回再次调用它之前没有任何 barrier;当 load warps 在
```cpp
NamedBarrier(..., kNamedBarrierEpilogue).arrive_and_wait();
```
上等待时,barrier 的 arrive/wait 参与次数失配,造成死锁。

## 修复方案
改 `csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp` 中 `compute()` 早退路径的一行:
```cpp
-      ).arrive();
+      ).arrive_and_wait();
```
使早退的 compute warps 在返回前也等待 load warps 汇合,barrier 两边计数恢复一致。diff 其余部分为该文件内 tab→空格的缩进规范化,无逻辑变化。

## 影响与回归风险
- 回归面:仅 CUTLASS SM100 MLA warpspecialized kernel 的 epilogue named barrier 用法;所有走该 kernel 的推理路径(如 DeepSeek MLA on SM100)行为一致化,不再可能因 early return 死锁。`arrive_and_wait` 使早退路径多一次等待,理论上对早退 tile 有微小开销。

## 关联
正文未列出(注明 Co-authored-by: Robert Shaw <robshaw@redhat.com>)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
