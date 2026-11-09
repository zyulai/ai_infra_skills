# PR #9838: [BugFix][Kernel] Fix Illegal memory access in causal_conv1d in H100

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#9838](https://github.com/vllm-project/vllm/pull/9838) |
| 作者 | mzusman |
| 合并日期 | 2024-10-31 |
| labels | ready |
| 热度 | 👍 0 · 💬 3 |
| 分类 | Attention / Kernel |
| bug 类型 | crash |

## 一句话结论
H100 上 causal_conv1d forward 越界读 smem_exchange 致非法内存访问;补边界分支并放宽 bf16 测试容差。

## 问题现象
causal_conv1d forward 内核出现 illegal memory access;作者称据其所知只在 H100 机器上显现,因此 #9189 时被遗漏。另外 `test_selective_state_update_with_batch_indices` 的 bfloat16 diff 容差过紧,H100 上 diff 略高导致测试失败。

## 根因
正文只说明是 forward 内核的 illegal memory access,未披露具体机制。由 diff 推断(含新增代码注释):一处是当 `last_thread == kNThreads - 1` 时访问 `smem_exchange[last_thread + 1]` 会造成越界非法访问,现仅当 final state 数据确实落在该处(`(offset + kWidth - 2) >= kNElts` 且 `last_thread + 1 < kNThreads`)才访问;另一处是当 final state 被分到最后两次 `smem_exchange` 之间(`final_state_position` 为非正)时,需要分别从 chunk `n_chunks-2` 的最后线程与 chunk `n_chunks-1` 的线程 0 的 smem_exchange 装载 final state 的两段。

## 修复方案
`csrc/mamba/causal_conv1d/causal_conv1d.cu` `causal_conv1d_fwd_kernel` 两处:1) 新增 `final_state_position < 0 && seqlen > kWidth` 分支,分别从相邻两个 chunk 的 smem_exchange 取 final state 的两段;2) else 分支加守卫:
```cuda
if ((offset + kWidth - 2) >= kNElts && (last_thread + 1 < kNThreads)){
    // In case last_thread == kNThreads - 1, accessing last_thread + 1 will result in a
    // illegal access error on H100.
    // Therefore, we access last_thread + 1, only if the final state data sits there
    reinterpret_cast<vec_t *>(x_vals_load)[1] = smem_exchange[last_thread + 1];
}
```
测试:`test_causal_conv1d` 的 seqlen 参数表加入 1025;`test_causal_conv1d_varlen` 的 final_states 断言改为按 `state_indices` 比较;`test_selective_state_update_with_batch_indices` 的 bfloat16 容差由 `7e-2, 7e-2` 放宽到 `1e-1, 1e-1`(与带 heads 的同类测试一致),diff 打印改用 `out[:batch_size]`。

## 影响与回归风险
- 回归面:causal_conv1d CUDA forward 内核(Mamba 卷积路径)与内核测试;新增分支只在 final state 跨 chunk 边界/最后一线程时生效,常规路径数值不变。

## 关联
#9189(遗漏来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
