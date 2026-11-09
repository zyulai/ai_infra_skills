# PR #58154: [Bugfix][Distributed] Preserve symmetric collective outputs

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#58154](https://github.com/vllm-project/vllm/pull/58154) |
| 作者 | jeffnvidia |
| 合并日期 | 2026-10-20 |
| labels | bug, ready, nvidia |
| 热度 | 👍 0 · 💬 31 |
| 分类 | Other |
| bug 类型 | general |

## 一句话结论
NCCL symmetric-memory AllGather/ReduceScatter 直接返回持久注册的 scratch buffer,后续同形状集合操作会覆盖仍存活的结果,静默产生错误输出;返回前 clone 修复。

## 问题现象
symmetric-memory 路径的输出可能与普通集合通信路径不一致,且是静默错误(模型输出悄悄变错)。复现:TP16 Qwen2.5-72B 运行中,受影响的 ReduceScatter 路径与普通 collective 路径在全部 16 条测试 prompt 上都不一致;修复后恢复 16/16 top-1 输出 token 一致。vLLM 0.26.0 对照组复现了两个操作的结果别名(retained-result aliasing)。

## 根因
正文披露:#46703 把这些路径改为复用持久注册的 scratch buffer,却把 buffer 直接返回。同形状同 dtype 的后续集合操作会覆盖先前仍存活的结果——集合操作本身算出的值是对的,但返回的 tensor 没有独立存储(output-ownership bug)。正文还说明 #49194、#55211 分别只覆盖特化 MoE ReduceScatter 和 AllReduce 所有权,均未修复通用的 symmetric ReduceScatter 与单 tensor AllGather。

## 修复方案
`vllm/distributed/device_communicators/cuda_communicator.py`:
1. `_reduce_scatter_symm_mem`:调用方未提供 output 时(记 `scratch_output = output is None`),返回 `output.clone() if scratch_output else output`,调用方自带 output 时仍直接返回;
2. `_all_gather_symm_mem`(单 tensor):返回 `symm_output.clone()`;
3. 更新 `_get_symm_scratch` docstring:明确这些 buffer 是内部 scratch,返回它们会让后续同 key 集合操作覆盖早先的存活结果;
4. `tests/distributed/test_nccl_symm_mem.py` 增加回归测试:保留两次同 key 结果,断言 `data_ptr()` 不同且两值均正确,并在 CUDA graph replay 改变输入下验证同一所有权契约。

```diff
+        scratch_output = output is None
         if output is None:
             output = self._get_symm_scratch(
                 "rs_out", output_shape, input_tensor.dtype, input_tensor.device
             )
         pynccl_comm.reduce_scatter(output, symm_input)
-        return output
+        return output.clone() if scratch_output else output
```

## 影响与回归风险
- 影响窗口:#46703 引入 scratch 复用之后(正文提及,未给具体版本)。
- 回归面:使用 NCCL symmetric-memory 的 generic AllGather(单 tensor)与 ReduceScatter 路径,每次泛型调用新增一次 output 大小的 device-to-device 拷贝与一份新分配,可能带来少量延迟/显存开销;batched AllGather 与现有 AllReduce 不变(本就返回新存储);cached symmetric scratch 与 window registration 保留。调用方显式提供 output 的路径语义不变。

## 关联
- [PR #46703](https://github.com/vllm-project/vllm/pull/46703)(引入 scratch 复用)
- [PR #49194](https://github.com/vllm-project/vllm/pull/49194)(特化 MoE ReduceScatter 的 caller-owned output)
- [PR #55211](https://github.com/vllm-project/vllm/pull/55211)(scratch 增长上界与 AllReduce 所有权)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
