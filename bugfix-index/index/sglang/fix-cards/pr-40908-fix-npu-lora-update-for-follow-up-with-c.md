# PR #40908: [Fix] [NPU] LoRA: Update for follow-up with changes in sgl-kernel-npu

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#40908](https://github.com/sgl-project/sglang/pull/40908) |
| 作者 | vlserov |
| 合并日期 | 2026-10-20 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论

sgl-kernel-npu #825 改了 `sgemmv_shrink`/`sgemmv_expand` 接口;NPU LoRA 后端同步传 `n_slices` 并转换输入 dtype,适配新接口。

## 问题现象

sgl-kernel-npu 侧 PR #825 已合入 main,改变了 LoRA 相关算子接口,导致 SGLang 的 Ascend NPU LoRA 后端与之不兼容,需要跟随更新。正文未描述用户可见的具体症状(如 hang 的触发条件与日志);卡片元数据将 bug 类型标为 hang。

## 根因

按 PR 正文:根因是接口变更——sgl-kernel-npu #825 的改动要求 "For sgemmv_shrink have to change interface in order to support different slicing"(sgemmv_shrink 需要改接口以支持不同的切片方式)。SGLang 侧调用点未传新参数,与已合入的内核新接口不匹配。具体报错或 hang 机制正文未披露。

## 修复方案

改 `python/sglang/srt/lora/backend/ascend_backend.py`:

- `run_qkv_lora` 与 `run_gate_up_lora` 中,调用 `torch.ops.npu.sgemmv_shrink`(经 batch info 的调用点)时新增传入 `n_slices` 参数;qkv 路径使用既有切片数,gate_up 路径显式置 `n_slices = 2`(gate/up 两路):

  ```diff
              self.batch_info.lora_ranks,
              self.batch_info.scalings,
              lora_a_output,
  +            n_slices,
          )
  ```

- `run_lora_b_sgemm` 中 `torch.ops.npu.sgemmv_expand` 的输入 `x` 显式转为 `x.to(dtype=torch.float)`。

## 影响与回归风险

- 影响窗口:正文未给出版本范围;对应 sgl-kernel-npu #825 合入 main 之后的版本窗口。
- 回归面:Ascend NPU 的 LoRA 前向(qkv / gate_up / lora_b sgemmv 三处调用)。`n_slices` 只在 gate_up 路径硬编码为 2,qkv 路径取值来自既有变量(diff 上下文未展示其定义);`x.to(dtype=torch.float)` 会在 NPU 上额外产生一次 dtype 转换。验证为 `test_npu_lora_qwen3_6_27b_logprob_diff` 的 logprob 对比(KL(sglang, trainer)=1.069917e-03,输出 "ok"),未附吞吐数据。

## 关联

- sgl-kernel-npu #825(接口变更来源):https://github.com/sgl-project/sgl-kernel-npu/pull/825

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
