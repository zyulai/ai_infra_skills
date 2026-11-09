# PR #3413: [Fix] Fix accuracy bug and refactor codes for lora

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3413](https://github.com/sgl-project/sglang/pull/3413) |
| 作者 | Fridge003 |
| 合并日期 | 2025-02-10 |
| labels | enhancement, lora |
| 热度 | 👍 2 · 💬 0 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
LoRA 输出错位(outcome misalignment)源于 `qkv_proj`/`gate_up_proj` 的 LoRA 支持不完整;修复 k_proj 无 LoRA 时的堆叠逻辑并补齐 gate_up/ColumnParallel 支持,同时重构 LoRA 代码。

## 问题现象
#2671 报告了应用 LoRA 时输出错位(output misalignment)的 bug。正文未附具体复现日志或数值,仅说明该 bug 与 LoRA 在 `qkv_proj` 与 `gate_up_proj` 模块上的支持不完整有关。

## 根因
正文披露:bug 的实际原因是 LoRA 对 `qkv_proj` 和 `gate_up_proj` 模块的支持不完整(incomplete lora support),由本 PR 修复。

(由 diff 推断)具体机制之一是:原 `initialize_weights` 从 `k_proj` 权重出发拼接 q/k/v,当某个 LoRA adapter 没有给 `k_proj` 挂 LoRA 时,该路径无法正确构造 stacked `qkv_proj` 权重;新代码 `stack_qkv_proj` 改为收集 q/k/v 目标模块集合,以 `q_proj` 为锚点,若 k_proj 无 LoRA 则用 `torch.zeros_like(weights[v_name])` 初始化为零,再拼出 qkv 权重。此外旧 `MergedColumnParallelLinearWithLoRA.apply_lora` 按 `base_output.shape[-1]` 对半切分输出,对 gate/up 输出维度不等的层会切错位置;新实现改走各后端的 `run_gate_up_lora`(由 diff 推断)。

## 修复方案
修复项(正文列出):
1. 修复 `k_proj` 无 LoRA 时 qkv LoRA 模块的处理:`lora.py` 中新增 `LoRAAdapter.stack_qkv_proj`,以 q_proj 为锚、k_proj 缺失时补零。
2. 增强 `gate_up_proj`(挂 `MergedColumnParallelLinear`、自定义 kernel)的 LoRA 支持:后端新增 `run_gate_up_lora`(triton 走新增的 `gate_up_lora_b_fwd`,flashinfer 分别对 gate/up 各做一次 `run_lora_b_sgemm`)。
3. 修复 `prepare_lora_batch` 中 buffer slot 的驱逐逻辑:`mem_pool.py` 的 `LoRAMemoryPool.prepare_lora_batch`。
4. 为 `ColumnParallelLinear` 层添加 LoRA 支持(新 `ColumnParallelLinearWithLoRA.apply_lora` 实际执行 A/B sgemm)。

重构项:定义 `LoRAMemoryPool` 类管理 buffer 内存(mem_pool.py);LoRA 层定义移入新的 `layers.py`;辅助函数放入 `utils.py`;重命名函数(`BaseLoraBackend`→`BaseLoRABackend`、`fuse_qkv_lora_b`→`fuse_stacked_lora_b`、`LoraBatchInfo`→`LoRABatchInfo`)、补类型注解与注释。

关键 diff 摘录(stack_qkv_proj 中 k_proj 缺失补零):

```diff
+                # If k_proj doesn't have lora, initialize it to zero
+                k_proj_weight = (
+                    weights[k_name]
+                    if "k_proj" in target_module
+                    else torch.zeros_like(weights[v_name])
+                )
```

注:diff 在 40000 字符处被截断(文件粒度),截断部分(mem_pool.py、utils.py 等)未展示,本卡仅描述可见部分与正文。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:影响所有 LoRA 使用路径:qkv/gate_up 权重堆叠逻辑(`lora.py`)、LoRA 后端接口与 triton/flashinfer 后端(`backend/`)、LoRA 层包装与调度(`layers.py` 新文件)、memory pool 的 batch 准备与 slot 驱逐(`mem_pool.py`)。大量类/函数重命名对下游引用是破坏性改动;k_proj 补零与按 `run_gate_up_lora` 的新切分逻辑会改变此前错位输出的数值结果。正文未提供 accuracy/benchmark 结果(检查项未勾选)。

## 关联
- #2671(报告输出错位 bug 的 issue)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
