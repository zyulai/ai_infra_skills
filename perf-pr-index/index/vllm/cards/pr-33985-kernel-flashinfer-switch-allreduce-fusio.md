# PR #33985: [Kernel] FlashInfer: switch allreduce fusion to unified API

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mmangkad |
| 合并 | 2026-02-09 |
| 分类 | Attention / Kernel(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/33985 |

## 一句话结论

FlashInfer allreduce+norm fusion 迁移到统一 allreduce_fusion API 与 workspace 生命周期(需 flashinfer >= 0.6.3)。

## 动机

FlashInfer 已把 trtllm_allreduce_fusion 及其 IPC workspace 系列函数收拢为统一的 flashinfer.comm.allreduce_fusion / create_allreduce_fusion_workspace;vLLM 继续调用旧 API 会被版本演进卡住,且旧接口要求自行管理 world_rank/trigger_completion_at_end/use_fp32_lamport 等参数与 workspace 指针。

## 优化点

allreduce_rms_fusion.py 编译 pass 改调 allreduce_fusion(input, workspace, pattern, ...):rank/world_size/hidden_dim/workspace_ptrs/trigger_completion_at_end 等参数收进 workspace 对象;workspace 用 create_allreduce_fusion_workspace(backend="trtllm", dtype=model_dtype) 创建,element_size 按真实模型 dtype 计算(替代 use_fp32_lamport 的 4 字节假设),析构走 workspace.destroy() 并吞异常。one-shot 判定、launch_with_pdl、fp32_acc、SWIZZLED_128x4 layout 等 fusion 语义保持不变。benchmark_fused_collective.py 与 fusion 测试同步迁移,可用性探测改为同时检查 allreduce_fusion 与 create_allreduce_fusion_workspace 两个符号。

## 关键改动

```text
vllm/compilation/passes/fusion/allreduce_rms_fusion.py: pass 迁移到统一 API + workspace 对象
benchmarks/kernels/benchmark_fused_collective.py: 同步迁移(workspace/dtype 选择)
tests/compile/passes/distributed/test_fusion_all_reduce.py: 符号探测更新
```

```python
workspace = flashinfer_comm.create_allreduce_fusion_workspace(
    backend="trtllm", world_size=self.tp_size, rank=rank,
    max_token_num=self.max_token_num, hidden_dim=self.hidden_dim,
    dtype=self.model_dtype)
...
flashinfer_comm.allreduce_fusion(input=allreduce_in, workspace=_FI_WORKSPACE,
                                 pattern=pattern_code, ...)
```

## 性能声明

PR 未给出性能数字,定位为 API 迁移。验证:tests/compile/distributed/test_fusion_all_reduce.py 8 passed(427.25s,覆盖 RMSNorm / FP8 静态量化 / FP4 静态量化模型与 oneshot 开关组合)。依赖前置:flashinfer-python >= 0.6.3。

## 验证与回归风险

回归风险:硬性提升 flashinfer 版本下限(旧版无统一 API 直接禁用 fusion);workspace dtype 改为按模型 dtype 推导,fp32 模型的 use_fp32_lamport 特判被移除,fp32 训练/推理路径的 workspace 布局与旧行为的差异未在 PR 中单独验证;其余融合语义参数不变。

## 关联

同区域卡:#21325(修复同一 fusion 的启用判断,本批次前序)、#20691(sm100 FlashInfer fused allreduce 集成)、#21069(allreduce+rmsnorm+quant fusion)、#24248(torch native allreduce fusion 对照路线)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
