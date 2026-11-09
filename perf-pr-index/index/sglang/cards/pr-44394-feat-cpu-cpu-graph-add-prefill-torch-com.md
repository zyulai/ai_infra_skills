# PR #44394: [Feat] [CPU] cpu-graph: Add prefill torch compile graph and dynamic graph support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @CaoE |
| 合并 | 2026-11-05 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | documentation, dependencies, sgl-kernel, intel, cpu, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 被截断(8 个文件未收入),收录部分含 cpu_graph_runner、AOT CPU kernels、rope/mamba custom op 等 |
| 链接 | https://github.com/sgl-project/sglang/pull/44394 |

## 一句话结论

把 CPU torch.compile graph 从 decode 扩展到 prefill 并支持动态 shape 回退图,Intel AMX 系统 Qwen3.5-35B(BF16, TP4)prefill 图在 BS=1/Input=1024 下 TTFT 提升约 6%。

## 动机

CPU 路径此前只有 decode 的 torch.compile graph,而 CPU 推理的 prefill 阶段 kernel launch/Python 开销占比更高,却没有图化覆盖;同时 CPU decode 图只支持静态 bucket shape,无法处理超出 bucket 的请求。内部实现迁移而来,目标是 Intel AMX 系统上 prefill/decode 都有 compile graph,并为 bucket 外请求提供动态 shape 回退。

## 优化点

扩展 `CPUGraphRunner` 支持 prefill graph 捕获与 replay,匹配顺序为"精确 bucket → padded bucket → 动态回退":静态 prefill graph 按 total-token bucket 捕获,启用 `SGLANG_TORCH_DYNAMIC_SHAPE=true` 时额外捕获 mark_dynamic 的动态图(dim 0 上限用 2^31-1 哨兵,pad 时配合 prefill padding 与 hidden-state trimming)。为让 Dynamo 顺利捕获做了一系列机制修复:新增 `_PrefillForwardDispatcher`,不同 prefill bucket 通过 `active_forward` 切换编译 callable 而不替换 `model.forward`,避免 serving 期间实例字典变化触发 guard 重编译;捕获/replay 期间用 `__slots__` 兼容类 patch 掉 TorchDynamo 无法实例化的 `msgspec.Struct` layer-boundary 对象(`Contribution`/`DeclaredSum`/`OwedOutput`/`UnreducedOutput`/`ExitDecision`),并临时解包带 `sys._getframe` 的 deprecated getter、禁用 duck-shape 与 custom-op aliasing 检查。custom op 侧把 rope/`chunk_gated_delta_rule_cpu` 等 schema 改为 in-place(`Tensor(a!)`、返回 `()`),Python wrapper 对 3D 输入先 clone,补齐 fake/meta 注册(含新增 qk_rmsnorm 系列、shm 集合通信 op);AMX attention metadata 把 `max_extend_len` 从 Python `.item()`(触发 recompile guard)改为传 -1 由 C++ 侧动态计算。CPU decode 配置优先级明确为 `decode.bs` > `decode.max_bs` > `--torch-compile-max-bs`,prefill bucket 需要 1 时从 `prefill.max_bs` 生成,并拒绝 `breakable`/`tc_piecewise` 等不支持的 CPU graph backend。prefill graph 为实验特性,默认关闭,须显式启用 `full` backend。

## 关键改动

```text
python/sglang/srt/model_executor/cpu_graph_runner.py: prefill graph 捕获/replay、动态图、Dynamo 兼容 patch、prefill forward dispatcher
python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp: rope/线性注意力 op schema 改 in-place,新增 fake 注册
python/sglang/srt/layers/attention/intel_amx_backend.py: max_extend_len 传 -1 交 C++ 动态计算,避免 .item() 触发 recompile
python/sglang/srt/arg_groups/memory_hook.py: decode bs/max_bs/torch_compile_max_bs 优先级解析
python/sglang/srt/environ.py: 新增 SGLANG_TORCH_DYNAMIC_SHAPE 与 SGLANG_CPU_GRAPH_COMPILE_BACKEND
```

```python
with (cpu_graph_patch_boundary_classes(), _disable_duck_shape(), ...):
    yield torch.compile(_make_compile_wrapper(model, model_forward),
                        backend=compile_backend, dynamic=dynamic)
```

## 性能声明

**Qwen3.5-35B(BF16, TP4)Prefill graph:约 6% TTFT Speedup(BS=1, Input=1024)**,测试平台为 Intel AMX CPU 系统(PR 标签 intel/cpu;正文未给出绝对 TTFT 数值与其他 batch/序列长度的结果)。decode graph 的增益未在本 PR 中重新测量。

## 验证与回归风险

PR 列出的测试覆盖:backend gating、bucket 生成、选项优先级、padding、dispatcher 稳定性、动态回退、encoder-decoder decode graph 复用;复用现有 real-model benchmark 与 MMLU 测试(显式启用 CPU prefill graph 与动态回退),并验证 graph 与 eager 精度一致、执行无 graph break/运行时重编译。回归风险:patch 层边界类与解包 getter 属于全局可变状态,仅限 CPU graph 上下文使用,若其他路径复用这些类需还原;rope/线性注意力 custom op schema 变更(返回值变化)是 ABI 级破坏,要求 AOT kernel 与 Python wrapper 同步升级;prefill graph 默认关闭属实验特性,`full` backend 才生效。CI 状态:Base 未运行(no_entry_sign),Extra 失败(:x:),AMD ROCm 10 通过。

## 关联

同区域:CPU decode torch.compile graph(本 PR 的前序实现,decode.bs 复用关系被本 PR 规范化)、CUDA graph/compile 系列;pack 未列出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
