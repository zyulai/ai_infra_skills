# PR #56323: [6/N][warmup][DSv4] Migrate sampling, and DFlash JIT kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-09-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | speculative-decoding, ready, ci/build, deepseek, cpu, quantization, mrv2, DSv4, dflash |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(20 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/56323 |

## 一句话结论

把 sampling 与 DFlash 的 JIT kernel 纳入 vLLM warmup 框架,在启动期预编译、消除服务中首次调用时的 JIT 编译卡顿。

## 动机

Triton/JIT kernel 首次以新的 compile key 被调用时才编译,落在请求路径上会造成延迟尖刺。DSv4(DeepSeek-V4)系列 warmup 工作(本 PR 为 6/N,灵感来自 #56154)的目标是把 speculative decoding 相关的 sampling、rejection 与 DFlash kernel 迁移到统一的 JIT warmup 框架,在引擎启动阶段枚举所有编译变体并预热。

## 优化点

本 PR 主要是 warmup 基础设施的能力扩展而非 kernel 本身改动:(1) `VllmJitKernel._expand_warmup_cases` 支持在 AST-traced warmup 方法里声明符号化域——`WarmupIntRange` 新增 lambda `advance`(如 `advance=lambda v: v*2` 生成 1,2,4,8),新增 `WarmupChoices` 显式有限取值集,以及 `_when()` 谓词过滤符号 case; internally 编译并 exec 生成的 case 函数,对域做笛卡尔积展开。(2) `jit_warmup_triton_helper` 新增 `triton_kernel_dispatcher_with_warmup` 装饰器:被装饰的 dispatch 函数返回 `LaunchSpec`(grid + kernel kwargs),框架据此推导 Triton compile key(处理 autotune configs、heuristics wrapper、runtime knobs、JIT identity)并在 `get_warmup_keys()` 时穷举大 range(如 1..8193)后按 key 去重;`TritonWarmupTensor` 用 fake CUDA tensor 让 traced 代码可以直接做 `shape[0]`、`is_contiguous` 等操作。(3) `JitWarmupRegistry.capture` 装饰器在模型 `__init__` 期间自动收集 `register_warmup` 调用。(4) mamba 的 `scatter_states` 作为样板迁移到该 dispatcher。dispatch 未被子类覆写时跳过 trace 并报错,避免静默退化。

## 关键改动

```text
vllm/model_executor/warmup/jit_warmup.py: WarmupChoices/_when/_expand_warmup_cases/JitWarmupRegistry.capture
vllm/model_executor/warmup/jit_warmup_triton_helper.py: triton_kernel_dispatcher_with_warmup、TritonWarmupTensor、triton_warmup_inputs
vllm/model_executor/layers/mamba/ops/scatter_states.py: scatter_states 迁移为 dispatcher 形式(样板)
tests/model_executor/test_spec_decode_rejection_warmup.py: 删除(迁移到新框架)
```

## 性能声明

PR 正文无 benchmark 数字(正文仅一句 "Inspired on https://github.com/vllm-project/vllm/pull/56154",性能目标即启动期预热以消除运行时 JIT 编译)。间接信号:`.buildkite/test_areas/models_basic.yaml` 中 Basic Models (Other) 的 CI 超时从 35 分钟调到 38 分钟,反映 warmup 扩展增加了启动期工作量。PR 未注明测试条件下的加速数据。

## 验证与回归风险

`tests/model_executor/test_jit_warmup.py` 新增 lambda advancement、registry capture 用例;`test_jit_warmup_triton_launcher.py` 新增约 290 行测试,覆盖装饰器返回 launcher、大 range 先穷举后去重、dispatch 断言传播、fake CUDA tensor、native launcher 装饰、pointer dtype 显式声明、autotune/heuristics 的 key 推导。风险:warmup 框架是共享基础设施,`get_warmup_keys` 的调用约定从位置参数 `args=(self.vllm_config,)` 改为 keyword `vllm_config=`,依赖旧约定的自定义 kernel 需要适配;`VllmJitKernel.dispatch` 未覆写时现在抛 `TypeError`(原先抽象方法在实例化时报错);`scatter_states` 迁移改变了 Triton launch 路径,影响 mamba/GDN 状态散写。

## 关联

同系列:warmup/DSv4 迁移的第 6 个 PR,前作 #56154;删除的 `spec_decode_rejection_warmup` 测试对应 spec decode rejection warmup 路径的重构。同分类可参考 spec decode 相关卡(如 pr-54163、pr-51725)。
