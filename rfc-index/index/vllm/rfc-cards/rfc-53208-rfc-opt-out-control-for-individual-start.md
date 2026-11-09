# RFC #53208: [RFC]: Opt-out control for individual startup kernel warmups

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#53208](https://github.com/vllm-project/vllm/issues/53208) |
| 作者 | YongfengX |
| 创建日期 | 2026-08-21 |
| 状态 | ⚪ closed(关闭于 2026-08-24) |
| labels | RFC |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Attention / Kernel |

## 一句话结论
给启动期各内核 warmup 加一个通用 opt-out 轴(命名跳过列表 + Platform 钩子),让 AOT 平台跳过与己无关的 JIT 预热。

## 提案动机
`kernel_warmup()`(vllm/model_executor/warmup/kernel_warmup.py:98-215)在 CUDA graph capture 前从 `Worker.compile_or_warm_up_model` 运行一段固定预热序列,按模块 docstring,其存在是为 JIT 编译的内核服务。部分条目已可单独控制——`VLLM_DEEP_GEMM_WARMUP=skip`、`kernel_config.enable_cutedsl_warmup`、`kernel_config.enable_flashinfer_autotune`,且 `kernel_config.enable_jit_warmup` 是注册表的总闸——但整个集合不成体系,还有条目完全无开关(如 `qwen_triton_warmup`)。作者维护一个非 CUDA 加速器的 OOT 平台插件,把 GDN ops 重定向到 AOT 编译内核库:没有 JIT 可藏,预热纯属启动成本;更糟的是上游为参考 Triton 内核构造的 dummy 输入不满足 AOT 内核的 host 侧参数校验——`_warm_fused_sigmoid_gating_delta_rule_update_kernel` 构造 rank-2 的 `initial_state`,参考 Triton 路径只读 `initial_state.stride(0)` 并按裸指针转发所以从未暴露问题,而作者的内核要求 rank 4 并拒绝该输入,导致引擎启动失败。

## 提出的方案
一个通用 opt-out 轴,而非为每个 warmup 新增环境变量,给出两个可组合的 sketch:
- **A — 命名 warmups + 跳过列表**:给 `kernel_warmup()` 中每个条目一个稳定名称,新增如 `VLLM_SKIP_KERNEL_WARMUPS=qwen_triton,deep_gemm` 的环境变量;镜像已有的 `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`(vllm/envs.py:1716-1724,在 kernel_warmup.py:218-234 消费),后者已是逗号分隔的名称跳过列表。
- **B — Platform 钩子**:`current_platform.get_skip_kernel_warmups() -> set[str]`,在 `KernelConfig.set_platform_defaults` 已调用 `current_platform.get_default_ir_op_priority` 的位置消费;平台最清楚自己的内核是 JIT 还是 AOT,应由插件自行声明,而不是要求运维设环境变量。
- 组合方式:B 提供默认值,A 允许用户覆盖。默认行为不变——空跳过列表或未实现钩子时按现状运行全部预热。纯增量、向后兼容。
- 当前 workaround:插件把私有符号 `qwen_triton_warmup._warm_fused_sigmoid_gating_delta_rule_update_kernel` monkeypatch 成 no-op,依赖私有符号、任何上游重构都会破坏——这正是促成此 RFC 的原因。

## 优化点与收益
正文未给出量化数字。收益为启动可用性与工程健壮性:AOT/非 CUDA 平台消除纯启动开销,避免预热 dummy 输入不满足 AOT 参数校验导致的引擎启动失败,并去掉对私有符号的 monkeypatch。

## 现状与落地
已关闭(关闭于 2026-08-24,创建 3 天后)。正文的 Feedback Period、CC List、Any Other Things 均为 _No response_,无实现或结论信息,落地情况需查关联 PR。

## 关联
正文未列出 RFC/PR/文档链接(正文引用的均为 vLLM 代码路径,如 vllm/envs.py、vllm/config/kernel.py 等)。

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
