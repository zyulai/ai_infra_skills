# RFC #64262: [RFC] Signature-driven JIT warmup v2 for Triton kernels

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64262](https://github.com/vllm-project/vllm/issues/64262) |
| 作者 | LopezCastroRoberto |
| 创建日期 | 2026-10-28 |
| 状态 | 🟢 open |
| labels | RFC |
| 热度 | 👍 0 · 💬 0 |
| 分类 | CUDA Graph / Compiler |

## 一句话结论
用内核签名声明的 Triton JIT warmup v2:去掉 dispatcher 样板、grid 回到调用点,让模型完全 de-JIT 化。

## 提案动机
现有 warmup v1 契约要求每个 Triton 内核配套 warmup-inputs 函数、dispatcher、注册与改写过的调用点:参数列表被写三遍,grid 与 constexpr 离开调用点,warmup inputs 还须手工与 dispatcher 同步 —— 实际发生过漂移(#56654 把 `idx_mapping`/`sample_idx_mapping` 声明为 int32,运行时却是 int64)。结果是模型仍未完全 de-JIT 化,`--jit-monitor-mode error` 会失败。

## 提出的方案
- 签名说清要 warm 什么:指针用 `Ptr[...]` 注解给 dtype,`constexpr` 用默认值,调用方在 `register_warmup` 传部署值;签名表达不了的用 `@triton_warmup` 装饰器,host launcher 用 `@triton_warmup_launcher`。
- 调用点保持普通 `kernel[grid](...)`,后端不变;`@triton_warmup` 返回同一个 `JITFunction`,不包裹内核、无 dummy run。
- 注册经新的 `JitWarmupRegistry.register_or_defer`:活跃 registry 内直接注册,registry 外进进程级队列、`warmup()` 时补收(修复 v1 在 registry 外静默丢弃、逼出 `@capture`/`activate()` 的问题)。
- 启动时把每个注册展开为候选 launch 参数(每个 Triton 整数类一个代表),经 Triton binder 与 `_triton_key_deriver` 派生 cache key、去重后并行 `kernel.warmup(...)` 编译。
- 范围仅 Triton(CuTeDSL/TileLang 暂留旧契约);opt-in per kernel,不跑在 launch 路径;showcase 迁移 #56323 的 17 个内核(kernel 侧 +367/−876 行),PR 只含内核侧改动以聚焦 API 评审。

## 优化点与收益
- 代码量:`apply_top_k_top_p_triton` 一个 24 行 inputs 函数替代 #56323 中 4 个 warmup-inputs 函数 + 4 个 dispatcher(210 行),`topk_topp_triton.py` 与 `topk_topp_sampler.py` 净减 168 行(+94/−262)。
- 不"一刀切 do_not_specialize"的依据(B300、经真实 launcher 实测的减速):`sample_recovered_tokens_kernel` 慢 `1.6–2.6×`,top-k 慢 `1.4–1.7×`,top-k + top-p 慢 `1.2–1.3×`,MTP `_shift_input_embeds_kernel` 慢最高 `2.4×`,`scatter_states` 无差(带宽受限拷贝)。
- 风险:依赖三个 Triton internals(`__triton_builtin__` 标记、忽略未知注解名、给 `JITFunction` 加 `register_warmup`);属性按名绑定;launcher replay 临时换模块 globals 且不可并发;autotuned/heuristics 内核暂不支持。

## 现状与落地
仍在讨论/等待评审:open 状态、0 评论;PR 只含 17 个内核的迁移,支撑代码与测试在 API 达成一致后再提交;旧 dispatcher 继续可用,内核可逐个迁移。

## 关联
- #47456 与 #49349(JIT warmup v1 基础,正文未附链接)
- #56323(17 内核迁移来源)与 #56654(其部分回滚、契约漂移案例)
- [PR #62015(top-k/top-p launcher 形式覆盖)](https://github.com/vllm-project/vllm/pull/62015)

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
