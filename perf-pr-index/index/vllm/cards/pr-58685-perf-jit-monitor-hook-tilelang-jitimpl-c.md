# PR #58685: [Perf] JIT monitor: hook TileLang JITImpl.compile instead of __call__

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-10-08 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/58685 |

## 一句话结论

JIT monitor 从包装 TileLang `__call__` 改为 hook `compile`,缓存命中不再过 wrapper,GLM-5.3-Flash 每 eager step 省 0.7ms CPU,TTFT 改善 2-3%。

## 动机

JIT monitor 在 warmup 后开启,此前包装的是 TileLang 的 `JITImpl.__call__`。但 `__call__` 在几乎全部调用中都命中 kernel cache,只有 cache miss 时才调用 `self.compile()`——即 monitor 的监控对象本应是低频的 compile 事件,却被挂在每次 kernel launch 的热路径上,每次调用都要为构建 cache-miss key 付费。对于 CUDA graph 之外有大量 TileLang launch 的模型影响显著:GLM-5.3-Flash 每个 eager step(prefill/混合 batch)约有 72 次 mhc TileLang 调用。

## 优化点

把 hook 点从 `JITImpl.__call__` 移到 `JITImpl.compile`:monitor 的包装函数 `_compile_with_monitor` 只在真正 compile(即 cache miss)时执行,构建 verbose detail 与 cache-miss key 的开销也随之移入 cold path;cache 命中的热路径完全不再经过 monitor。功能上 monitor 仍能捕获 warmup 之后的每一次 JIT compile,递归深度计数与日志逻辑保持不变。测试侧的 fake `JITImpl` 同步改为 cache miss 时调用 `compile()`,与真实实现行为一致。

## 关键改动

```text
vllm/utils/jit_monitor.py: _init_with_monitor 中改为包装 jit_impl_cls.compile,删除热路径上的 cache_key 构建
tests/jit_monitor/test_hooks.py: FakeJITKernel 的 __call__ 改为 cache miss 时调用 self.compile()
```

关键代码摘录(节选,<=15 行):

```python
# JITImpl.__call__ only calls compile() on a kernel-cache miss, so
# hooking compile() keeps the per-call hot path untouched.
original_compile = jit_impl_cls.compile

@functools.wraps(original_compile)
def _compile_with_monitor(self, *args, **kwargs):
    global _tilelang_jitimpl_compile_depth
    _tilelang_jitimpl_compile_depth += 1
    try:
        ...
        return original_compile(self, *args, **kwargs)
    finally:
        _tilelang_jitimpl_compile_depth -= 1

jit_impl_cls.compile = _compile_with_monitor
```

## 性能声明

- 微基准(H100,2000 次 `mhc_post_tilelang` 调用,µs/call):main,monitor off 15.7 / on 25.9;本 PR,off 15.5 / on 15.6。即 monitor 开启时的 wrapper 开销从约 10.2µs/call 降到约 0.1µs/call。
- 折算:约 10µs x 72 次 ≈ 每个 eager step 减少 0.7ms CPU(GLM-5.3-Flash)。
- E2E(zai-org/GLM-5.3-Flash,TP8,H100,fp8 KV,MTP 5 speculative tokens,async scheduling,vllm bench serve random 1024 in / 128 out,每点 2 次平均):TTFT 改善 2-3%(如 c=32 mean TTFT 338.2 → 331.3 ms,c=8 226.2 → 219.2 ms);TPOT 在噪声内(decode 走 CUDA graphs)。

## 验证与回归风险

- 测试:pytest tests/jit_monitor/(25 passed, 7 skipped);E2E 以 GLM-5.3-Flash serving benchmark 验证。
- 回归风险面:仅影响开启 JIT monitor 且使用 TileLang 的路径。monitor 捕获语义不变(仍捕获 warmup 后所有 compile);若存在不经过 `compile()` 的 JIT 入口(如直接调用内部编译方法)则不再被监控,但按 PR 描述 `__call__` 的 miss 路径就是 `compile()`。对非 TileLang 的 JIT(mamba/FlyDSL 等)hook 无影响。

## 关联

- 同区域:索引库已有 pr-61903(CuteDSL bf16x3 router kernel)、pr-42235(flashinfer cutedsl)等 CuteDSL/TileLang 相关卡片;JIT monitor 专项无其他卡片。
- PR 正文声明已检索 open PR,无重复实现。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
