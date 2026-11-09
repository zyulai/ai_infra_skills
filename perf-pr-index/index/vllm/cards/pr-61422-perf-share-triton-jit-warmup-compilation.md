# PR #61422: [Perf] Share Triton JIT warmup compilation pool across registered kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ayush1399 |
| 合并 | 2026-10-16 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 3 个文件(warmup 实现、triton helper 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/61422 |

## 一句话结论

让多个 Triton warmup kernel 共享编译池,H100 冷启动 JIT 预热快 9–12%(16 线程 23–28%)。

## 动机

PR 正文 Overview:这是 #58582 的后续。此前启动预热时,每个符合条件的 Triton kernel 依次「排空」自己的 `AsyncCompileMode` 池,单 key 的 kernel 之间只能串行编译,编译无法跨 kernel 重叠。当注册了多个 Triton warmup kernel(MRV1 sampler/spec-decode kernels、DSv4、GLM-5 Next)时,冷启动 JIT 预热时间被串行编译拉长。

## 优化点

`JitWarmupRegistry.warmup()` 改为把所有 `supports_parallel_compile` 的 kernel 放进同一个 `compile_triton_warmup()` 池里一起编译,其余 kernel 仍串行执行。引入一个 ContextVar 阻止 `compile_many` 嵌套创建池,而直接调用 `compile_many` 时仍自建池;dispatch 仍留在调用线程(与 #58582 一致)。正文列出的限制:tqdm 进度条按 dispatch 推进,会先跳到前面再等池排空;编译错误在共享池退出时才暴露,而不是在出错 kernel 的 `compile_many` 内。作者明确声明对 ROCm、autotuned kernels、非 Triton owners、以及只有单个注册 kernel 的配置无变化。

## 关键改动

```text
vllm/model_executor/warmup/jit_warmup.py: JitWarmupRegistry.warmup() 把并行编译 kernel 汇总到单一编译池,其余串行
vllm/model_executor/warmup/jit_warmup_triton_helper.py: 引入 ContextVar 防止 compile_many 嵌套建池
tests/model_executor/test_jit_warmup_triton_launcher.py: 更新测试,断言两个单 key kernel 必须并发编译(在 base commit 上会失败)
```

关键机制代码(并行项统一进共享池,串行项照旧):

```python
        parallel_items: list[tuple[VllmJitKernel[Any], dict[Any, None]]] = []
        serial_items = []
        for kernel, compile_keys in kernel_items:
            if (isinstance(kernel, VllmTritonJitKernel)
                    and kernel.supports_parallel_compile):
                parallel_items.append((kernel, compile_keys))
            else:
                serial_items.append((kernel, compile_keys))
...
            if parallel_items:
                compile_triton_warmup(
                    lambda: compile_kernels(parallel_items),
                    sum(len(keys) for _, keys in parallel_items),
                )
            compile_kernels(serial_items)
```

## 性能声明

PR 正文 Claims 与 Validation:默认线程下冷 JIT warmup 快 9–12%,16 线程下 23–28%,条件是注册了多个 Triton warmup kernel。硬件 H100、`VLLM_USE_V2_MODEL_RUNNER=0`、每组 3 次取均值,冷缓存下 `JIT kernel warmup finished in …s`:Gemma 4 31B + EAGLE3、TP=8(12 kernels),4 线程 20.72s → 18.86s(−9%),16 线程 15.25s → 11.69s(−23%);Qwen3-0.6B-FP8 + ngram(9 kernels),4 线程 17.99s → 15.89s(−12%),16 线程 11.55s → 8.37s(−28%)。正文另注明 Gemma 4 在默认 MRV2 只注册 FA4 一个 kernel,无可重叠:5.74s → 5.73s。无模型 eval,因为只改编译调度、不改编译出的 kernel。

## 验证与回归风险

`pytest tests/model_executor/test_jit_warmup_triton_launcher.py`:15 passed,且更新后的并发断言在 base commit 上失败。影响路径:cold JIT warmup 阶段的编译调度;正文声明 ROCm、autotuned kernel、非 Triton owner、单 kernel 配置不受影响,编译错误暴露时机与 tqdm 进度条行为发生变化,但编译产物不变。

## 关联

前序 PR #58582(正文称本 PR 为其 follow-up);正文另提及最接近的相关工作 #56154(compile-key derivation)与 #56761(TileLang),并声明无其他开放 PR 共享编译池。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
