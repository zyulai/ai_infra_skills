# PR #56683: [Perf] Parallelize mHC pre-norm JIT warmup

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @WoosukKwon |
| 合并 | 2026-09-14 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | ready |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/56683 |

## 一句话结论

把 mHC pre-norm 的 TileLang JIT warmup 改为多 worker 并行编译,9 个变体冷编译 95s 缩到 43s(2.19x)。

## 动机

mHC(multi-head compression 相关 kernel,用于 DSV4.1 等模型的 pre-norm/pre-mix 路径)在启动时需要 JIT 编译多个 TileLang 变体(不同 input/pre-mix 配置与 n_splits)。原 `JitWarmupKernel.warmup()` 逐个 key 串行 `compile()`,冷缓存下 9 个变体要 95 秒,直接拖慢引擎启动;而这些变体的编译互相独立,天然可并行。

## 优化点

机制是给 JIT warmup 框架加批量编译钩子:基类 `VllmJitKernel` 层面新增 `compile_many()`,默认实现仍是顺序循环(保持其他 kernel 行为不变),`warmup()` 与 warmup registry 改为统一调用 `compile_many`。mHC 的 warmup 类覆写它:先用 `dict.fromkeys` 去重并剔除已缓存 key,缺失数 ≥2 时调用 TileLang 的 `par_compile()`,每个进程最多 4 个 compiler worker 并行编译独立 TIR 模块;注意 TileLang 的 module elaboration 仍是串行的,并行的只是编译阶段。已缓存变体与运行时 dispatch key 均保持不变,只改编译调度,不改 kernel 数学。与 #56682 的 dummy-initialization 改动相互独立。

## 关键改动

```text
vllm/model_executor/warmup/jit_warmup.py: 基类新增 compile_many() 钩子,warmup() 改走批量路径
vllm/model_executor/kernels/mhc/warmup.py: 覆写 compile_many(),缺失 key 走 TileLang par_compile(≤4 workers)
```

```python
missing = list(dict.fromkeys(
    key for key in compile_keys if key not in self._compiled_cache))
if len(missing) < 2:
    return super().compile_many(missing)
compiled = self.kernel.par_compile(
    [asdict(key) for key in missing], num_workers=min(4, len(missing)))
```

## 性能声明

GB200 上 9 个冷缓存 DSV4.1 pre-norm 变体(hidden size 5120、hc_mult 4,三种 input/pre-mix 配置,n_splits 1/4/16):串行 95.03s vs 4 个 compiler worker 43.39s,**隔离编译阶段 2.19x 提速**。PR 明确说明这是单次探索性测量(排除 import/key 枚举,空 TileLang 缓存,host 侧引擎初始化并发),未声明全引擎启动加速;另一次 TP4 真实权重启动中,5 个变体已由 profiling 编译,剩余 4 个并行编译约 14s。未跑模型质量 benchmark(不改 kernel 数学)。

## 验证与回归风险

`tests/model_executor/test_jit_warmup.py` 与 `tests/kernels/test_mhc_jit_warmup.py` 共 60 个既有测试通过(18.86s);GB200 上 batch warmup 后 3 个 GPU reference/CUDA graph replay 用例(broadcast、identity、carried pre-mix,33 tokens、fused normalization、DeepGEMM、20 轮 Sinkhorn)通过;两次运行的生成 CUDA 源码集合一致。无新增测试。风险面窄:默认路径仍串行,仅 mHC 走并行编译;`<2` 个缺失 key 时回退基类,worker 数按缺失数截断,行为可预期。

## 关联

同区域:#51802、#49707(mHC warmup 覆盖率相关 open PR)、#56682(mHC dummy-initialization,与本 PR 独立)。
