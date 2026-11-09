# PR #10709: [4/4] Introduce CachedKernel to reduce CSGMV kernel launch overheads by 60%

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @lifuhuang |
| 合并 | 2025-09-22 |
| 分类 | Attention / Kernel |
| Label | high priority, run-ci |
| Diff | diffstat 缺失(pack 未提供 +additions/-deletions 与文件数统计);diff 本身可用,涉及 3 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/10709 |

## 一句话结论

用 key 缓存已编译 Triton kernel,把 LoRA CSGMV 的 kernel launch 开销降约 60%,E2E latency 降 4%-9.4%。

## 动机

在前面几轮优化(本 PR 为系列 [4/4])之后,LoRA kernel 本身的计算性能已显著改善,triton kernel launch 时间成为 LoRA 路径的主要开销来源之一。Triton 内置的 cache lookup 机制偏通用因而较慢(作者引用 triton-lang/triton#2637 的相关讨论)。因此希望绕过 Triton 的通用派发路径,用更轻量的自定义缓存直接命中已编译 kernel。

## 优化点

引入 `CachedKernel` 包装类与 `cached_triton_kernel` 装饰器:装饰器接收一个 `key_fn`,从每次调用的 `(args, kwargs)` 中提取缓存键(本 PR 中为 `NUM_SLICES` 与 `BLOCK_M` 两个 constexpr)。首次调用仍走 `fn[grid](...)` 完成编译并按 key 存入 `self.kernel_cache`;命中后直接调用已编译 kernel 的 launcher,跳过 Triton 通用的 specialization 派发与查找。保持 `kernel[grid](...)` 的调用语法不变,启动器内部把 kwargs 按 signature 补齐成完整位置参数列表。机制上属于"用业务侧已知的最小键集合替换通用缓存查找",键集合越小、命中越快。约束:不支持带默认值的 kernel 参数(会触发 assert)。

## 关键改动

```text
python/sglang/utils.py: 新增 CachedKernel 类(kernel_cache + launcher)与 cached_triton_kernel 装饰器
python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py: expand kernel 挂上缓存装饰器
python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py: shrink kernel 挂上缓存装饰器
```

```python
@cached_triton_kernel(lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"]))
@triton.jit
def _chunked_lora_expand_kernel(...):
```

## 性能声明

- apply_lora duration:319 us → 123.4 us(约 -61%,对应标题的 60%;profiling 截图,PR 未注明模型/硬件/workload 条件)
- E2E latency:与最新 main(CSGMV v2)相比,bs 8-64 下 E2E latency 降低 4% - 9.4%(截图,PR 未注明测试条件)

## 验证与回归风险

PR 正文未注明新增测试。风险:CachedKernel 绕过了 Triton 内置的 specialization/缓存机制,若 `key_fn` 未覆盖所有影响编译结果的参数,可能错误复用不匹配的编译产物;不支持默认参数 kernel。影响路径集中在 LoRA 的 chunked SGMV expand/shrink 两个 triton op。

## 关联

同系列前序 [1/4]-[3/4] 为 LoRA kernel 优化(具体 PR 编号素材未注明);正文提到此前已合入 CSGMV v2。无其他已知关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
