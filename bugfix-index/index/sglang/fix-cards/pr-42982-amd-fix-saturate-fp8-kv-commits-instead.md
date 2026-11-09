# PR #42982: [AMD][Fix] Saturate FP8 KV commits instead of reproducing e4m3fnuz NaN overflow

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#42982](https://github.com/sgl-project/sglang/pull/42982) |
| 作者 | michaelzhang-ai |
| 合并日期 | 2026-10-10 |
| labels | run-ci, jit-kernel, run-ci-extra, memory-pool |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
AMD 上 FP8 KV 提交对 e4m3fnuz 溢出写出 NaN,与 eager 路径字节不一致并污染 KV slot;统一在 FP8 cast 前做饱和截断。

## 问题现象
`jit-kernel-unit-test-amd` 自 #40785 合并起在 `main` 上持续红,其后每次运行该 job 均以同一方式失败:同一索引处 `30732 / 65536` 个元素不匹配。该 job 只在 PR 触及 `python/sglang/kernels/` 时运行,多数 PR 跳过它,故一直未被察觉(证据见 #42981)。这不只是测试问题:ROCm 上 `set_kv_buffer_prefix_valid` 的 fused 路径与 eager 回退对同一输入写出不同字节——fallback 把 NaN 写进已提交的 KV slot,该 slot 之后每次被读取都会污染 attention。

## 根因
正文披露:torch 的 `float8_e4m3fn` cast 溢出时饱和到最大有限值;`float8_e4m3fnuz` 没有无穷,其 eager cast `fp8e4m3fnuz_from_fp32_value` 对每个 ≥256.0 的量级返回 NaN(`0x80`)。而 Triton 的 cast 对两种类型都饱和,因此 #40785 的 fused commit kernel 在 CUDA 上与 eager 一致、在 gfx94x 上发散。

## 修复方案
把「饱和」定义为契约,在所有三处 FP8 cast 之前统一截断:
- `cache_move.py`:Triton kernel 新增 `_saturate_to_fp8_range(val, FP8_MAX: tl.constexpr, DTYPE)`,NaN 因比较恒为 false 而落过两个 `tl.where` 分支、仍转为 FP8 NaN;截断在源 dtype 内做以保持原有 lowering;
- `memory_pool.py`:eager 回退同样截断,使两条生产路径字节一致;
- `test/kernels/prefix_valid.py`:共享 oracle 同样截断。

```python
limit = torch.finfo(dtype).max
return value.clamp(-limit, limit)
```

上限用 `torch.finfo(fp8).max` 而非保守常量 `fp8_max`(fnuz 上为 224.0):在 224.0 截断会把 eager 会舍入到 240.0 的值下拉一档损失精度。早期修订曾在 FP32 截断,使 bf16/fp16→fp8 变成 fp32→fp8,在 fp16+host scalar scale 下有 2/65536 个元素偏一档 FP8 subnormal,故改回源 dtype 截断。CUDA 上 eager cast 本就饱和到 448,三处改动均为 no-op。

## 影响与回归风险
- 影响窗口:自 #40785 合并起的 `main`(该 AMD job 自此持续红)。
- 回归面:FP8 KV commit 的 fused Triton kernel 与 eager fallback、共享测试 oracle;非 FP8 提交路径走 `set_kv_buffer_prefix_valid_tiled`,未被触及;NPU job 在非 CUDA/HIP 提前返回,不受影响。CI 上仍有与本路径无关的红项(MiniMax decode kernel 128 KiB LDS 超 64 KiB 限制、mi35x、`test_e2e_spec_eagle.py` 服务端停止响应)。

## 关联
#40785, #42981

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
