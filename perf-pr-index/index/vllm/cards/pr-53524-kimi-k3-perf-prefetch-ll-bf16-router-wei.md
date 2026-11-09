# PR #53524: [Kimi-K3][Perf] Prefetch ll_bf16 router weights for M=1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mingg26 |
| 合并 | 2026-09-01 |
| 分类 | Attention / Kernel (tier: major) |
| Label | ready, verified, kimi, k3 |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/53524 |

## 一句话结论

Kimi-K3 单 token(M=1)场景下,ll_bf16 router GEMM 在 PDL 等待前预取权重 tile,中位加速最高 +20.13%。

## 动机

Kimi-K3 的 MoE router 用 `ll_bf16` dot-product GEMM 计算,decode 单 token 时 M=1,该 kernel 与上游 AttnRes kernel 之间通过 PDL(programmatic dependent launch)衔接。原实现在 kernel 入口先执行 `griddepcontrol_wait()` 阻塞等待上游激活,再开始加载权重;而 router 权重是静态的、不依赖上游产出,这段等待时间被白白串行化,是 AttnRes-to-router 路径上的可压缩延迟。

## 优化点

机制改动:在 PDL wait 之前先把权重的前若干个静态 tile 预加载到寄存器(最多 8 个 `main_tiles`,`main_prefetch_tiles = min(main_tiles, 8)`),然后才调用 `griddepcontrol_wait()` 等待激活 A;主循环随后改走 `_vector_dotprod_prefetched` 变体,直接用已驻留的寄存器权重参与点积,把权重加载延迟藏在 PDL 等待窗口内。调度上按运行时 M shape 分派:`M == 1` 走 weight-prefetch 的 PDL 特化 kernel,其余 M 走原 kernel 不变;两种变体各自维护独立编译缓存(CompileKey 增加 `prefetch_pdl_weights` 维度)。M=4 上的强制预取实验显示有回退,因此生产分派刻意限定 M==1。

## 关键改动

```text
vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py: 新增
  prefetch_pdl_weights 配置、_vector_dotprod_prefetched 核函数,wait 前预取权重 tile
vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py: 新增
  ll_bf16_gemm_c1_pdl_kernel 实例,ll_bf16_gemm 按 M==1 分派
tests/kernels/test_ll_bf16_gemm.py: M=1 / M>1 分派边界单元测试
```

关键机制摘录(_ll_bf16_dotprod.py,wait 前预取):

```python
if const_expr(self.use_pdl and self.prefetch_pdl_weights):
    for tile in cutlass.range_constexpr(prefetch_tiles):
        cute.autovec_copy(tB[None, tile], prefetched_b[tile, None])
if const_expr(self.use_pdl):
    cute.arch.griddepcontrol_wait()
```

## 性能声明

测试硬件:NVIDIA B300 SXM6 AC,CUDA graph,测前 flush L2,SM 时钟锁定 1980 MHz。计时区域为 AttnRes + ll_bf16 合并区间(含 PDL handoff)。条件:M=1,N ∈ {256, 384, 896},AttnRes block 数 {1, 4, 8},共 9 组生产组合全部中性或更快:

- median speedup:+0.26% 至 +20.13%;
- mean speedup:+2.20% 至 +19.05%;
- M=1, N=896, blocks=8:16.288 us → 14.240 us(median,+14.38%)。

对照实验:M=4, N=896, blocks=8 强制启用预取,median 16.288 → 16.320 us(-0.20%),mean -1.30%,故生产分派仅在 M==1 启用。作者注明不声称先前更大 diff 的 TP8 serving 结果。

## 验证与回归风险

依赖 `tests/kernels/test_ll_bf16_gemm.py`(136 passed),覆盖 dot-product 与 split-K 形状、单 token 生产形状、数值边界、确定性输出、CUDA graph、分派与缓存行为;另通过 ruff check/format 与增量 `_C_stable_libtorch` 构建。风险点:仅 M=1 decode 路径受益;分派逻辑依赖 `hidden_states.shape[0]`,若上游 shape 语义变化会误选 kernel;预取只覆盖前 8 个 main tiles,K 很大时尾部 tile 仍走原路径。作者声明未做模型输出评估(改动仅为 kernel 调度,数值套件通过)。

## 关联

- PR #54261:提供本 PR 依赖的 AttnRes(此前本 PR 内的 AttnRes 改动已移除,由该 PR 承担)
- 同区域:同目录 cute_dsl 线性 kernel 家族(PDL 特化、split-K 变体)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
