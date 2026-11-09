# PR #18782: [VLM][LLM] Optimize fused_moe triton kernel tma

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-02-14 |
| 分类 | Attention / Kernel (tier: major) |
| Label | run-ci |
| Diff | pack 未提供 diffstat(diff 完整可见,仅改 fused_moe_triton_kernels.py 一个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/18782 |

## 一句话结论

削减 fused_moe Triton TMA 路径的 host 开销(allocator 置一次 + 权重 TensorDescriptor LRU 缓存),serving 基准 TTFT 提升约 5-10%。

## 动机

Profile Qwen3-VL-30B-A3B 发现 prefill 阶段 fused_moe 有明显 GPU bubble:一是 CUDA Graph 默认未开,大量小 kernel 反复 launch;二是 scheduler CPU 参与小 op 计算,延迟 kernel 启动。本 PR 聚焦 fused_moe 自身的 host 侧开销——TMA 路径每次调用都要执行 `triton.set_allocator` 并为持久权重 B 新建 TensorDescriptor,这些重复的 Python 开销在热路径上被放大。

## 优化点

两项进程级去重。其一,`triton.set_allocator` 只设一次:原先 `invoke_fused_moe_kernel()` 在 `a_use_tma or b_use_tma` 时每次调用都重新注册 allocator(为 TMA descriptor 做全局显存分配),现以模块级 `_TMA_ALLOCATOR_SET` guard 保证每进程一次;只影响 TMA 路径,非 TMA 执行不变。其二,权重 B 的 TensorDescriptor 用有界 LRU 缓存:键为 `(B.data_ptr, B.shape, B.stride, B.dtype, BLOCK_SIZE_N, BLOCK_SIZE_K)`(存储身份/布局 + tile 形状),命中即复用描述符;缓存上限由 `SGLANG_TMA_BDESC_CACHE_MAX` 控制(默认 64),设 0 关闭,防止无界增长。设计取舍:不缓存激活 A 的 descriptor(A 每次调用 shape/stride 可能变化,命中率低,且可能延长临时张量生命周期);缓存不加锁,GIL 下保证正确性,并发下最坏情况重复创建描述符但结果正确。

## 关键改动

```text
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py: _set_triton_tma_allocator 置一次 guard;_get_b_tma_desc_cached LRU 缓存
```

```python
def _get_b_tma_desc_cached(B: torch.Tensor, block_n: int, block_k: int):
    key = (
        int(B.data_ptr()),
        tuple(B.shape),
        tuple(B.stride()),
        str(B.dtype),
        int(block_n),
        int(block_k),
    )
    desc = _B_DESC_CACHE.get(key, None)
    if desc is not None:
        _B_DESC_CACHE.move_to_end(key)
        return desc
    desc = TensorDescriptor(B, B.shape, B.stride(), [1, block_n, block_k])
    _B_DESC_CACHE[key] = desc
    if len(_B_DESC_CACHE) > _B_DESC_CACHE_MAX:
        _B_DESC_CACHE.popitem(last=False)
    return desc
```

## 性能声明

测试条件:Qwen3-VL-30B-A3B-Instruct,`--tp-size 4 --attention-backend fa3 --mm-attention-backend fa3 --chunked-prefill-size 8192`,`bench_serving` image 数据集(560x560 jpeg x 1),256 prompts,input 128 / output 32,max-concurrency 32。pack 未注明 GPU 型号。

| 指标 | Baseline | PR |
|---|---|---|
| Request throughput (req/s) | 13.09 | 14.34 |
| Total token throughput (tok/s) | 5551.33 | 6082.15 |
| Mean TTFT (ms) | 951.86 | 858.76 |
| Median TTFT (ms) | 786.24 | 697.06 |
| Mean TPOT (ms) | 77.12 | 71.99 |
| Mean E2E Latency (ms) | 2419.69 | 2214.52 |

PR 正文结论:无质量回归(gsm8k 不变),TTFT 提升约 5-10%。另注明遗留问题:jit kernel 每层都会重新编译,后续继续优化 VL MoE 性能。

## 验证与回归风险

Accuracy Tests 一节为空,质量结论仅一句"gsm8k unchanged"(无数字);Checklist 未勾选,diff 无新增测试。回归面:仅 `invoke_fused_moe_kernel` 的 TMA 分支;缓存键含 data_ptr,若权重被释放后地址复用而 shape/dtype 相同,理论上可能命中过期键(实践中权重持久);`SGLANG_TMA_BDESC_CACHE_MAX` 环境变量控制上限;非 TMA 路径与其他 MoE 实现不受影响。

## 关联

后续(PR 正文预告):VL MoE 的 jit kernel 每层重复编译问题将继续优化。同区域:fused_moe Triton kernel 与 TMA descriptor 机制;pack 未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
