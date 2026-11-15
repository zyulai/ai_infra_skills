# PR #46781: [Perf] spec: Avoid blocking H2D for MRoPE delta in spec decode

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hn1209 |
| 合并 | 2026-11-14 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/46781 |

## 一句话结论

把 spec decode 每步构建的 MRoPE delta 堆叠张量先经 pinned host memory 暂存再做 `non_blocking` H2D,消除 pageable copy 强制的 `cudaStreamSynchronize`:TPOT 在 A800 降 11%、B300 降 35%,MTP 部署每步 GPU 利用率 65.6%→81.2%。

## 动机

`ForwardBatch.compute_spec_mrope_positions` 为多模态请求构建 per-step MRoPE position 张量:在 host 上 stack 各请求的 `mrope_position_delta` 后 `.to(device=device)` 传到设备。stack 输出是 pageable 的,PyTorch 会在该 H2D copy 后附加 `cudaStreamSynchronize`——驱动无法让 DMA 异步进行(OS 可能在传输中途换出源页),host 会阻塞到 stream 上所有已入队 kernel 退休。该函数在 speculative decode 每步都跑(target verify 与 draft extend 都调用),sync 之后的排队工作就是整个 in-flight draft/verify 窗口:在 MTP (NEXTN) TP8 部署上,一个 8 字节 copy 变成每次调用多毫秒级停顿;profiling 显示每 decode step 两次停顿(约 1.4 ms 与 7.7 ms),每次都排空 stream、然后 host 一条条重发 kernel,让 GPU 饿着而不是在 primed queue 上跑在前面。

## 优化点

把 stacked delta 张量先放进 pinned host memory 再 H2D,使 `non_blocking=True` 真正生效(只在源为 pinned 时生效):正文给出的 helper 逻辑是 `torch.stack` 后若 pinned memory 可用,则 `torch.empty(..., pin_memory=True)` 暂存、`copy_` 后 `staging.to(device=device, non_blocking=True)`。CUDA 的 per-stream ordering 仍保证 copy 在任何读取 `mrope_delta_tensor` 的 kernel 之前完成——只移除 host 侧等待。

- `is_pin_memory_available(device)` 为 False 时(CPU-only/不支持的加速器)回退原阻塞 `.to(device=device)`,行为不变。
- 结果的 value、dtype、device、shape 均不变;额外 host 分配是每步一个 `[batch_size, 1]` int64 张量(几百字节),PyTorch caching host allocator 跨步复用 pinned block,不会无限增长。
- 不触碰 text-only fast path(batch 无多模态请求时在本张量构建前已返回);无新 env var / flag——pinned memory 可用时始终生效,与本文件中其他 `non_blocking=True` H2D(如 `_compute_mrope_positions_decode`)一致。

## 关键改动

```text
python/sglang/srt/model_executor/forward_batch_info.py(pack diff 仅展示此文件): compute_spec_mrope_positions 改调 async_h2d_stack;新增 async_h2d_stack 导入
```

```python
-            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
+            mrope_delta_tensor = async_h2d_stack(mrope_deltas, device=device)
```

## 性能声明

- 内部验证(MTP/NEXTN,TP8,Qwen3.5-VL-class 模型,`livingroom_planner` 多模态 workload,稳态 decode):每步 GPU 利用率(draft/verify/draft-extend 所有 CUDA stream 的 busy 时间求和)65.6%→81.2%(+15.6 pp);`scheduler.run_batch` wall time 中位数 20375→17630 µs(−13.5%);一个 decode step 内阻塞 `cudaStreamSynchronize` 时间 9119→5517 µs(−3602 µs);两个目标停顿(`[[1,1],[1,1]]` 形状的 pageable H2D,target-verify 与 draft-extend 内,约 1.4 ms 与 7.7 ms)在 trace 中消失。
- 独立微基准(在 stream 上先入队约 240 个小 matmul 再调用该函数):host 侧调用耗时原路径约 250 ms → 本 PR 约 0.2 ms。
- TPOT:A800 上提升 11%(4.51→4.07 ms),B300 上提升 35%(2.38→1.76 ms)。
- 第三方端到端(@esarafian,Qwen3.8-Flash-Next / NEXTN MTP):GB300 确定性 12 图像 + 8 纯文本 greedy 请求输出与 main 逐字节一致;RTX PRO 6000 (TP2) median tok/s +6.2%(962.5 vs 924.5),accept length 同为 2.48;GB300 (TP1) A-B-B-A 消融隔离 `forward_batch_info.py`:1003 vs 884 tok/s(+13.5%),median TPOT 14.95 vs 17.32 ms(−14%),各 run accept length 2.42–2.44。

## 验证与回归风险

`test/registered/kernels/ops/speculative/test_next_verify_prep.py` 的 `test_spec_mrope_positions_graph` 已经以 `mixed=True`(正是本 PR 触碰的分支)端到端调用 `compute_spec_mrope_positions` 并在 eager 与 CUDA-graph replay 下对照 reference 断言;该测试未修改即通过(输出值不受影响的预期结果)。回归风险低:仅替换 host→device 传输方式,fallback 路径覆盖无 pinned memory 的环境;pinned 分配由 caching allocator 复用。

## 关联

无直接 PR 引用(作者注明本工作于 Spirit AI 任职期间作为 SGLang speculative decoding 优化的一部分完成)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
