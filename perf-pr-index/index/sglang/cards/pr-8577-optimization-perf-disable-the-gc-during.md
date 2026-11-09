# PR #8577: [Optimization][Perf] Disable the GC during CUDA graph capture to speed up by up to 3x

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @b8zhong |
| 合并 | 2025-07-31 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/8577 |

## 一句话结论

在 CUDA graph capture 期间冻结 Python GC,消除 gc.collect() 扫描开销,capture 提速 2.3x–3.7x。

## 动机

服务启动时 CUDA graph capture 阶段会分配大量对象,Python GC 周期性运行时反复扫描长寿对象,capture 越大浪费越多,直接拖慢启动时间。思路来自 vllm-project/vllm#21146(致谢 @mgoin):在 capture 期间禁用 GC 并冻结现有对象,可消除这部分开销。

## 优化点

新增 `freeze_gc` context manager 包裹 capture 流程:

1. capture 前先 `gc.collect()` 主动清理一轮垃圾;
2. 默认(未开 `--enable-cudagraph-gc`)调用 `gc.freeze()`,把全部现存对象标记为永久代,使后续 GC 完全不再扫描这些长寿对象;
3. capture 结束后 `gc.unfreeze()` 恢复,保证运行期 GC 行为不变。

新增 server arg `--enable-cudagraph-gc`(默认 False,即默认冻结 GC),需要 capture 期间正常 GC 的场景可显式打开。改动只作用于 capture 窗口,不影响推理稳态。

## 关键改动

```text
python/sglang/srt/model_executor/cuda_graph_runner.py: 新增 freeze_gc()
  context manager,在 capture() 中包裹 graph_capture();先 collect 再
  freeze,finally 中 unfreeze
python/sglang/srt/server_args.py: 新增 enable_cudagraph_gc 开关(默认
  False)与 --enable-cudagraph-gc CLI 参数
```

关键片段(diff 摘录):

```python
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
```

## 性能声明

PR 正文:CUDA graph capture 期间禁用 Python GC, capture 时间提速 **2.3x–3.7x**。测试为全新容器环境、默认 capture batch sizes 无覆盖:

- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`:23/23 batches 耗时 25s(1.09s/it)→ 10s(2.25it/s)。
- `Qwen/Qwen3-0.6B`(小模型):23/23 batches 耗时 6s(3.55it/s)→ 1s(13.31it/s)。

精度:`lm-eval` 结果相同。作者另注明服务关闭时会重新启用 CUDA graph capture 相关清理(其认为影响不大)。

## 验证与回归风险

- 精度验证依赖作者提供的 `lm-eval` 一致性结论,PR 未新增单元测试(diff 中无测试改动)。
- 回归面:capture 期间 GC 冻结意味着该窗口内产生的循环引用垃圾要等到 unfreeze 后才能回收,极端情况下可能短暂抬高内存占用;`--enable-cudagraph-gc` 提供了回退开关。
- freeze/unfreeze 若与并行 capture 或异步异常路径交叉,finally 恢复逻辑是正确性关键点。

## 关联

vllm-project/vllm#21146(同思路的上游来源,PR 正文引用)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
