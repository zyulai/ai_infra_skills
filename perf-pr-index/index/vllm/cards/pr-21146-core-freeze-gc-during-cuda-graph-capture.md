# PR #21146: [Core] Freeze gc during cuda graph capture to speed up init

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-07-24 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | ready, v1, startup-ux |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 gpu_model_runner.py 与 envs.py 两个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/21146 |

## 一句话结论

CUDA graph capture 期间用 `gc.freeze()` 冻结存量对象、消除 capture 循环中反复触发的 Python GC,Qwen3-0.6B 的 capture 时间从 35 秒降到 2 秒。

## 动机

vLLM 启动时要对几十个 batch size 逐一捕获 CUDA graph。capture 循环会持续分配/释放 Python 对象,导致 CPython 分代 GC 在整个过程中反复全堆扫描;对小模型而言 capture 本身很轻,GC 反而成为主导开销(Qwen3-0.6B capture 67 个形状耗 35 秒),严重拖慢服务启动。

## 优化点

在 `capture_model` 中包一层 `freeze_gc` contextmanager:capture 前先 `gc.collect()` 把当前垃圾清干净,再调用 `gc.freeze()` 把所有存活对象从后续 GC 的候选集合中排除,整个 capture 循环期间 GC 不再扫描这批对象,结束时 `gc.unfreeze()` 恢复。新增环境变量 `VLLM_ENABLE_CUDAGRAPH_GC`(默认 0,即冻结)允许恢复 GC 运行。作者指出更彻底的方案是未来 PyTorch 版本引入 pytorch/pytorch#158193。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: capture_model 中新增 freeze_gc contextmanager,与 graph_capture 叠加使用
vllm/envs.py: 新增 VLLM_ENABLE_CUDAGRAPH_GC 开关(默认冻结 GC)
```

关键片段(gpu_model_runner.py):

```python
@contextmanager
def freeze_gc():
    gc.collect()
    should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
...
with freeze_gc(), graph_capture(device=self.device):
```

## 性能声明

`vllm serve Qwen/Qwen3-0.6B`(PR 未注明 GPU 型号):
- Before:capture 67 个形状耗时 34 秒(1.92 it/s),日志 "Graph capturing finished in 35 secs, took 0.59 GiB"。
- After:同样 67 个形状 2 秒完成(28.07 it/s),显存占用 0.59 GiB 不变。
即 capture 时间约 17 倍缩短;PR 未提供其他模型或 E2E 数字。

## 验证与回归风险

验证为前后启动日志对比,diff 中未新增单元测试。风险面:capture 期间 GC 被抑制,若模型/框架在 capture 中产生大量新垃圾,内存峰值可能升高(capture 结束后恢复,实测 capture 显存 0.59 GiB 不变);默认行为改变可通过 `VLLM_ENABLE_CUDAGRAPH_GC=1` 回退;改动位于 v1 GPU model runner 的 capture 入口,不影响推理执行路径。

## 关联

同分类(CUDA Graph):#20059(full cudagraph 与独立 sampling 支持,为后续默认化铺路)、#25444(默认 FULL_AND_PIECEWISE,capture 数量与启动时间相关)、#22763(Cutlass full CUDA graph 支持)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
