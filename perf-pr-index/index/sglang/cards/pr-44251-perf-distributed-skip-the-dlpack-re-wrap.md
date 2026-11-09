# PR #44251: [Perf] distributed: Skip the DLPack re-wrap of custom all-reduce results

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-10-17 |
| 分类 | Parallelism / Communication (tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(单文件,+4 行) |
| 链接 | https://github.com/sgl-project/sglang/pull/44251 |

## 一句话结论

custom all-reduce 结果已是 torch.Tensor 时跳过 `torch.from_dlpack` 重包装,每次调用省约 15-20 µs host 时间,GB300 TP2 eager 生图延迟 -6% 到 -11%。

## 动机

`sglang/kernels/ops/communication/all_reduce.py` 的 `custom_all_reduce` 以 `torch.from_dlpack(result)` 收尾。在 apache-tvm-ffi 0.1.11 且 torch 的 DLPack exchange API(`torch_c_dlpack_ext`,lmsysorg/sglang 镜像内含)加载后,tvm-ffi 调用本就返回 `torch.Tensor`,`torch.from_dlpack` 变成纯 torch → DLPack → torch round trip,期间还要从 device 字符串构建 current-stream 对象并做一次可能重试的 `__dlpack__` 调用。在 Grace host 的 GB300 上这一 round trip 每次调用约 20 µs——约为 eager TP all-reduce ~42 µs host 时间的一半,是 tvm-ffi 调用本身成本的两倍。Diffusion DiT 在 eager 模式下每个 block 要做多次 all-reduce,GB300 TP 下 step 是 host-bound,该成本直接体现为端到端延迟。

## 优化点

一行级修复:`custom_all_reduce` 在 `module.all_reduce(...)` 返回后,若 `result` 已是 `torch.Tensor` 则直接返回,跳过 `torch.from_dlpack`;否则(例如设置 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 时)保留原 `from_dlpack` 分支。返回张量与原先别名同一块内存;`from_dlpack` 传入的是 current stream,round trip 本来就不增加跨 stream 等待,去掉它不改变任何顺序;不涉及内存分配,CUDA-graph capture 与编译路径(`diffusion_all_reduce`)均不变。

## 关键改动

```text
python/sglang/kernels/ops/communication/all_reduce.py: custom_all_reduce 增加 isinstance(result, torch.Tensor) 快速返回
```

```python
result = module.all_reduce(comm, input, algo.algo_name, graph_params, use_multicast)
# Already a torch.Tensor once tvm-ffi loads torch's DLPack exchange API;
# re-wrapping it is a pure round trip (~15 us per call on GB300).
if isinstance(result, torch.Tensor):
    return result
return torch.from_dlpack(result)
```

## 性能声明

GB300 x2(Grace host,进程 pin 到 GPU 的 NUMA 节点),base `c54ba66cd4`,torch 2.13,apache-tvm-ffi 0.1.11。Host time per call(eager、inference mode、GPU 保持忙碌只计时提交):diffusion TP 组 `GroupCoordinator.all_reduce` 0.4 MiB 42.0 → 20.7 µs、4.5 MiB 39.2 → 18.2 µs、27 MiB 43.7 → 20.6 µs(tvm-ffi 调用本身约 10 µs)。E2E(client wall clock 中位,TP2 eager,主/候选 server 常驻同机、请求 10 对轮流):Qwen-Image-2.1 512² 1462.4 → 1298.3 ms(-11.2%,10/10)、1024² -10.1%,FLUX.1-dev 512² -6.2%、1024² -8.0%,均 10/10 胜。

H200 x2(Xeon 8558P,TP2 eager,base `c15edf314d`):`custom_all_reduce` host time(bf16 [1,4096,3072])22.39 → 6.42 µs(x86 上 round trip 同样约 16 µs)。E2E 跨实例对比 Qwen-Image 512² 797.0 → 686.6 ms、FLUX.1 512² 2196.0 → 1856.9 ms,但该机上两个完全相同的常驻 server 可差至 3.5%,跨实例列仅供参考;#44642 的进程内 A/B(同样消除 re-wrap)在本 PR 之上再测得 -0.2%(Qwen-Image)/-2.4%(FLUX.1)。SRT 侧经 `CustomAllReduceV2` 的 eager TP all-reduce(CUDA graph 外的 extend/prefill)同样受益,但未测量。

## 验证与回归风险

`test_custom_all_reduce.py --num-gpu 2` 在 GB300 x2:exchange API 加载时 180 passed;`TVM_FFI_DISABLE_TORCH_C_DLPACK=1` 下 180 passed(覆盖 `from_dlpack` 分支)。E2E 所有配置输出图像 md5 与 main 一致。风险极小:仅当 tvm-ffi 返回非 torch.Tensor 时行为才与原先不同,且该分支原样保留;不新增测试(Checklist 注明既有测试覆盖两分支);CI Extra 未通过。

## 关联

与 #43859、#44158 相互独立(它们在别处削减 host 时间);#44642(进程内 A/B,绕过该 wrapper 同样消除 re-wrap)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
