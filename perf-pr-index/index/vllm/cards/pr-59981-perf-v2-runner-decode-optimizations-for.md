# PR #59981: [Perf] V2 runner decode optimizations for NVIDIA Confidential Computing

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ybgao-nvidia |
| 合并 | 2026-11-09 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | speculative-decoding, ready, nvidia, mrv2, verified |
| Diff | pack diffstat 缺失(+/− 未知);按 diff 文本计数可见 12 个文件,另有 5 个文件因体积被丢弃,diff 末尾截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/59981 |

## 一句话结论

把 NVIDIA 机密计算(CC)下 V2 runner 的 decode 吞吐损失从 29–52% 压到 7–10%,恢复到非 CC 基线的 91–93%。

## 动机

NVIDIA Confidential Computing 通过 bounce buffer 加密 PCIe 流量,对 V2 runner 有两个后果:其一,pinned host memory 从设备端不连贯——`UvaBufferPool` 让 GPU 经 UVA 直接读 pinned 主机内存,CC 下 GPU 读到的是陈旧快照,模型输出变成垃圾(问题最早在 #57224 报告);其二,`cudaMemcpyAsync` 在 pinned 内存上变成 host-synchronous,发起线程要等拷贝及其 stream 上全部在途工作完成,导致 H2D 输入拷贝把调度线程阻塞在在途 forward 之后,每步 D2H token 读回又拖延下一次 CUDA graph launch。本 PR 取代 #52226,把 CC decode 优化移植到 V2 GPU model runner 并修复 UVA 不连贯 bug。

## 优化点

全部改动以 CC 检测为门控,非 CC 路径不动。第一,新增 `Platform.is_confidential_compute()`,CUDA 上经 NVML 自动检测,`VLLM_CONFIDENTIAL_COMPUTE=0/1` 可覆盖。第二,CC 下 `is_uva_available()` 返回 `False`,缓冲池回退到显式拷贝;对 pinned 槽位注册 in-place H2D 目标,保持"槽位仅在无 GPU 读端在途时复用"的池不变式。第三,H2D 走 staged copy:在空闲的 staging stream 上发起 host-synchronous 拷贝使主机只等传输本身,再用 `wait_stream`/`record_stream` 把 compute stream 排序在其后;`install()` 直接 patch `torch.Tensor.copy_/to/cuda` 入口,`encoder_decoder.py` 等调用点无需 CC 特判。第四,D2H 延迟化:`AsyncOutput`/`AsyncPoolingOutput` 构造时在主 stream 记录 source-ready 事件,`get_output()` 在 executor 输出线程上发 D2H 拷贝,不再阻塞调度线程;D2H stream 在 torch 的 32 路 stream 池之外创建,避免句柄复用导致事件被重定向。作者还观察到 CC 下在空闲 copy stream 上记录的事件会迟到一个整 step 才触发,改为把事件记在 producer stream 上。

## 关键改动

```text
vllm/v1/conf_compute_utils.py: 新模块,install() patch torch 拷贝/事件入口,staged H2D + 工作线程 D2H
vllm/platforms/cuda.py: NVML 检测 CC + 环境变量覆盖
vllm/utils/platform_utils.py: is_uva_available() 在 CC 下返回 False
vllm/v1/worker/gpu/buffer_utils.py: NonUvaBuffer 支持 pinned 回退并注册 in-place H2D 目标
vllm/v1/worker/gpu/async_utils.py: new_d2h_stream() 门控封装;fault flag 改在主 stream 计算
vllm/v1/worker/gpu/model_runner.py: output_copy_stream 换用 new_d2h_stream,CC 下安装补丁
```

```python
def _h2d(src, device, dtype=None, out=None):
    cur = torch.cuda.current_stream(device)
    staging = _staging_stream(cur.device_index)
    with torch.cuda.stream(staging):
        tmp = _orig_to(src, device=cur.device, dtype=dtype, non_blocking=True)
    cur.wait_stream(staging)
    tmp.record_stream(cur)
    return tmp if out is None else _orig_copy_(out, tmp, non_blocking=True)
```

## 性能声明

Decode benchmark:Qwen3.5-397B-A17B-FP8,TP4,B300,CC enabled,对比 CC off 与 main 分支 CC on。输出 tok/s per GPU(concurrency 4/8/16/32/64/128):CC off 129/212/352/534/796/1229;main, CC on 85/147/250/397/617/953;本 PR, CC on 118/198/322/490/741/1120。每 token CC slowdown 从 51.8%/44.2%/40.8%/34.5%/29.0%/29.0% 降到 9.3%/7.1%/9.3%/9.0%/7.4%/9.7%,即各并发下从 29–52% 降到 7–10%,CC 吞吐达非 CC 基线的 91–93%。

## 验证与回归风险

PR 正文列出新增 `tests/v1/worker/test_conf_compute_copies.py`(CC 下 UVA 可用性、`staged_h2d()` pinned/pageable 源、`async_tensor_h2d()` 路由、`StagedH2DBuffer` 池 round-trip、deferred D2H 语义);diff 中另见新增 `tests/v1/worker/test_async_output_fault_flag.py`(EP fault flag 保留到拷贝完成)与 `tests/v1/worker/test_conf_compute_streams.py`(池外 D2H stream、graph capture/join、ROCm 导出解析)。风险集中在 `install()` 的全局 monkeypatch(`torch.Tensor.copy_/to/cuda`、Event/Stream 方法):补丁对 torch.compile tracing 与 CUDA graph capture 期间的拷贝做了 bypass,超过 `_MAX_STAGED_BYTES`(32 MiB)的 H2D(如 KV cache swap)也回退原路径;非 CC 进程不受影响,但事件子类化使 C 侧创建的事件不被覆盖,均为作者已注明的边界。

## 关联

取代 #52226(V1 runner 上的同方向优化);UVA 不连贯问题首报于 #57224,#57404 曾以运行时 coherence probe(`is_uva_available()`)独立修复,本 PR 改为 CC 检测门控并加 staged-copy fallback。无其他同分类 PR 信息。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
