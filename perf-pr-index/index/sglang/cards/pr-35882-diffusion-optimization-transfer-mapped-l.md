# PR #35882: [diffusion] optimization: transfer mapped layers through a courier thread

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mickqian |
| 合并 | 2026-08-22 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | run-ci, diffusion |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 envs.py、gpu_worker.py、host_memory_budget.py、layerwise_offload.py、测试与 perf baseline 文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/35882 |

## 一句话结论

新增 courier 工作线程把 checkpoint 映射层(mapped layers)的搬运移出 compute 线程,MiniMax-H3 视频去噪步总耗时从 538.8 s 降到 369.1 s,快 31%。

## 动机

以 checkpoint 文件映射(mmap)为源的 H2D 拷贝无论怎么请求都是同步的——驱动要先经自己的缓冲区搬运未 pin 的内存;而 prefetch hook 跑在 compute 线程上,导致每个 mapped layer 的传输都会阻塞 denoise step 整整一次传输的时间。在显存受限的消费级卡 + layerwise offload 场景下,这成为去噪阶段的主要瓶颈之一。

## 优化点

新增 `MappedLayerCourier` 守护线程:compute 线程 `prefetch_layer(non_blocking=True)` 时把 mapped layer 交给 courier,courier 将该层字节 stage 进两个 pinned slot 之一(页缓存命中时退化为普通 memcpy),再在自己的 CUDA stream 上发起真正异步的 device copy 并 record event。compute 线程的大 tensor 拷贝会释放 GIL,所以读 layer i+1 与计算 layer i 真正重叠。安全性设计:参数绑定(bind)仍在 compute 线程的 collect 阶段执行,模块状态不会被并发触碰;`release_all` 会先 drain 在途层;courier 死亡或 kill-switch 环境变量(`SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER`)置位时回退直接同步拷贝路径。同时新增测试钩子环境变量,可在 CI 上模拟小显存卡(12 GiB)与小 host 内存(32 GiB)以覆盖受限放置路径。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py: 新增 MappedLayerCourier(2 个 pinned slot、独立 stream、submit/collect 握手与失败回退),prefetch_layer 分流 mapped 层
python/sglang/multimodal_gen/envs.py: 新增 courier kill-switch 与测试用 host/device 内存模拟环境变量
python/sglang/multimodal_gen/runtime/managers/memory_managers/host_memory_budget.py: host_memory_available_bytes 支持测试强制值
python/sglang/multimodal_gen/test/server/gpu_cases.py: 新增 MiniMax-H3 12 GiB 消费级卡预算用例(run_perf_check)
python/sglang/multimodal_gen/test/server/perf_baselines/5090.json: 该用例的 perf baseline(median denoise 11000 ms)
```

```python
def _ship(self, layer_idx, slot_turn):
    slot = self._slots[slot_turn]
    ... window.copy_(cpu_tensor)  # stage 进 pinned slot
    with torch.get_device_module().stream(self._stream):
        gpu_tensor.copy_(window, non_blocking=True)
        event.record(self._stream)   # 独立 stream 上真异步
```

## 性能声明

测试条件(PR 注明):MiniMax-H3 fl2va,单张 RTX 4090,12 GiB allocator cap,32 GiB host envelope,864×480 / 124 frames / 20 NFE,interleaved same-window A/B:

- baseline:denoise 总计 **538.8 s**(中位 25.0 s/step);with courier:**369.1 s**(中位 18.0 s/step)→ **31% faster**;作者指出 host 内存带宽争用越大效果越明显。
- 输出等价性:与 baseline 的视频 SSIM(0.9941)等于 baseline 自身 run-to-run SSIM(0.9941),音频流 bit-identical。
- 与 #35867(per-layer pinning)、#35862(VAE 留在 mapping)叠加后,同一预算下 lossless bf16 denoise 两轮 interleaved 中位 **11.0–12.0 s/step**,优于同 GPU、同 32 GiB psutil 约定、同带宽窗口下测得的 **ComfyUI 13.0 s/step**。

## 验证与回归风险

新增 4 个单测(`test_layerwise_offload.py`):courier 传输正确性、kill-switch 强制同步路径、`release_all` drain 在途层、courier 中途死亡回退同步拷贝(前两个测试在 courier 关闭时会失败,保证路径被覆盖);新增 `test_host_memory_budget.py` 用例验证强制 host 尺寸行为;新增 5090 GPU 集成用例并写入 perf baseline(expected_median_denoise_ms 11000, runtime_peak_vram_mb 12288)。风险:mapped 层路径变化影响 diffusion layerwise offload 全链路,非 CUDA 平台 `pin_slots=False`;courier 与 #35867 触及同一文件,合并顺序需 rebase(PR merge note 注明)。CI 状态中 Extra run 显示失败(:x:),Base run 通过。

## 关联

前序:#35867(per-layer pinning)、#35862(VAE 保持在其 mapping 上),三者叠加到达 11.0–12.0 s/step;同分类 diffusion 加速链路。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
