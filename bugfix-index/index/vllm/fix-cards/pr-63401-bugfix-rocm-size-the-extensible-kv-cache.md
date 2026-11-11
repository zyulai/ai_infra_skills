# PR #63401: [Bugfix][ROCm] Size the extensible KV cache from exact VRAM use, not 2 MB-rounded free memory

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#63401](https://github.com/vllm-project/vllm/pull/63401) |
| 作者 | GodRishUniverse |
| 合并日期 | 2026-11-08 |
| labels | bug, ci/build, documentation, mrv2, nvidia, ready, rocm |
| 热度 | 👍 0 · 💬 20 |
| 分类 | KV Cache / Memory |
| bug 类型 | resource |

## 一句话结论
ROCm 的 hipMemGetInfo 把每笔分配按至少 2 MB 计,warmup 后可扩展 KV cache 少算几个 GB;改用 amdgpu 精确 VRAM 计数补回,Llama 3.1 8B KV 容量 +48%。

## 问题现象
正文 Overview:在 ROCm 上,`hipMemGetInfo` 把每笔分配计为至少 2 MB;warmup 期间加载的大量小 kernel code object 让空闲内存看起来比实际少几个 GB,可扩展 KV cache(#56492)据此自 sizing,容量偏小。仅 unpartitioned discrete ROCm GPU 受影响。实测(Radeon AI PRO R9700 32 GB,`--gpu-memory-utilization 0.97`,Llama 3.1 8B,每配置单次运行):KV cache 从 51,712 增至 76,288 tokens(+48%),concurrency 128 下 output tok/s 918.9 -> 1,054.0,median TTFT 3,934 -> 1,084 ms,无 OOM、无失败请求。

## 根因
正文「Root cause」段披露:amdgpu 驱动故意把每笔分配向上取整到 2 MB 后计入所报告的空闲内存(commit 1ac354beecfd,「drm/amdgpu: Pessimistic availability based on rounded up allocations」),除非 GPU 内存保留在系统内存(`apu_prefer_gtt`);驱动自身的分配检查用的却是精确尺寸。warmup 加载的许多小 kernel code object 因此隐藏了几个 GB。sysfs 计数器 `mem_info_vram_used` 按整个设备精确计数。作者另注:`amdsmi_get_gpu_memory_usage` 应读到同一计数,但读不到时会回问 KFD、又回到 2 MB 取整值。

## 修复方案
新增 `vllm/platforms/rocm_device_memory.py`:`find_vram_used_file` 经 KFD topology + DRM sysfs 按 UUID/PCI 定位本 GPU 的 `mem_info_vram_used`(跳过集成显卡、分区 GPU、计数缺失或 VRAM total 与 HIP 不匹配的设备,容差 1%);`exact_free_memory` 采样计算 `hidden = total - used - free` 并加回,直到读数稳定(0.2 s 未上升,上限 2 s)。

平台钩子:`Platform.free_device_memory`(interface.py,默认即原 `device_memory_info` 语义),ROCm 平台覆写之;`extensible_kv_cache.py`、`gpu_worker.py` 改用该钩子,启用 extensible cache 时 `init_free_memory` 也按新钩子读取:

```python
@classmethod
def free_device_memory(cls, device: torch.device | None = None) -> int:
    free = rocm_device_memory.exact_free_memory(device)
    return super().free_device_memory(device) if free is None else free
```

补回量永不超过物理空闲、永不为负;其他进程的分配不加回。配套新增 sysfs 查找(无 GPU 可跑)与 ROCm GPU 测试。

## 影响与回归风险
- 影响窗口:正文未说明受影响版本/提交范围,不写。
- 回归面:按 diff 与正文 Limitations——仅 unpartitioned discrete ROCm GPU 生效,iGPU、分区 GPU、sysfs 缺失/不匹配及非 ROCm 平台走原路径(no-op hooks);这些 GPU 上 `gpu_memory_utilization` 从此约束精确占用。单 GPU 多实例:开启 extensible cache 时 `request_memory` 跳过启动检查,先启动实例的取整已计入后启动实例的 baseline 且不加回,后启动实例 cache 偏小(依启动顺序),但不会 OOM。KFD 页表保留与 RAS 页不在精确计数内(R9700 约 80 MB),由 3 GB ROCm margin floor(#65418)覆盖。仅在 R9700(gfx1201)验证;CDNA 依驱动代码推断行为相同但未验证;AMD CI 的 `mi300_1`/`mi355_1` 池是否分区未知,若非分区则该路径在未验证状态运行。V1 runner 与 `--no-enable-extensible-kv-cache` 不变。

## 关联
- [#56492](https://github.com/vllm-project/vllm/pull/56492)
- [#65413](https://github.com/vllm-project/vllm/pull/65413)
- [#65418](https://github.com/vllm-project/vllm/pull/65418)
- [#55828](https://github.com/vllm-project/vllm/pull/55828)
- [#60121](https://github.com/vllm-project/vllm/pull/60121)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
