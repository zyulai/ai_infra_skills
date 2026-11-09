# PR #56492: [Core] Extensible KV cache

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @njhill |
| 合并 | 2026-11-02 |
| 分类 | KV Cache / Memory(major) |
| Label | frontend, ready, ray, torch.compile, ci/build, cpu, kv-connector, nvidia, mrv2, scheduler, kv-cache-manager, mooncake |
| Diff | 素材未提供 diffstat 数字;pack diff 收录测试文件为主,47 个文件因超长未收录(实现文件在 PR 中) |
| 链接 | https://github.com/vllm-project/vllm/pull/56492 |

## 一句话结论

KV cache 用 VMM 虚拟地址预留 + 按需物理映射,warmup 后按实测提交最终大小:GB200 默认多出 4–10% KV tokens,引擎初始化快 21–34%。

## 动机

KV cache 必须在 CUDA graph capture 之前存在(captured graph 烘焙其地址),因此现行实现按 capture/warmup 之前的 profiling 估计来定大小,之后的内存只能靠猜:graph 内存要么被忽略、要么用一次"丢弃式" capture pass 估计(拖慢启动);warmup 与运行期 transient(attention workspace、sampler/logprobs 缓冲、spec-decode drafts)只靠 slack 吸收 —— 即 `gpu_memory_utilization=0.92` 默认留下的 8% 显存(192 GiB 卡上约 15 GiB 闲置);估计过于乐观时运行期照样 OOM。

## 优化点

`ExtensibleTensor`(vllm/utils/extensible_tensor.py + vmm_driver.py)通过 ctypes 调 CUDA `cuMem*`/HIP `hipMem*` VMM API:先为全容量预留虚拟地址范围、不映射物理内存,按 driver granule 按需提交物理页(新提交块清零),经 DLPack 暴露给 torch。`ExtensibleKVCache` 作为 allocator hook 挂进 `allocate_kv_cache`,按层(或张量)切分预留段,kernel 与 layer view 无感知;capture 前仅 null block 有物理 backing,warmup 提交其触达的块。warmup 结束后每个 worker 实测空闲内存与 transient 峰值(取 profiling 峰与 warmup 峰的较大者,另留 `max(1 GiB, 0.75×transient)` margin),engine 取跨 worker/DP 最小值,一步 `extend_kv_cache` RPC 提交最终 block 数 —— 地址不变,已捕获 graph 与 layer view 继续有效。开启该 feature 且未显式设 utilization 时默认 1.0,并跳过 graph 内存 profiling pass;不支持的平台/配置自动回退估计式 sizing;sleep 模式 unmap/remap 同地址;weight reload 可借用 cache 已释放的物理页。

## 关键改动

```text
vllm/utils/extensible_tensor.py / vllm/utils/vmm_driver.py: VMM 虚拟预留 + 按需物理映射的张量(ctypes cuMem*/hipMem*)
vllm/v1/worker/extensible_kv_cache.py: ExtensibleKVCache allocator hook、warmup 后实测提交
vllm/v1/worker/gpu_worker.py / vllm/v1/core/kv_cache_planning.py: _measure_kv_cache_blocks 与 finalize/shrink/align 逻辑
tests/v1/worker/test_extensible_kv_cache.py 等: 新增 VMM、planning、CLI 行为测试
```

关键片段(实测提交流程,来自 PR 描述):

```text
each worker: blocks = budget - measured_transient_peak - max(1GiB, 0.75*peak)
engine: min(workers, DP engines) -> shrink configs -> extend_kv_cache RPC
```

## 性能声明

GB200(189 GiB/GPU)、默认设置、每 start 为新进程、3 轮交错取中位数:
- KV 容量:8 个模型默认多 4.1%–9.5% tokens(4.9–11.3 GiB/GPU),多数 7–10%,如 Qwen3-32B 103.1→113.0 GiB(+9.5%)、Qwen3-8B +7.5%;DeepSeek-V4-Flash 最低(+4.1%,其 graph/workspace 需约 15 GiB)。显式 `gpu_memory_utilization=0.92` 时反而比现状小 0.6–3.7%(减去了实测的 graph 与 warmup 内存)。
- 启动:engine init(不含权重加载)下降 21–34%(5–34 s),如 Qwen3-235B-A22B-FP8 TP4:76.1→50.1 s(−34%);端到端启动下降 6–18%。主要节省来自省掉估计 graph 内存的丢弃式 capture pass。
- 不再需要为避免 OOM 手调 `gpu_memory_utilization`;该参数保留为共享 GPU 的上限。

## 验证与回归风险

单测覆盖 ExtensibleTensor/VMM driver、ExtensibleKVCache 布局与 commit、sizing 公式、torch 内存上限、CLI/config 解析、planning shrink/align、connector 兼容矩阵;E2E 验证 greedy 输出开/关一致、sleep/wake 前后一致;NIXL P/D 精度在 GB200 与 ROCm(补丁版 UCX/ROCr)验证。风险面:仅 CUDA/ROCm + V2 model runner 默认开启,不支持场景(HiSparse、Elastic EP、`kv_cache_memory_bytes`/`num_gpu_blocks_override`、LMCacheMPConnector、custom_mem_pool connector、ROCm 上 NIXL/Mooncake/MoRIIO 等)自动 off,显式开启报错; ROCm 需 TheRock ≥ 7.12(HIP VMM 缺陷探测回退);服务中直接 `reload_weights` 无清理 pause 时可能告警或失败。

## 关联

产品化自 @zhuohan123 在 #47363 的演示,替代 #50779。同区域(KV cache 显存管理):#44594(KV cache watermark)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
