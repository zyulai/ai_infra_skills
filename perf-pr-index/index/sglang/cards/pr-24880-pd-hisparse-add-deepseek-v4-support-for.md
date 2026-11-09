# PR #24880: [PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @huangtingwei9988 |
| 合并 | 2026-06-05 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | high priority, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留 11 个文件并注明 "2 file(s) dropped" |
| 链接 | https://github.com/sgl-project/sglang/pull/24880 |

## 一句话结论

让 HiSparse 的 PD 直传(Prefill→Decode DRAM)支持 DeepSeek V4:host 侧统一为 page-padded C4 布局,新增 DSV4 C4 页间直传 JIT kernel,免去传输时的布局转换。

## 动机

正文 Motivation/Modifications 为空(模板未填),机制信息取自 diff:此前 DSV4 的 HiSparse host offload 走 `deepseek_v4/hisparse_transfer.cuh` 的 `offload_to_cpu` 持久 kernel,其 host cache 是无 padding 的 trivial linear 布局,而设备侧是 page-padded 布局(每页 64 token、576 字节对齐)——两种布局不一致,PD 直传 host DRAM 页到 decode 设备时无法逐页拷贝,需要转换。本 PR 把两侧布局统一为 page-padded C4,使 prefill 侧 host DRAM 页可被 decode 侧直接消费。

## 优化点

(1) `kvcacheio.cuh` 布局统一:删除 `get_pointer_gpu`/`get_pointer_cpu` 双路径,合并为 `get_pointer_paged`,host 与 GPU 共用同一 `kPageSize=64`、`kPageBytes`(按 576 字节对齐)、`kScaleOffset` 的 paged C4 布局。(2) 删除旧 `hisparse_transfer.cuh`(offload_to_cpu 持久 kernel)及其 python 包装,`hisparse.cuh` 新增通用 `transfer_cache_dsv4_mla<BLOCK_SIZE>` kernel(warp 粒度循环,按 (src,dst) 索引对逐层 `transfer_item`),经 JIT 包装 `transfer_cache_dsv4_mla`(block_size 默认 1024,functools.cache 模块缓存)暴露;`load_cache_to_device_buffer_dsv4_mla` 注释改为 "page-padded device + page-padded host C4 layout"。(3) PD 侧接线:`decode.py` 的 KV manager 在 HiSparse + DSV4 池时把设备 C4 buffer 的 contiguous 信息(第 c4_layer_num 层之后)追加进传输列表;`pop_preallocated` 对 direct-to-host 的索引用 allocator 的 `hisparse_page_size`(压缩页大小)而非逻辑 page_size 计算页索引;`_pre_alloc` 的 host 索引分配改用 `coordinator.host_token_len(fill_len)`。(4) `hisparse_coordinator.py`、`hisparse_memory_pool.py`、`memory_pool_host.py` 配合 paged host 布局调整分配/映射。

## 关键改动

```text
python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh: host/GPU 统一 paged C4 布局
python/sglang/jit_kernel/csrc/hisparse.cuh: transfer_cache_dsv4_mla 页间直传 kernel(替换 offload_to_cpu)
python/sglang/srt/disaggregation/decode.py: 设备 C4 buffer 追加进 PD 传输 + hisparse_page_size 页索引
python/sglang/srt/mem_cache/hisparse_memory_pool.py, memory_pool_host.py: paged host 池适配
```

## 性能声明

无速度对比数据(Speed Tests 章节为空)。仅有一次 GSM8K 200 样本运行:Score 0.975,Output throughput 27.094 token/s,Total latency 693.701 s(模型/硬件/拓扑条件未注明,亦无 baseline 对照)。

## 验证与回归风险

`jit_kernel/tests/test_hisparse.py` 更新覆盖新 kernel;checklist 五项全部未勾选。风险面:host 池布局从 linear 改 paged 是持久化格式变化,升级时旧 host 缓存数据不兼容;PD 直传路径仅覆盖 Mooncake 相关索引序列化(前序 #37771 显示非 Mooncake backend 走 NotImplementedError);`hisparse_page_size` 与逻辑 page_size 的区分依赖 allocator 属性存在,非 DSV4 HiSparse 路径回退原 page_size。

## 关联

同区域:PR #37771(HiSparse 投机解码多步 swap,后续)、PR #23882(DeepSeek V4 主支持);HiSparse hicache 系列。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
