# PR #20343: HiSparse for Sparse Attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xiezhq-hermann |
| 合并 | 2026-03-23 |
| 分类 | Attention / Kernel |
| Label | high priority, deepseek, run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含大量文件,另有 17 个文件因超长被丢弃 |
| 链接 | https://github.com/sgl-project/sglang/pull/20343 |

## 一句话结论

引入 HiSparse:解码时把 NSA 稀疏注意力(DeepSeek-V3.2、GLM-5)的闲置 KV cache 卸载到 CPU 内存,扩大可支撑的 batch size 与吞吐。

## 动机

NSA 稀疏注意力模型每步解码只访问被 indexer 选中的少量 top-k token 的 KV,但显存里仍要为完整序列保留 KV cache,大量"闲置"KV 占据 GPU 显存,限制了 batch size。HiSparse 把这部分闲置 KV 存到 CPU 内存,按需把 top-k token 的 KV swap-in 到 GPU,从而提高显存利用与吞吐。本 PR 是前一次稀疏模型 hicache 支持(#14619,@hzh0425 @huangtingwei9988)之后的方案。

## 优化点

核心是"设备热缓冲 + 主机冷存 + 单内核 swap-in"三层结构。设备侧为每个请求维护固定大小的 hot buffer(LRU 管理)加一个额外 page 放最新 token;主机侧 `MLATokenToKVPoolHost` 按 `host_to_device_ratio`(默认 2)扩充冷存容量,prefill 结束后由独立的 `write_staging_stream` 异步把全部 prefill KV backup 到 host(`ack_staging_queue` + event 查询 + TP all_reduce MIN 同步各 rank 就绪数)。解码每步只把上一步 token 的 KV eager backup 到 host,保持冷存完整。当 NSA indexer 选出 top-k token 后,新增 JIT CUDA 内核 `load_cache_to_device_buffer` 单次完成:top-k token 在共享内存 open-addressing 哈希表(Knuth 乘法哈希)中登记,对 hot buffer 做 warp 级 hit/evict 前缀和扫描(ballot + popc + inclusive scan),hit 直接复用、miss 按淘汰 LRU slot 从 host 拷贝(`ld.global.nc` / `st.global.cg` 64-bit 分块拷贝),输出 `top_k_device_locs` 供 attention 读取;kernel 模板按 `(BLOCK_SIZE, NUM_TOP_K, HOT_BUFFER_SIZE, IsMLA)` JIT 特化,`num_real_reqs` 标量让 CUDA graph pad 块提前退出(graph-safe)。协调器 `HiSparseCoordinator` 管理请求 admission(staging)、device buffer 增长(bulk alloc)与 loc 映射;nsa_backend 在 decode 路径经 `swap_in_selected_pages` 生成 page_table_1,并对 HiSparse 强制 `force_unfused_topk`(fused topk 路径不兼容)、禁用 MHA one-shot prefill;extend 路径把逻辑 loc 翻译为 hisparse device loc。当前仅支持 MLA(token 级稀疏,page-wise 稀疏为 todo)。

## 关键改动

```text
python/sglang/jit_kernel/csrc/hisparse.cuh: 新增 load_cache_to_device_buffer_kernel(哈希 hit 检测 + LRU 前缀和 + host->device warp 拷贝)
python/sglang/jit_kernel/hisparse.py: JIT 封装 load_cache_to_device_buffer_mla(模板特化缓存)
python/sglang/srt/managers/hisparse_coordinator.py: HiSparseCoordinator(staging 异步 backup、device buffer 增长、swap_in_selected_pages)
python/sglang/srt/mem_cache/hisparse_memory_pool.py: HiSparseNSATokenToKVPool / HiSparseTokenToKVPoolAllocator
python/sglang/srt/layers/attention/nsa_backend.py: decode/extend 接 swap-in 与 loc 翻译,force_unfused_topk
```

## 性能声明

PR 正文未给出 benchmark 数字(motivation 仅为定性目标:利用 CPU 内存扩大 batch size、提升 NSA 模型吞吐;Benchmarking 章节留空)。性能收益取决于 swap-in 带宽与 top-k 命中率,PR 未注明实测数据。

## 验证与回归风险

代码内含调试用 `naive_load_topk`(per-request python 循环)作为 swap-in 内核的验证对照。diff 中被丢弃的 17 个文件可能包含注册测试。风险面:仅覆盖 MLA 布局与 token 级稀疏;nsa prefill MHA one-shot 路径被禁用(`hisparse_coordinator is None` 条件);fused topk 在 HiSparse decode 下强制关闭,可能损失该路径优化;host pool 分配失败直接 RuntimeError 中断请求;`_eager_backup_previous_token` 与 `_grow_device_buffers` 在 CPU 上逐请求循环,可能成为高 batch 下的 host 瓶颈;TP 下就绪同步依赖 all_reduce MIN。

## 关联

前序:PR #14619(稀疏模型 hicache 支持,同一作者群的先行尝试)。同区域:NSA/DSA 稀疏注意力栈(DeepSeek-V3.2、GLM-5)及后续 DCP(PR #43735 注明 DSA DCP 拒绝 HiSparse 组合)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
