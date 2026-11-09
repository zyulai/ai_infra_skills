# PR #53781: [3/N] HiSparse: host-resident sparse-MLA decode hot-buffering

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-09-12 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, speculative-decoding, ready, torch.compile, ci/build, v1, multi-modality, deepseek, cpu, kv-connector, nvidia, mrv2, DSv4, dflash, scheduler, kv-cache-manager, glm, minimax, mooncake, hisparse |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 123 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/53781 |

## 一句话结论

为 sparse-MLA decode 在 GPU KV cache 之下增加 host 常驻层:GPU 放不下的 KV page spill 到 pinned host memory,attention 时按 top-k 命中把缺失行热缓冲进每请求 GPU hot buffer。

## 动机

sparse-MLA(GLM / DeepSeek-V3.2 类模型)decode 只需选中的 top-k KV 行,但长上下文下完整 KV 无法常驻 GPU。此前的 #46326 采用"全部 host 常驻"设计;本 PR 取代它,改为尽量把 KV 保留在普通 device cache,只在必须回收 GPU 容量时才把 page spill 到 pinned host pool,避免常驻 host 副本带来的额外内存与同步开销。目标是在 GPU 显存受限的长序列 sparse-MLA decode 场景下维持高吞吐。

## 优化点

机制上,active indexer cache 仍在 device 上产出 sparse attention 需要的逻辑 top-k 位置;一个 fused CUDA resolver 把每个选中位置解析到三种可能来源之一:普通常驻 GPU page、请求 GPU hot buffer 的已有行、pinned host pool。resident 命中先于 hot-cache 查找返回;hot 命中更新 GPU 侧 LRU;miss 则只 gather 被选中的 host 行替换 LRU 条目。解析、替换与 LRU 更新全部留在 GPU 上,兼容 CUDA graph replay。关键设计:HiSparse 不维护私有的 host 侧 indexer KV 副本,indexer 仍是普通可前缀缓存的 GPU cache group;host block 与常驻 block 使用相同 token hash,但仅在内容持久化后才发布 hash,保证不完整的 host 写入不会被前缀复用。swap-in 算法移植自 SGLang 的 `load_cache_to_device_buffer` kernel(sgl jit_kernel/csrc/hisparse.cuh),并适配 vLLM 寻址:token 以全局 KV slot id 为键,host pool row i 即全局 slot i,免去 per-request host 位置表。P/D 侧 NIXL 仍是传输层,decoder 在 admission 时逐请求决定落地策略:前缀能放进 GPU pool 则写常驻 page,否则 bulk sparse-MLA KV 落入注册的 HiSparse host pool。

## 关键改动

```text
csrc/libtorch_stable/hisparse_kernels.cu: 新增 decode hot-buffer kernels(resolve_residency/gather_plan/gather_compact/invalidate_written_slots)
csrc/libtorch_stable/cache_kernels.cu: cp_gather_and_upconvert_fp8_kv_cache 扩展支持 host_cache(pinned)行解析
vllm/(coordinator/connector/runtime 等组件文件在 diff 中被略去,见 PR 全文): HiSparseCoordinator/HiSparseConnector/HiSparseConnectorWorker/HiSparseRuntime
```

关键片段(K/V 循环前的来源解析,cache_kernels.cu):

```cuda
const int32_t device_row = device_row_ids[source_row];
if (device_row >= 0) {
  token_ptr = src_cache + (device_row / block_size) * cache_block_stride +
              (device_row % block_size) * cache_entry_stride;
} else {
  token_ptr = host_cache +
      static_cast<int64_t>(host_row_ids[source_row]) * host_entry_stride;
}
```

resolver kernel 内含 open-addressing hash(Knuth 乘法散列)+ shared-memory 前缀和 compaction,按 LRU 序分类 hit/evictable/miss 并回写 LRU 顺序。

## 性能声明

PR 正文给出 "Preliminary B300 results",为三张吞吐对比图(工作负载标注为 glm_20k_10k、glm_32k_8k、glm_60k_10k,日期 2026-08-20),但具体数字仅在图片中,正文文本未给出数值与对照基线说明;硬件为 NVIDIA B300。除此之外无其他 benchmark 数字。

## 验证与回归风险

diff 中可见对 pinned host memory(`cudaMemoryTypeHost`)、row id 张量 dtype/contiguity 的运行时校验,以及 LRU slot 越界"corruption tripwire"防御(越界 slot 按 no-hit 处理退化为 re-miss)。CUDA-graph padding 行以 -1 发布,保证 graph capture 安全。回归风险面较广:涉及 KV cache manager、scheduler、kv-connector、CPU/多模态等多个 label 区域;spill 路径依赖 worker 对 copy 的 stream-order 确认。指标拆分至 #53782,DeepSeek V4 压缩 cache geometry 拆分至 #53592。scope 限定 NVIDIA sparse-MLA 路径。

## 关联

取代 #46326(always-host-resident 设计);依赖/配套 #53779(connectors 选择外部可传输 cache group)、#53780(NIXL region 携带 stride/capacity/block size 元信息)、可选的 #53263(同节点 P/D 经 GPU memory 中转避免 UCX TCP loopback)。系列标注 3/N。索引内同区域(Attention / Kernel)卡片:#13747(FlashMLA)、#21078(FlashInfer MLA TRTLLM decode)、#12528(MLA decode optimizations)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
