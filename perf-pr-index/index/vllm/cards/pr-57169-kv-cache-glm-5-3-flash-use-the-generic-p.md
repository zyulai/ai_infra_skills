# PR #57169: [KV Cache] GLM-5.3-Flash: use the generic packed KV layout; CircularBufferSpec tail with speculative ring slots

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andakai |
| 合并 | 2026-10-09 |
| 分类 | Speculative Decoding(tier: major) |
| Label | rocm, intel-gpu, speculative-decoding, ready, deepseek, kv-connector, nvidia, mrv2, dflash, kv-cache-manager, glm, DSv4.1, mooncake |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(68 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/57169 |

## 一句话结论

把 kernel block 拆分从整组共享的 block table 下放到每个 AttentionGroup,使 GLM-5.3-Flash 迁移到通用 packed BLHNC KV 布局并删除模型专用 KV 代码。

## 动机

此前 `BlockTable` 为整个 KV cache group 统一拆分 manager block(`kernel_block_sizes`):同组所有 backend 必须共用一个 kernel block size,block table 与 slot mapping 都在 kernel 粒度,而 packed(block-outermost)cache 因为一个 layer 的 blocks 之间夹着其他 layer 的 page,根本无法按组拆分。GLM-5.3-Flash 的 kpool indexer 需要 32/64-token 的 DeepGEMM 页,与同组 sparse MLA 的大 block 冲突,只能靠 GLM 专用的 `storage_block_size` workaround 绕过。

## 优化点

新机制:block table 保持在 manager 粒度,每个 `AttentionGroup` 在 `build_metadata`(含 cudagraph capture 与 drafting 变体)内把 manager block `b` 映射到自己 backend 需要的 kernel blocks `b*stride + j`;映射表按 ubatch 持久预分配,CUDA graph 捕获稳定 buffer;所有 runner 与 drafter(V1/MRv2、EAGLE、Gemma4、Step3.5、DFlash)统一走该路径,Fused multi-step draft decode 在 replay FULL graph 前重映射。`prepare_kernel_block_sizes` 先用 `select_common_block_size` 为组选公共尺寸(packed 组保持 manager block、整组不拆),再由各组自选更小的 `kernel_block_size`。KV 视图绑定用 `as_strided` 按 packed block stride 切出 kernel 视图;仅压缩 cache(`tokens_per_state > 1`)允许拆分,token/pooled cache 拒绝;`block_stride_alignment` 支持 `MultipleOf(n)`。GLM-5.3-Flash 侧:采用 packed `BLHNC` 布局,删除 custom groups、byte layout、`KpoolTailSpec` 与 `storage_block_size`;kpool indexer 变成声明 DeepGEMM 页尺寸(32/64,SM120 上 64)的 compressed spec,tail 是 `kpool + num_speculative_tokens` 大小的 `CircularBufferSpec` ring(带 speculative ring slots)。示例:H200 manager block 640 token、index_kpool=4 时,sparse MLA 不拆(640),indexer 组拆为 5 个 32-token DeepGEMM 页。

## 关键改动

```text
vllm/v1/worker/utils.py: prepare_kernel_block_sizes / select_common_block_size / AttentionGroup 按 backend 映射 kernel blocks
vllm/v1/kv_cache_interface.py: KpoolTailSpec 改为通用 CircularBufferSpec;packed 布局 stride 计算
vllm/v1/attention/backends/mla/indexer.py: Glm5NextIndexerBackend 声明 DeepGEMM 页尺寸的 kernel block
vllm/v1/core/sched/block_table.py(v1)/worker/block_table.py: BlockTable/InputBatch 移除 kernel_block_sizes
```

## 性能声明

精度表(GSM8K / GPQA Diamond / acceptance length,无请求错误):GB200 GLM `FLASHINFER_MLA_SPARSE` MTP5 main 0.9280/0.7323/4.73 → PR 0.9333/0.7374/4.73;GB200 DSv4.1 DSpark5 0.9257/0.8384/4.08 → 0.9261/0.8460/4.01;H200 GLM `FLASHMLA_SPARSE`+`fp8_ds_mla` 0.9325/3.937 → 0.9348/3.930;MI350X GLM `ROCM_AITER_MLA_SPARSE` acceptance 3.33 → 4.82。prefix caching 中块命中输出与 main 逐字节相同;NIXL 1P1D GLM 0.9401 / DSv4.1 0.9265,0 failed transfers;Mooncake P/D GLM 0.9136。

性能(PR vs main,中位数,并发 1/16/64):H200 GLM sonnet 输出吞吐 +11.2%/−1.6%/+1.2%,H200 DSv4.1 4K/1K 吞吐 +6.2%/+1.9%/+0.5% 且 TPOT −14.6%(并发 1);GB200 GLM sonnet TTFT −21.2%(并发 1)/−24.8%(并发 64);MI350X GLM 4K/1K(random)吞吐 −8.6%/−9.9%/−9.2% 为最差项,TPOT +7.8%/+3.3%/+4.3%。PR 未注明 benchmark 的请求长度分布细节(仅标注 4K/1K、32K/1K、sonnet)。

## 验证与回归风险

新增 `test_glm5next_indexer_block_sizes.py`、`test_sparse_mla_packed_stride`(FlashInferTRTLLM 与 FlashMLA fp8_ds_mla 的 packed stride)、`test_deepgemm_paged_mqa_packed_manager_block_stride`;`test_kpool_indexer_block_sizes.py`、`test_kpool_tail_slot_mapping.py` 改用 `Glm5NextIndexerBackend`/`CircularBufferSpec`;多个 attention backend 测试改用 `supports_block_size()`。风险:这是 KV cache 基础设施的重构,影响所有 hybrid/MLA 模型、NIXL/Mooncake/HiSparse 等 KV consumer(它们用同样方式选 kernel block size)、CUDA graph 捕获路径与 DFlash/EAGLE draft 路径;`KpoolTailSpec` 移除对下游 spec 代码是 breaking rename;MI350X 一个 workload 出现约 −9% 吞吐回归(部分被 TPOT 改善抵消)。

## 关联

同区域:kv-cache-manager 系列与 spec decode KV 布局卡(pr-55688 ReplaySSM、pr-36164 slots);涉及 GLM-5.3-Flash/GLM5Next、DSv4.1、DFlash(pr-36847)等模型栈。PR 正文未列出其他前序/后续 PR 编号。
