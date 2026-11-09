# PR #46205: [KV Offload] Support packed HMA KV cache layout

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-06-20 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | ready, v1, kv-connector |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 6 个文件(kv_cache_utils、offloading worker、CPU offload spec/manager、envs、测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/46205 |

## 一句话结论

把 DeepSeek V4 式 packed KV 布局推广到多 group HMA 模型(gpt-oss/Gemma),CPU offload 注册从每层切片 12 个 CPU 张量减到 1 个,full-row ref 路径快约 13.5%。

## 动机

多 group HMA(hybrid/多头异构注意力)模型如 gpt-oss、Gemma 3/4 有多个 KV cache group;此前只有 DeepSeek V4 走 packed per-block 布局,其他多 group 模型按默认布局分配,offload 时需对每个 layer 切片分别注册 CPU 张量,产生大量小张量与每层引用,GPU-CPU 拷贝与注册开销随 group 数增长。

## 优化点

复用并泛化 DeepSeek V4 的 packed 布局,机制分三层。第一,布局规划:`_get_kv_cache_config_deepseek_v4` 重命名为通用的 `_get_kv_cache_config_packed`,新增 `_use_packed_kv_cache_groups` 判定(全部 group 为 `UniformTypeKVCacheSpecs` 即 DSV4,或 `VLLM_USE_PACKED_HMA_KV_CACHE=1` 且 group 数 > 1);packed 布局按 page size 分桶,每个 (slot_idx, page_size) 发一个 `KVCacheTensor`,带 `offset` 与跨 group 的 `block_stride`,所有层别名同一个物理 backing,块数据连续排布。第二,offload 注册:`OffloadingConnector` worker 检测带 `block_stride` 的 packed tensor(非 DSV4),用 `as_strided((num_blocks, block_stride))` 视图把它注册为一个 canonical backing tensor,每个 KV group 只持一个 full-row ref(`CanonicalKVCacheRef(0, block_stride)`),保留 packed 拓扑。第三,CPU 侧适配:CPU offload spec 与 simple_kv_offload manager 在 packed 时用 `tensors[0].size` 计算总字节(避免把别名张量重复求和),CPU 配置镜像 `offset`/`block_stride`。DSV4 路径行为不变,新路径由 env flag opt-in。

## 关键改动

```text
vllm/v1/core/kv_cache_utils.py: _get_kv_cache_config_packed(原 DSV4 专用泛化);_use_packed_kv_cache_groups 判定
vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py: packed HMA 注册为单一 backing tensor + 每 group 一个 full-row ref
vllm/v1/kv_offload/cpu/spec.py、vllm/v1/simple_kv_offload/manager.py: packed 张量求和与 CPU 配置镜像适配
vllm/envs.py: 新增 VLLM_USE_PACKED_HMA_KV_CACHE(默认 False)
tests/v1/core/test_contiguous_kv_packing.py: 默认/packed 两种 backing 布局的断言
```

packed 注册(节选):

```python
packed_tensor = tensor.as_strided(
    (num_blocks, block_stride),
    (block_stride, 1),
    storage_offset=0,
)
self._register_handlers(
    CanonicalKVCaches(
        [CanonicalKVCacheTensor(packed_tensor, block_stride)],
        [[CanonicalKVCacheRef(0, block_stride)]
         for _ in kv_cache_config.kv_cache_groups],
    )
)
```

## 性能声明

PR 正文 benchmark:`openai/gpt-oss-20b`,B300,128K context,`OffloadingConnector`,2 次 CPU-hit 迭代:packed HMA full-row refs 使用 1 个 CPU 张量、平均约 124.95 ms,对比 per-slice 注册的 12 个 CPU 张量、约 144.37 ms,约 ~13.5% faster;`google/gemma-3-1b-it`,4K:1 个 CPU 张量对 4 个,CPU-hit 延迟基本持平(~12.23 ms 对 ~12.32 ms)。测试条件:PR 未注明并发、block 大小与 workload 细节。

## 验证与回归风险

测试:`pytest tests/v1/core/test_contiguous_kv_packing.py tests/v1/simple_kv_offload/test_scheduler.py tests/v1/kv_connector/unit/offloading_connector/test_worker.py tests/v1/kv_offload/cpu/test_gpu_worker.py -q`;ruff/mypy/SPDX 等 hook 通过。新增测试覆盖三态:flag 关闭时 HMA group 保持默认 backing(2 个共享张量)、flag 开启时切到 packed backing(单张量 + offset/block_stride)、单 group 保持 unpacked(block_stride=0)。回归风险:默认关闭(`VLLM_USE_PACKED_HMA_KV_CACHE=False`),DSV4 路径由同名别名保持行为;packed 布局下 offload 组件依赖 "所有 tensor 同时带或不带 block_stride" 的断言,混合布局会 assert 失败;`as_strided` 视图与物理 backing 的别名关系要求 offload 侧不得独立改写共享区域。

## 关联

基于已合并的 #44577(DeepSeek V4 packed 布局);#37885 的 flagged 窄替代(gpt-oss/Gemma 式 HMA packing)、#44865 相关;正文 duplicate check 未发现重复工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
