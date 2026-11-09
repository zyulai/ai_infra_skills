# PR #37771: [HiSparse] Add speculative decoding support with multi-step KV cache swapping

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @huangtingwei9988 |
| 合并 | 2026-10-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | hicache, run-ci, jit-kernel, run-ci-extra, release-highlight, hisparse, memory-pool, bypass-fail-fast, highest-priority |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长,素材保留部分源码文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/37771 |

## 一句话结论

让 HiSparse 稀疏 KV 支持多步投机解码(EAGLE/MTP):target-verify 阶段按 top-k 索引做一次性多步 swap-in,避免退化为逐步 LRU 换页。

## 动机

HiSparse 此前与投机解码互斥(`validate_hisparse_feature_combos` 直接 raise "does not support speculative decoding yet"):draft 池没有 HiSparse 映射、混合 batch 无法合并 HiSparse decode batch。若不做多步整合,EAGLE 多步验证只能退回 sequential per-step LRU swapping,交换开销被 draft 步数放大。kernel 侧的多步 gather/commit swap kernel 已在 #32162 完成,本 PR 把它接入 serving 路径。

## 优化点

核心机制:(1) HiSparse coordinator 新增 `HiSparseSpecState` 与 `swap_in_selected_pages`,target-verify forward 时把 grouped top-k indices(num_reqs × num_steps)交给 coordinator,用 `load_cache_to_device_buffer_spec_mla` 一次换入验证所需的页,内含跨步 miss 去重、extra-page lookup、scratch 管理与 cache commit;(2) draft KV 保留在自己的逻辑 GPU 地址空间,不占用 HiSparse target KV 的物理映射,target KV 仍走 host/device 放置;(3) Mooncake PD 迁移为 draft 条目携带独立的 device-only 逻辑页索引,target host/physical 页索引分开传,PP 切层时把 draft 尾部条目排除在 MLA entry span 之外;(4) decode 侧准入预算改为 `padded_buffer_size + spec_swap.scratch_capacity` 每 req。约束:仅支持线性投机(`--speculative-eagle-topk 1`),不支持 `--speculative-adaptive`(swap state 静态定长)。

## 关键改动

```text
python/sglang/srt/managers/hisparse_coordinator.py: HiSparseSpecState + swap_in_selected_pages / spec_mla load kernel 接入
python/sglang/srt/layers/attention/dsa_backend.py: target_verify 分支用 grouped top-k 换页;indexer metadata 覆盖 verify 模式
python/sglang/srt/arg_groups/hisparse_hook.py: 解除 spec 禁用,加线性投机/adaptive 校验
python/sglang/srt/disaggregation/mooncake/conn.py + common/conn.py: draft 条目 device-only 页索引随 PD 迁移
python/sglang/srt/disaggregation/decode.py: 准入预算计入 spec scratch
```

## 性能声明

无速度 benchmark(Speed Tests 章节为空)。仅 Accuracy A/B(GLM-5.2-W4AFP8,1P1D TP8/DP8 DP Attention,FP8 KV cache,EAGLE 3 步 top-k 1 共 4 draft tokens,CUDA Graph 开启,HiSparse top-k 2048、4096-slot device buffer、host/device 比 4;唯一变量是 HiSparse 开关):

- GSM8K 20-shot(200 样本):ON 195/200(97.50%)vs OFF 192/200(96.00%),+1.50 pp
- GPQA-Diamond(198 样本):ON 180/198(90.91%)vs OFF 182/198(91.92%),-1.01 pp

## 验证与回归风险

上述 CPU 侧 accuracy 测试为 PR 自带验证;CI Extra 与 AMD ROCm 10 为红。风险面:DSA/DeepSeek 稀疏 attention 的 target-verify 路径行为变化;PD 迁移在 draft 条目存在时的 Mooncake-only 限制(非 Mooncake backend 显式 NotImplementedError);混合 chunk 仍不支持;spec swap state 静态定长,动态 draft 形状(adaptive)被校验拒绝;PP + draft 条目的层切分逻辑(`_mla_kv_entry_span_with_pp` 减去 num_draft)对非 spec 路径 num_draft=0 保持原行为。

## 关联

前序:#32162(multi-step gather/commit swap kernel 实现)。同区域:PR #24880(HiSparse DSV4 P2D DRAM)、HiSparse 相关 hisparse/hicache 系列 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
