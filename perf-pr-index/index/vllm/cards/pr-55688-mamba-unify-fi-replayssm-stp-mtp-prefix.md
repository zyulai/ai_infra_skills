# PR #55688: [Mamba] Unify FI ReplaySSM STP/MTP/Prefix Caching Lifecycle

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @askliar |
| 合并 | 2026-11-04 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, ready, ci/build, cpu, mrv2, scheduler, kv-cache-manager |
| Diff | diffstat 缺失(pack 未提供 + additions/-deletions 与文件数);diff 有截断(40 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/55688 |

## 一句话结论

为 FlashInfer ReplaySSM(Mamba 线性注意力状态回放)补齐 `align` 模式的 prefix caching 并统一 STP/MTP 生命周期,GB300 上 MTP3 解码吞吐最高 2.20x。

## 动机

ReplaySSM 以 ring buffer 形式把最近 token 的 SSU 状态留在 Mamba page 内,用于在 MTP/STP 部分接受时回放状态;此前缺少 prefix caching 支持,且 V1/V2 两个 Model Runner、MTP 与 STP、prefix caching 各自的 ring tracker 生命周期逻辑分散,部分接受、preemption、请求重排时难以保证 live state 与已缓存前缀一致,checkpoint 发布与 block 释放的时序也有 page 对齐问题。

## 优化点

在 `align` 模式下为 FlashInfer ReplaySSM 增加 prefix caching(含 MTP),V1 与 MRv2 两个 runner 共用:replay ring 保持在 Mamba page 内部;发布 accepted boundary 时用 `replayssm_materialize` 只物化被接受的边界、不改写 live state 与 cached-prefix 源(测试名 `test_replayssm_materializes_only_accepted_boundary_with_compacted_rows`);lifecycle 工作(重置、checkpoint、行压缩)按 cache group 批量执行,跨 group 保持等价的逻辑回放历史;V1 的 accepted-token 计数留在 GPU 上,减少 D2H 同步;修复 Mamba page 对齐与"先发布 checkpoint 再释放 block"的顺序。实现上把原先散在 mixer 的 `update/commit/reset_replayssm_ring_trackers` 三组 host 侧 tracker 精简为 `ring_pos` + `write_pos` + `is_flush` 并移入 `ReplaySSMModelContext` 的 postprocess plan(`_postprocess_replayssm_kernel` + `_compact_replayssm_requests_kernel`)。

## 关键改动

```text
vllm/v1/worker/replayssm_utils.py: ReplaySSMModelContext 统一 V1/V2 postprocess/compact/materialize 生命周期
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py: 精简 ring tracker,统一 FlashInfer/Triton SSU backend 派发
vllm/v1/kv_cache_interface.py / vllm/v1/worker/utils.py: Mamba align 模式的 KV cache 分配与组配置
vllm/model_executor/warmup/replayssm_warmup.py: ReplaySSM kernel 预热
```

## 性能声明

PR 正文数据(Super 3,GB300,FlashInfer Python 0.7.2.post1,论文 workload,OFF/ON/ON/OFF 各两次取平均):

- ReplaySSM OFF → ON(TP1, V2, FP32 SSM state, CUDA graphs, 无 prefix caching):Natural MTP3 batch 512 输出 TPS 10,950 → 24,037(2.20x),batch 256 为 9,516 → 19,217(2.02x);Standard decoding batch 256 为 11,412 → 14,631(1.28x)。MTP3 batch 512 峰值并发解码 219 → 512。
- OFF → ON 且双方开 prefix caching:MTP3 batch 256 per-request decode TPS 60.21 → 91.12(+51.3%),总输出 TPS 8,747 → 18,043,0 → 256 请求命中前缀;Standard batch 256 43.88 → 54.56(+24.3%);MTP3 batch 512 per-request decode −6.1% 但总 TPS 10,266 → 20,384。
- Prefix caching OFF → ON(4x GB300 TP4, FP16 SSM state, concurrency 1,51,200 共享 token + 20 轮 2,048 in/1,024 out,MTP3 合成接受率 3.1151):E2E output TPS 304.76 → 518.88(+70.26%),TTFT 1,658.9ms → 249.8ms(−84.94%),每次 ON 运行记录 1,513,680 个 prefix-hit token。
- 与 main 对比(同 TP4 workload,ReplaySSM 开、prefix 关):STP decode TPS −0.05%,MTP decode TPS +1.86%,MTP E2E −0.54%(基本持平)。
- GSM8K(Super 3, 1,319 题, 5-shot, V2, TP2, natural MTP5):cache `none` 93.33%(accept 4.254)vs `align` 92.80%(accept 4.266,prefix-hit 861,972 token)。

## 验证与回归风险

71 个 component tests 通过 + 17 个 E2E test identities(FP16 TP2 MTP、prefix 复用、preemption、partial acceptance);CI 改为 `VLLM_TEST_REQUIRE_REPLAYSSM=1` 强制执行 ReplaySSM 用例,并新增 AMD mirror;`tests/kernels/mamba/test_ssu_dispatch.py`、`test_replayssm_plan.py` 大幅重写。风险:触及调度器、kv-cache-manager、V1/V2 worker 共享路径;ring tracker 结构变更影响所有 FlashInfer ReplaySSM 用户;`MambaStateShapeCalculator.append_replayssm_ring` 的 ring 长度计算与 spec 形状变化可能影响其他 Mamba 模型;GPU 执行在 hosted CI 仍 deferred。

## 关联

同区域:Mamba/ReplaySSM 相关卡(pr-39822 warmup、pr-54163 mamba align context recompute)、KV cache manager 系列(pr-57169);文档指向 automatic_prefix_caching.md 的 FlashInfer ReplaySSM 小节。
