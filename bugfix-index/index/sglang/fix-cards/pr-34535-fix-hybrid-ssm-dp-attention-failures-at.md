# PR #34535: [Fix] hybrid-SSM: DP attention failures at low concurrency

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34535](https://github.com/sgl-project/sglang/pull/34535) |
| 作者 | elvischenv |
| 合并日期 | 2026-10-19 |
| labels | apple-silicon, blackwell, bug, bypass-fail-fast, lora, run-ci |
| 热度 | 👍 0 · 💬 33 |
| 分类 | Attention / Kernel |
| bug 类型 | general |

## 一句话结论

hybrid-SSM 模型 DP attention 两处崩溃:max-running-requests 小于 DP 数时各 worker 得 0 槽、decode 被 relabel 成 extend 后 mamba 走错元数据;分别改为保底 1 槽与 logical_forward_mode。

## 问题现象

hybrid-SSM(Nemotron-H)部署在并发扫描下跑 DP attention(`--tp 4 --ep 4 --dp 4 --enable-dp-attention`,每并发一实例)出现两类失败:(1) `--max-running-requests` 小于 `attn_dp_size` 时每 worker 分得 0 槽,服务器无法启动,在混合模型上还伪装成误导性的 mamba 报错(`Hybrid (mamba/linear-attention) state cache is too small ... max_num_reqs=0`,尽管 mamba pool 有 348//5=69 个槽);(2) 并发 64 及以上,DP scheduler 在 mamba track 边界上崩溃:`AssertionError: conv-state tracking requires mamba_track_seqlens and extend_prefix_lens ...`。仅带修复 1 时,并发 64 起每个扫描点都有一台 DP scheduler 死于此(64 处 165/320 完成,512 处 2380/2560)。修后 1–512 并发全量完成,512 并发输出 3161.54 tok/s。

## 根因

正文披露两个独立根因:

1. 每 DP worker 份额 `max_running_requests // attn_dp_size` 向下取整为 0(如 `1 // 4`),请求池/mamba 池等随之配成 0。
2. hybrid-SSM 下 `DpPaddingMode.get_dp_padding_mode` 在任一 DP rank 空闲而另一 rank 在 extend 时强制 `MAX_LEN` 模式,`ForwardBatch.prepare_mlp_sync_batch` 随之把 decoding rank relabel 成 `EXTEND`(每行补成 1-token extend,让各 rank 对 collective 呈现同形)。行本身仍是 decode 行,但 mamba 元数据读标签为它们构建 prefill 元数据;当某行落在 `--mamba-track-interval` 边界上时,decode batch 设了 `mamba_track_mask` 而 `mamba_track_seqlens`(仅由 `prepare_for_extend` 填充)为 `None`,调度器崩溃。正文并按代码审查指出同一 relabel 还会让 `kimi_linear`/`kimi_k3` 的 KDA gate 预处理走 prefill 分支、`qwen3_5` 构建 prefill 版 strided GDN 视图喂给 decode kernel。

## 修复方案

1. **per-worker 份额保底 1**:`KVCacheConfigurator`(请求池、mamba 池含 disable_radix_cache 路径、spec 中间缓冲、GDN ReplaySSM ring)、`compute_swa_request_cap`、`DSV4PoolConfigurator`、PD decode in-transfer reserve(`pd_disaggregation_hook`)、LoRA MoE CUDA graph 缓冲、MLX resolver 等所有 `max_running_requests // attn_dp_size` 处统一改 `max(1, ...)`;除法仍取整,各消费方保持同一口径。
2. **新增 `ForwardBatch.logical_forward_mode` 属性**:读回 decode→extend relabel(仅当原始模式是 decode 且当前是 extend 时返回 `_original_forward_mode`)。需要 decode 语义的消费方改读它:mamba `Mamba2Metadata.prepare_mixed`、`MambaAttnBackendBase._forward_metadata` 与 `HybridLinearAttnBackend.forward` 的 linear 层派发、`kimi_linear`/`kimi_k3` KDA gate、`qwen3_5` GDN 布局。全注意力层仍按 1-token extend 跑,只有携带状态的侧被路由回 decode。另在 two-batch-overlap `filter_batch` 加断言明确不支持该 relabel。

新增三个 CPU 单测(`TestPdDecodeExtraSlots`、`TestPerWorkerRequestSplit`、`TestDpDecodeRelabel`),回归用例均在修复回退时失败。端到端在 4× GB300 上 Nemotron-3-Ultra-550B 验证,1–512 并发全部请求完成,GSM8K flexible-extract 0.9386。

## 影响与回归风险

- 回归面:所有启用 attention DP 时按 `max_running_requests // attn_dp_size` 定尺寸的池与缓冲(请求池、mamba/SWA 池、spec 缓冲、PD reserve、LoRA graph、MLX)——上限变松(至少 1 槽);以及 hybrid-SSM/线性注意力模型的 mamba 元数据构建、KDA gate 预处理与 GDN 输入布局的派发逻辑——被 DP relabel 的 decode 行为改回 decode 语义,真实 extend 不变。two-batch overlap 与该 relabel 组合现在显式报错(此前未定义)。

## 关联

- #41818(`--dp 4 --enable-dp-attention` 现拼作 `--attn-dp-size 4`)
- 崩溃 traceback 来自 `cae69be584` 运行(正文引用)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
