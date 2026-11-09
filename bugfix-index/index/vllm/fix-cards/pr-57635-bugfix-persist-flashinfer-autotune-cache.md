# PR #57635: [Bugfix] Persist FlashInfer autotune cache per rank to fix the rank-0-only cache-hit deadlock

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#57635](https://github.com/vllm-project/vllm/pull/57635) |
| 作者 | MadhavJivrajani |
| 合并日期 | 2026-10-06 |
| labels | bug, nvidia, ready |
| 热度 | 👍 0 · 💬 23 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
autotune 缓存只存 rank0 且广播全 rank,FlashInfer 按 rank 键控 MoE 条目致其余 rank 卡 reduce 死锁;改为每 rank 各存各载。

## 问题现象
8xH100 (SM90)、TP8+EP8 FP8 MoE,`vllm/vllm-openai:nightly-8a728663`(v0.28.1rc1.dev388)、FlashInfer 0.6.18、`$HOME` 为 pod 生命周期 emptyDir:引擎因无关 GPU 故障死亡后 kubelet 原地重启 18 次,每次重启 8 个 rank 都加载同一 `autotune_configs.json`,rank0 报 `Config cache hit for trtllm::fused_moe::gemm1`(及 gemm2),rank1-7 停在 `Tuning trtllm::fused_moe::gemm1: k/21 [30:00<...]`,EngineCore 每分钟报 `No available shared memory broadcast block found in 60 seconds`,30 分钟后 `torch.distributed.barrier` 超时,API server 以 `Engine core initialization failed` 退出。期间 GPU0 100%、GPU1-7 0%。把 `flashinfer_autotune_cache/` 目录改名后重启:autotune 1s(`Saved 42 configs`)、init 20s、READY——19 次相同失败 + 1 次成功,单变量对照。

## 根因
正文披露:`flashinfer_autotune` 只持久化 rank0 的 autotune cache,下次引擎启动把该文件广播给所有 rank;FlashInfer 按 `tp_rank`/`ep_rank`(`MoERunner.get_cache_key_extras`)键控持久化 MoE 条目,只有 rank0 的查找能命中;命中会跳过同步 tuning pass 的 per-tactic reduce,ranks 1..N-1 阻塞在 reduce 里,rank0 在尾部 barrier 自旋直到 30 分钟超时。任何第二次见到该文件的引擎启动都会触发(容器重启 + pod 生命周期卷,或持久 `VLLM_CACHE_ROOT`)。stale 与 fresh 缓存文件 42 个键相同,但 stale 是 rank0 的,其键只匹配 `ep_rank=0`。

## 修复方案
`vllm/model_executor/warmup/kernel_warmup.py`:每个 rank 保存并加载自己的文件(`autotune_configs_dp{dp}_rank{r}.json`);用一次 `all_gather_object`(CPU group)取代原 broadcast+barrier(不新增 collective):`_all_ranks_have_matching_cache` 按 FlashInfer `_metadata` + 条目数指纹判定「每个 rank 都有相互一致的文件」才加载;加载后再 gather 确认全部成功,失败则 `clear_cache()` 重 tune;仅 `tuner._dirty` 时保存。旧单文件缓存按文件名忽略,下次启动重新 prime。

## 影响与回归风险
- 影响窗口:正文未给出受影响提交范围(field report 版本 v0.28.1rc1.dev388;任何会第二次读到缓存文件的启动都触发)。
- 回归面:FlashInfer autotune warmup 的缓存文件命名(PP 阶段旧 `_tp_...` 命名改为 per-rank 命名,旧文件不再被读)与启动期 collective 序列;文件不匹配/缺失/加载失败时全员重 tune(冷启动变慢,正确性优先)。补丁 build 在受影响部署上的 live validation 尚未进行(正文如实说明)。

## 关联
- Fixes [#57423](https://github.com/vllm-project/vllm/issues/57423)
- #57579(替代方案:所有 rank 加载与 leader 相同的文件;正文说明本死锁源于非 leader rank 要在缓存里看到自己的条目,该方案不足)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
