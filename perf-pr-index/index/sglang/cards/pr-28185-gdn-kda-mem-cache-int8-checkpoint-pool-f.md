# PR #28185: [GDN][KDA][mem_cache] int8 checkpoint pool for the linear-attn prefix cache

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-06-18 |
| 分类 | Quantization(major;作用于 mem_cache/radix 前缀缓存) |
| Label | run-ci, run-ci-extra, linear-attention, release-highlight |
| Diff | 素材未提供 diffstat(diff 含 8 个文件,另有 3 个文件因体积省略) |
| 链接 | https://github.com/sgl-project/sglang/pull/28185 |

## 一句话结论

线性注意力前缀缓存改存 int8 checkpoint:同显存下可缓存前缀 ~2 倍,复用坍缩点 256→512,命中带宽内 probe 延迟 -26%、吞吐 +33%。

## 动机

KDA、GDN/Gated DeltaNet 等线性注意力模型每序列维护一个紧凑 recurrent state 而非增长的 KV cache;SGLang 已在 radix 树(`MambaRadixCache`)缓存这些 state 以便命中时恢复。瓶颈是容量:active mamba 池为每个缓存前缀存一个 bf16 state,池大小按 running set 而非缓存规模设计,且与长度无关(每条缓存序列一个完整 state)。当不同缓存前缀数超过池容量时,prefix reuse 坍缩——KV token 仍在,但 state 被逐出并重算。这对"大量不同短/中前缀"workload(多文档 RAG、大量短会话、agentic fan-out)是主要成本。

## 优化点

新增可选的专用 int8 checkpoint 池(`--enable-int8-mamba-checkpoint`,默认关):radix 缓存的 SSM temporal state 以 int8 存储,带 per-`(head, k-channel)` 对称 scale(`Int8CheckpointStore`);conv1d 窗口保持 bf16(很小)。每缓存 state 约 1 字节/元素(原 2 字节),同显存下池可容纳 ~2 倍不同前缀。选 int8 而非 fp8:recurrent state 近均匀分布(rank-1 更新的累加和),fp8-e4m3 的指数位浪费;1 字节下 int8-per-channel 误差 ≈0.5% vs fp8 ≈0.7%。质量上构造性安全:store 时量化一次、命中时反量化一次,state 不再进入 recurrence 的 quant→dequant 循环,无误差累积;gated-delta rule 的收缩性(每通道衰减 α<1)还会几何衰减注入误差。池分离的原因:recurrence kernel 需要 bf16 工作内存,故保留小 bf16 active 池(kernel 面,按 running set 大小)+ 独立 int8 checkpoint 池(radix 缓存);donate 时量化 active→int8,命中时 copy-on-write 反量化 int8→active。量化 hook 位于两种调度策略(no_buffer 拷贝 / extra_buffer ping-pong)的共同收敛点,策略无关且兼容未来 spec 路径。配套改动了 scheduler 的两池不变量检查与 throttle 统计(int8 模式下 radix 在 active 池占 0 slot)。

## 关键改动

```text
python/sglang/srt/mem_cache/mamba_checkpoint_pool.py: 新增 Int8CheckpointStore
  (int8 qdata [L,slots,H,d_v,d_k] + per-(head,k) scale)与 MambaCheckpointPool
python/sglang/srt/mem_cache/mamba_radix_cache.py:
  _commit_int8_checkpoint donate hook、COW 命中反量化、int8 slot 生命周期
python/sglang/srt/mem_cache/memory_pool.py: _init_mamba_pool 挂载可选 ckpt 池
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:
  COW 执行改走 ckpt_pool.load_to_active
python/sglang/srt/managers/scheduler_components/invariant_checker.py /
  pool_stats_observer.py: 两池不变量与 mamba 统计
python/sglang/srt/server_args.py: --enable-int8-mamba-checkpoint、--int8-mamba-ckpt-size
benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py: 复用容量 A/B benchmark
```

量化数学(节选):

```python
state_fp32 = state.to(torch.float32)
amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
scale = (amax / cls.QMAX).to(state.dtype)   # QMAX = 127,scale 轴为 d_v
q = torch.round(state_fp32 / scale.to(torch.float32)).clamp(-cls.QMAX, cls.QMAX).to(torch.int8)
```

## 性能声明

Qwen3.5-35B-A3B(GDN-hybrid),4x H20-3e,`extra_buffer`,`--max-mamba-cache-size 256`(active 256 / int8 512),前缀约 1000 tokens,parallel 8,committed benchmark `bench_int8_checkpoint_reuse.py`:

- K=128(两者都放得下):reuse 0.940→0.940,probe p50 149.9→152.4ms,rps 46.9→50.6
- K=384(bf16 已坍缩):reuse 0.020→0.940,probe p50 204.1→150.5ms,rps 36.9→49.0;在 256<K≤512 区间,int8 提供约 47 倍前缀复用、约 26% 更低 probe-prefill 延迟、约 33% 更高吞吐
- K=640(超出两池):0.000→0.048,双双坍缩;复用坍缩拐点 256→512(约 2x)
- 显存(同缓存量另一种花法):bf16 单池 1280 → mamba HBM 19.2GB、max_total_num_tokens 8.32M;int8(256 active + 1024 ckpt)→ 8.0GB、9.93M;已知 caveat:int8 池显存尚未计入 KV 预算(follow-up),靠 `--mem-fraction-static` 余量承载并有启动检查

## 验证与回归风险

三层精度证据:(1) 机制——fp32 gated-delta recurrence 实测注入误差饱和不随深度增长(1 个 checkpoint 约 6.6e-3 → 32 个链式约 1.3e-2 → 至 128 平坦);(2) logit 级——Qwen3.5-35B-A3B TP1 greedy 确定性下,命中反量化时 next-token 分布 KL 约 6e-3(bf16 复用基线本身 KL 约 5e-4),greedy top-1 全部不变;(3) 任务级——GSM8K 200 题 5-shot greedy,dequant 实测触发 599 次:WARM int8 0.832 [0.830-0.835] 落在 bf16 对照 0.842 [0.815-0.855] 带内,两者 WARM 均高于 COLD 0.805(复用路径在 overlap 调度下本身非确定)。方法论注意:int8 只有在 `--mamba-scheduler-strategy extra_buffer` 下才真正触发(no_buffer 在 divergent-suffix workload 上几乎不反量化,50 题仅 2 次 vs extra_buffer 599 次)。正确性:两种调度策略的 store/match/COW 验证通过;`server_args` 拒绝与 hierarchical cache(HiMambaRadixCache)及自定义 radix 后端组合。CI:Base 与 Extra 均通过。默认关闭时 bf16 基线路径逐字节不变。

## 关联

与 host-offload 的 `HiMambaRadixCache` 组合被显式拒绝(int8 不感知,文档注明 int8 密度未来也可减半 host-offload);KV 预算计入 int8 池显存为声明的 follow-up。同分类(Quantization)与同区域(linear-attention cache)其他 PR 编号素材未列出,写"无"。
