# PR #28451: [GDN][KDA] ReplaySSM buffered output-only decode for Linear Attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-06-26 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, run-ci-extra, linear-attention, release-highlight |
| Diff | pack 未提供 diffstat(diff 正文可用,另有 13 个文件因过长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/28451 |

## 一句话结论

为线性注意力 decode 引入 ReplaySSM 环形缓冲,把每步状态流量从读+写砍到约 0.53x,kernel 最高 1.5x、E2E TPOT -2.3%。

## 动机

GDN decode kernel 每步都从 HBM 读全量递归状态 `S [HV,V,K]` 并写回。在 H20-3e 上 batch>=64 时该 packed decode 有 70-79% 时间卡在 HBM 带宽上,每步的状态读+写流量(~8dn)是主导项(issues/28511)。ReplaySSM(Dao AI Lab, 2026)的思路是去掉每步状态写:每槽只保留最近 L 步的 `(d,k,g)` 环形缓冲,输出经历史重建读取,全量状态每 L 步才 flush 一次,每步状态流量降为只读(~4dn)。

## 优化点

新增 `fused_recurrent_linear_replayssm_decode` Triton kernel,`IS_KDA` constexpr 让同一 kernel 覆盖 GDN(per-head 标量门)与 KDA(逐 K 通道门)两种门控粒度;非 flush 步在 K-tile 内由检查点 `S_0` + 环形缓冲重建 `S = A·S_0 + Σ w_j·d_j·k_j^T`(总衰减 A 与各位置回放衰减 w 由缓存门 g 经 cumsum 构造),每个 tile 立即与 q/k 收缩成标量,完整 `[V,K]` 状态从不落到 HBM。重建的 `tl.dot` 强制走 tensor core(fp32→TF32、bf16→bf16 TC)——作者实测强制 IEEE fp32 dot 会让 kernel 慢 10-20x,把 1.5x 的收益变成 15x 的回退,是主要 perf trap。工程侧:`write_pos` 是 MambaPool 中按物理槽维护的 decode 位置计数(不能由 `seq_len % L` 推导,否则首个 decode 步会读到非空零环),槽分配与 COW 时归零、每 forward 推进一次;CUDA graph 下 `write_pos` 是图外每 replay 原地刷新的静态 per-bs buffer;`force_flush` 采用与 radix prefix cache 快照完全相同的条件(`seq_lens_cpu % mamba_track_interval == 0`、同一个 post-increment 的 `seq_lens_cpu`),保证环恰在快照读取的那步折入 `temporal[slot]`。flag 关闭时不分配环形张量、路径逐字节不变。KDA 暂未接 radix 协调(仅自然 L-1 环绕 flush)。

## 关键改动

```text
python/sglang/srt/layers/attention/fla/fused_recurrent_linear_replayssm.py: IS_KDA 参数化的输出-only decode kernel
python/sglang/srt/configs/mamba_utils.py: is_kda 属性与 num_k_heads_per_tp(环形 g_cache/k_cache 布局)
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py: flag 下的 decode dispatch
(其余 13 文件:MambaPool 环、radix force_flush、CUDA graph 集成等未收录)
```

## 性能声明

Roofline:H20-3e 上 packed GDN decode 在 batch>=64 时 70-79% HBM 带宽受限;ReplaySSM 把状态流量降到约 0.53x,高 batch 下 kernel 上限约 1.7x。Kernel microbench(随 PR 附 `bench_gdn_replayssm_decode.py`,L=8):batch>=16 时 1.2-1.5x 于 packed decode;batch 1 略慢(overhead-bound,非工作点)。E2E:Qwen3.5-35B-A3B,128 并发,256/512 in/out,TPOT 改善约 2.3%;作者说明端到端收益与模型相关——该模型 MoE 占比高、GDN 线性层只占 decode 的一小部分,GDN-dense 模型收益会更大。精度:GSM8K(200 题)flag-on 0.880 vs baseline 0.860,噪声内无回归(早期 GDN-only 集成片在 radix+CUDA graph 全开下为 0.900 vs 0.850);贪心解码与 baseline 逐字节一致直到第一个近平局 token,其后差异为良性 TF32 位翻转。

## 验证与回归风险

单元测试 `test_linear_replayssm_decode.py`:GDN fp32 在 L=1 时 bit-exact(packed decode 作为代数等价的正确性 oracle),L>1 用 tensor-core 现实容差(TF32 ~4e-4 / bf16 ~1e-3),含 buffer 中途强制 flush;KDA 对 `fused_recurrent_kda_packed_decode` 同容差通过。风险面:GDN 路径逐字节等于原 kernel,flag-off 路径零改动;已知缺口(KV-pool 预算不含 ring 字节,ring 大时可能 OOM,需调低 `--mem-fraction-static`);要求 `--linear-attn-decode-backend triton` 且 `mamba_scheduler_strategy=no_buffer`,`extra_buffer`/ping-pong 捐赠路径未覆盖(被守卫排除);KDA 无 radix 协调,长 decode 中途 prefix 复用语义依赖后续 PR;kernel 从 vLLM `fused_recurrent_replayssm.py` 移植,保持上游 diff 友好。

## 关联

Addresses #28511;算法来源 ReplaySSM(Dao AI Lab, 2026),移植自 vLLM 同名实现。同区域:GDN/KDA 线性注意力系列——#19775(GDN verify 融合)、#39273(KDA fused verify)、#24584(amd/deepseek_v4 attn early exit 属 NSA/DSA 线)。KDA radix 协调与 KV-pool ring 计费为声明中的 follow-up。
