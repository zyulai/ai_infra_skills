# PR #50842: [Feat] kimi-k3: DWDP for KDA and shared-expert weights on PD prefill

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @dmtri35 |
| 合并 | 2026-11-18 |
| 分类 | MoE (tier: major) |
| Label | documentation, blackwell, jit-kernel, release-highlight |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 摘录含 11 个文件、其中 5 个新文件,另有 3 个文件因篇幅略去) |
| 链接 | https://github.com/sgl-project/sglang/pull/50842 |

## 一句话结论

Kimi-K3 PD prefill 在 attention DP 下引入 `--kda-dwdp-size N`(DWDP):每个 attention-DP rank 只保留 1/N 的 KDA [q,k,v,g] in-projection 与 o_proj(以及被复制的 shared experts)权重,层运行前用 copy engine 经 NVLink 把整层权重拉进轮转槽,使原本 OOM 的 ADP8 在 8 x GB300 上放得下,以单请求 TTFT 换取 12-22% 批量吞吐与 3.9x KV 容量。

## 动机

在 attention DP 下,每个 Kimi-K3 PD prefill rank 都完整持有每个 KDA 层的 [q,k,v,g] in-projection 与 o_proj(attention TP 1 时每 rank 60.8 GB BF16),外加 EP all-to-all 后端下 24 GB 的复制 shared experts;ADP8 在 8 x GB300 上几乎放不下。全量 Kimi-K3 上 plain ADP8 直接在 weight post-processing 阶段 OOM,无法加载,DWDP 使其放得下。

## 优化点

1. 头对齐布局:rank r 恰好存 tensor-parallel rank r 的分片(每组 q/k/v/g 头的 1/N 行与匹配的 o_proj 列),N == TP 时与 KDA TP rank r 的权重字节一致;N 只需是能切分头的 attention-DP size 的因子。
2. 双槽轮转 gather:copy-engine peer 从映射到组内每个 rank 的 cuMem shard 拉取(每个 in-proj block 一次拷贝、每个 o_proj 列切片一次 cuMemcpy2DAsync),不用 SM、不用集合通信、不留副本;槽位即复制布局,prefill 输出与复制权重 bitwise 一致。
3. 严格门控(handle_kda_dwdp):仅 `--disaggregation-mode prefill`;拒绝 PP、attention CP、full CUDA graphs、跳过 load_weights 的 loader、weight cache、量化分片权重、第二个 K3 模型、在线权重更新与非 Kimi-K3 模型。
4. 仅支持未量化权重(UnquantizedLinearMethod);gather hook 用 eager_on_graph 在 CUDA graph 段之间发出,并把 gather 的 NVLink 流量避开 MoE all-to-all(wait_issued)。

## 关键改动

```text
python/sglang/srt/arg_groups/fields/parallel.py(新参数):
+    kda_dwdp_size: A[
+        Optional[int],
+        "DWDP for Kimi-K3 KDA projection and shared-expert weights on PD prefill "
+        "... Saves ~(1 - 1/N) of those weights per rank (about 53 GB for K3 at "
+        "attention TP 1, N=8) for one layer-weight gather per sharded layer per forward.",
+    ] = None
python/sglang/srt/layers/communication/kda_dwdp/manager.py(新文件,双槽轮转):
+NUM_SLOTS = 2
+    def on_layer_start(self, layer_idx: int) -> None:
+        if self._running is not None:
+            self._prefetch.release(self._running)
+        ...
+        self._prefetch.wait(k)
+        self._running = k
+        nxt = (k + 1) % len(self.layers)
python/sglang/srt/layers/communication/kda_dwdp/store.py(新文件,VMM shard + copy engine):
+            args = drv.CUDA_MEMCPY2D()  # Copy-engine strided copy; a torch copy into the strided view would launch a kernel.
python/sglang/srt/layers/communication/kda_dwdp/layout.py / prefetch.py(新文件):分片布局与 copy stream 上的 SlotPrefetcher(issue/wait/release,事件同步)。
python/sglang/srt/arg_groups/parallel_hook.py:handle_kda_dwdp 门控注册进 resolution pipeline。
python/sglang/srt/models/kimi_k3.py:KDA 层 bind_pair(fused_qkvg_proj/o_proj)与 shared experts bind_pair(gate_up_proj/down_proj)。
```

## 性能声明

(数字全部照抄 PR 正文,未独立核实。)

Kimi-K3 截断到 40 层,GB300:4,EP4 MegaMoE with #50838,aggregated prefill proxy:

| Input | ADP4 TTFT / batch-4 tok/s | ADP4 + DWDP4 | TP4 (reference) |
|---|---|---|---|
| 1K | 0.117 s / 17.4K | 0.131 s / 15.9K | 0.119 s / 17.3K |
| 8K | 0.336 s / 44.7K | 0.341 s / 44.2K | 0.151 s / 58.3K |
| 32K | 1.449 s / 59.1K | 1.471 s / 58.3K | 0.529 s / 61.6K |

DWDP4 weights 171.6 vs 192.9 GB per rank(−21.3 GB,其中 6.5 GB 为 shared experts);KV 6.35M → 10.32M tokens(+62%)。

全量 Kimi-K3,2x GB300:4,EP8 MegaMoE:plain ADP8 无法加载(OOM in weight post-processing),DWDP 使其放得下;ADP8 + DWDP8 在 32K 单请求 TTFT 3.60 s / batch-8 50.5K tok/s(KV 2.14M/rank)。Kimi-K3.1(2x GB300:4, EP8 MegaMoE)对比 prefill context parallel(TP8 + `--enable-prefill-cp` 8):128K 时 15.3 s vs 3.25 s;32K 时 3.77 s / 49.4K vs 0.818 s / 40.5K。正文总结:"DWDP trades single-request TTFT (one ADP rank prefills a lone request) for 12-22% more batch throughput and 3.9x the KV."。gather 在 8K+ 时被隐藏(DWDP4 与 ADP4 差距在 1-2% 内);~1K 时 forward 短于 gather(全量 K3 DWDP8:约 74 GB per rank per forward,约 55 ms 的 MoE-stage 等待),短 prompt 为此买单。内存:DWDP8 每 rank 释放 45.9 GB(shard 7.60 GB + 2 slots 1.76 GB vs 60.8 GB 复制)。

## 验证与回归风险

Fixtures:ADP4 vs ADP4 + DWDP4 的 prefill 输出(首 token id 与 logprob)20/20 bitwise;40 层 1P1D NIXL(Prefill ADP4 + DWDP4,decode ADP4 + KDA TP)eager 与 late staging 各 40/40、eager vs late 20/20 bitwise;头对齐 vs flat 布局:TTFT 8K 0.329 vs 0.330 s、32K 1.41 vs 1.42 s,4-request 32K tok/s 44.1-44.3K vs 44.0K,slots 每 rank +0.44 GB。测试:本地 8 层 Kimi-K3 1P1D NIXL PD 测试(GB300,prefill `--kda-dwdp-size 2`):prefill 首 token 与 logprob 在 100 至 16K tokens 及并发 burst 下与复制运行完全一致;nightly 版本测试将随后续 PR 到来(CI 容量原因)。CI 状态:Base/Extra/AMD 均 Skipped(未加 run-ci)。回归风险:(1) 门控面很宽但全部快速失败,量化权重、weight cache、full CUDA graph 等组合直接报错;(2) 短 prompt(<8K)与单请求 TTFT 变差;(3) 仅未量化权重,且与 HIP in-proj 融合(_merge_kda_inproj_weights_hip)互斥(设 _kda_dwdp 后返回 False);(4) VMM/FABRIC peer 映射、2 槽轮转与奇数层数 wrap 逻辑的边界正确性。

## 关联

同模型/同区域:PR #50838(EP MegaMoE,本 PR 测试基线)、PR #38344(Kimi-K3 prefill CP,本 PR 的对比项 --enable-prefill-cp)、PR #48745(KDA safe-gate 数值修复,均在本批次)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
