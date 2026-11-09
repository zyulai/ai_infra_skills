# PR #39273: [AMD] [GLM-5.3-Flash] Enable FP8 and MXFP4 serving on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hdt98 |
| 合并 | 2026-10-01 |
| 分类 | Quantization(major) |
| Label | documentation, quant, amd, deepseek, speculative-decoding, run-ci, apple-silicon, jit-kernel, run-ci-extra, unified-radix-cache, memory-pool |
| Diff | pack 未提供 diffstat(PR 正文注明:26 个生产文件 +361/-80,14 个测试文件 +563/-21,3 个文档 +179/-26) |
| 链接 | https://github.com/sgl-project/sglang/pull/39273 |

## 一句话结论

在 MI355X(gfx950)上全速、正确地服务 GLM-5.3-Flash FP8/MXFP4,AgentX 回放对 main +14.3% geomean 吞吐。

## 动机

目标是让官方 `zai-org/GLM-5.3-Flash` FP8 checkpoint 与 `OneNexus/GLM-5.3-Flash-MXFP4` 在 MI355X 上以"零 RoPE 稀疏注意力 + FP8 KV cache + graph 化 EAGLE 5/1/6"的完整栈正确、高速运行。`main` 已合入十个 Day-0 enablement PR(#38545、#38546、#38547、#39317、#39338、#39339、#39340、#39341、#39778、#39779),本 PR 补齐剩余的正确性修复与优化。原 PR 中的四个 CI/测试卫生修复移至 #40721。

## 优化点

主体是一批 correctness 修复,穿插两个实测优化:(1) EAGLE draft on ROCm——ROCm 启用 rejection sampling(#37134)后,verify 对 all-greedy batch 走 argmax、从不读 draft 分布,于是 draft 跳过全词表 softmax、Gumbel 采样与 probs stack,并捕获配套的 greedy graph variant(`eagle_worker_v2.py` + draft graph runner 按 greedy/非 greedy 两个 variant 捕获/回放);(2) KDA fused verify(HIP)——在 #39524 基础上,用 `v_mul_f32` 内联汇编单独舍入 conv 乘积(与 gfx950 参考实现 bit-exact,避免 LLVM 合并为 FMA 改变结果)、默认单 wave,gfx950 实测 GLM shape(T∈{6,8},H=HV=16,K=V=128,batch 1-16,bf16 QKV/fp32 conv 权重)下用 BV16 宽 tile 把 grid 从 32 tile 缩到 8 tile;(3) vision 2x2 downsample——非重叠 merge 等价于 unfold+linear,改用 `Conv2dLayer` 避开 MIOpen 的逐 shape 卷积搜索。正确性修复包括:K-pool top-k 阈值 bin 超 4096 候选时以 (score,index) 唯一 key 重扫防裁剪、logits 行预算计入 AITER 256 列 padding 防 2 GiB 越界、ragged >4096 行分块 top-k、attention backend 解包 hybrid KDA 找到 full-attention 子 backend、Mamba ping-pong 槽的可见前缀归属与 `mamba_prev_track_seqlen` 簿记、graph 读围栏在 HIP 无 in-graph marker 时改 POST_REPLAY 等。

## 关键改动

```text
python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh: 阈值 bin 溢出重扫 + HIP 动态 smem
python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py: HIP BV16 tile + ROUND_CONV_PRODUCTS
python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py: greedy sampling variant 图捕获
python/sglang/srt/models/glm5_next.py: vision downsample 改 Conv2dLayer;Quark scale 名映射
python/sglang/srt/mem_cache/unified_cache/components/mamba.py: _select_finished_checkpoint 槽位选择
```

## 性能声明

Serving(4x MI355X VF,TP4/EP4,FP8 E4M3 KV,EAGLE 5/1/6,393-entry AgentX 语料,每档 900 s,`main` 与本 PR 同卡背靠背):c1-c32 geomean output throughput +14.3%(c1 +6.3%、c8 +10.2%、c16 +15.0%、c32 +26.0%);c48 处 `main` 掉出 frontier,本 PR 维持 2,411.3 tok/s。P90 tok/s/user 每档均更高。OneNexus MXFP4 对官方 FP8 checkpoint(同策略):+12.4% geomean output throughput、+18.2% geomean P90 per-user(c1-c48,双方零错误);对 `amd/GLM-5.3-Flash-Quark-MXFP4` 差异在 0.1% 以内。算子级(KDA fused verify,gfx950 隔离 VF,synthetic GLM TP4 shape):batch 8/T=8 参考两 kernel 38.9 us → 融合 BV16 28.4 us(-27.0%);EAGLE 5/1/6 shape batch 8 为 31.50 → 23.02 us、batch 16 为 37.66 → 33.03 us,在 #39524 之上约 26% 且输出 bit-exact。Vision merge:`[256,1024,2,2]` 布局首调 4.50 → 3.31 s(-26.5%), warmed graph 63.28 → 25.62 us(-59.5%)。PR 声明 KDA/vision 是隔离算子研究而非模型吞吐,B300 对照仅为上下文而非受控跨硬件结论。

## 验证与回归风险

gfx950 上 14 个改动测试文件与 main 的 Mamba checkpoint/radix lock-ref 测试全部通过(1,626 passed / 0 failed),含 native k-pool top-k 与 KDA bit-exactness kernel;每次 serving boot 通过 basic、30,031-token 长上下文、非贪心与 image-cached follow-up 探针;cookbook 命令 GSM8K 0.980 / MMLU 0.866。serving 配置下精度 GSM8K 0.978 / MMLU 0.870(sgl-eval 0.1.2,500 例,temperature 0);OneNexus 对 Quark checkpoint 的 MMLU/GPQA 差异未达 p<0.05。风险面:K-pool top-k 溢出重扫路径、attention backend 解包、Mamba ping-pong 簿记与 `owned_kv_len`/overallocation assert 放宽影响 spec 解码 + Mamba 的所有模型路径;greedy graph variant 使 draft 图捕获数量翻倍(仅 rejection sampling 下)。

## 关联

前置:十个 Day-0 enablement PR(#38545、#38546、#38547、#39317、#39338、#39339、#39340、#39341、#39778、#39779)、#37134(ROCm rejection sampling)、#39524(fused KDA verify 前身);CI 修复拆分至 #40721。同区域:#41725(GLM decode ROCm 优化)、#28451(KDA ReplaySSM)。
