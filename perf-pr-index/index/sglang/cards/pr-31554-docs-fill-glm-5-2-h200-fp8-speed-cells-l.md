# PR #31554: [Docs] Fill GLM-5.2 H200 FP8 speed cells (low-latency, balanced); fix MTP notation

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Smallfu666 |
| 合并 | 2026-08-14 |
| 分类 | Speculative Decoding |
| Label | documentation |
| Diff | pack 未提供 diffstat(+None/-None across None files);diff 可见改 2 个文档文件(GLM-5.2.mdx 与 glm-5.2-benchmarks.jsx) |
| 链接 | https://github.com/sgl-project/sglang/pull/31554 |

## 一句话结论

纯文档 PR:按 acceptance-length-pinned 方法补齐 GLM-5.2 在 8xH200 + FP8 的三档速度基准数据,并修正 FP8/NVFP4 配方的 MTP 表示法混用。

## 动机

GLM-5.2 的三个 H200 + FP8 速度单元自 acceptance-length-pinned 测量方法落地以来一直处于"pending re-measurement"状态。作者按该方法在 8xH200 上实测了 low-latency 与 balanced 两档(连同 high-throughput 一并填入),并在审计配方时发现文档中的 MTP 记法有误:文档原来笼统写 balanced 为 2-1-3,实际 FP8 balanced 用的是 1-1-2。

## 优化点

无代码改动,全部是 benchmark 数据与文档修正:`glm-5.2-benchmarks.jsx` 中把三个 H200+FP8 的 `match` 桩替换为实测数据(标明 sglang_version `v0.5.14 @ 49e384ce`);`GLM-5.2.mdx` 的 Note 区分 FP8 balanced(1-1-2,即 `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`)与 NVFP4 balanced(保持 2-1-3),并注明 H200 FP8 单元与其他单元测自不同版本(main @ 09ca4fc)。测量方法与 B200/GB300 单元一致:通过 `SGLANG_SIMULATE_ACC_LEN` 钉住 EAGLE 接受长度(low-latency=3.5,balanced=2),high-throughput 无投机解码。

## 关键改动

```text
docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx: 填入 H200+FP8 low-latency/balanced/high-throughput 三组 speed 数据
docs/cookbook/autoregressive/GLM/GLM-5.2.mdx: 修正 MTP 记法,注明版本差异
```

数据单元格示例(balanced, DP8 + DeepEP + 1-1-2 EAGLE):

```jsx
match: { hw: "h200", quant: "fp8", strategy: "balanced", nodes: "single" },
sglang_version: "v0.5.14 @ 49e384ce",
speed: [
  { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
    ttft_ms: 7473, tpot_ms: 23.49, tokens_per_sec_per_gpu: 2343 },
],
```

## 性能声明

测试条件:单节点 8xH200 141GB(TWCC nano4),模型 `zai-org/GLM-5.2-FP8`(HF commit 70311cfa),bench_serving random 8192/1024、`--random-range-ratio 1.0`、flush-cache。low-latency 为 TP8 + EAGLE 5-1-6 + ACC=3.5;balanced/high-throughput 为 TP8 + DP8 DP-Attention + DeepEP(max-running 256;high-throughput 无投机解码)。TTFT/TPOT 为 P50;tokens_per_sec_per_gpu = 总(input+output)吞吐 ÷ 8。

| cell | conc | TTFT p50 (ms) | TPOT p50 (ms) | total tok/s/GPU | accept_length |
|---|---|---|---|---|---|
| low-latency (ACC=3.5) | 1 | 668 | 5.05 | 197 | 3.51 |
| low-latency (ACC=3.5) | 16 | 6148 | 16.11 | 813 | 3.48 |
| balanced (ACC=2) | 64 | 7473 | 23.49 | 2343 | 2.00 |
| balanced (ACC=2) | 256 | 80562 | 28.08 | 2391 | 2.00 |
| high-throughput | 1024 | 553480 | 61.71 | 1656 | N/A |

可复现性:低延迟与 balanced 在新节点独立复测全部指标差异 < 1.4%;high-throughput c=1024 换节点复测 TTFT −0.19%、TPOT −0.33%、总吞吐 +0.37%。与历史未钉 accept length 的数据对比:未钉低延迟 c=1 TTFT 665.5ms / TPOT 3.03ms vs 历史 662/3.03,独立复现误差约 1%。作者注明 balanced c=256 的 TTFT(80.6s)远高于 B200(18.7s)是 H200 141GB 上 KV pool 在 c~64 即饱和导致的排队(吞吐仅 +2% 而 TTFT p50 增大 10.8x)。

## 验证与回归风险

纯文档改动,无代码路径受影响,无新增测试(正文标注 Accuracy Tests N/A)。风险仅在文档一致性:两个版本号并存(v0.5.14 vs main @ 09ca4fc)可能让读者误比较,作者已留注释表示愿意在新容器上重测统一版本。

## 关联

同区域优化:同文档中 B200/GB300 + FP8 与 NVFP4 的 EAGLE/MTP 基准单元(同一 acceptance-pinning 方法),属 Speculative Decoding 相关文档。PR 正文未给出具体关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
