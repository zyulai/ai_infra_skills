# PR #44489: [Perf] [AMD] deterministic: Allow AITER decode attention on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @aarnetalman |
| 合并 | 2026-11-03 |
| 分类 | Speculative Decoding | tier: major |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 含 3 个文件:`arg_groups/overrides.py`、`aiter_backend.py`、`test_model_overrides.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/44489 |

## 一句话结论

让 MI350X/MI355X 确定性推理在 decode 阶段用 AITER attention,确定性吞吐提升 1.09-1.91x 且所有确定性检查通过。

## 动机

在 MI350X/MI355X (gfx950) 上,decode attention 是确定性推理模式的主要成本,而确定性模式此前只允许确定性 backend(如 Triton)作 decode attention。AITER 的 paged decode attention(`paged_attention_ragged`)把每个请求的 KV cache 切成固定 256-token 的 partition 并按请求逐一 reduce,切分只取决于该请求自身的 context length、与 batch 组成无关,因此是 batch-invariant 的,可以安全用于确定性模式。本 PR 建立在 #44438(确定性模式检查 `--prefill-attention-backend` / `--decode-attention-backend`)之上;decode 必须显式设置:`--attention-backend aiter --prefill-attention-backend triton` 仍被拒绝,因为 `--attention-backend` 本身必须是确定性 backend。

## 优化点

改动是配置门控而非新 kernel。`arg_groups/overrides.py` 的 `_deterministic_attention_backend` 改为先按 phase 构建 allowed backends 列表:decode 在 gfx950、非 MLA 模型、无 speculative decoding 三个条件同时满足时追加 `aiter`,否则在报错信息里写明拒绝原因;随后两个 phase 在同一循环里校验。`aiter_backend.py` 在确定性模式下,若 decode 实际不会运行 `paged_attention_ragged`(unified attention 或向量化 5D KV-cache layout,包括 sliding-window 模型)则在 backend 启动时拒绝,保证被接受的配置永远运行已验证 batch-invariant 的 kernel。明确的拒绝矩阵:AITER 作 `--attention-backend`/`--prefill-attention-backend`(AITER prefill 非 batch-invariant,测试中约 480 token 起结果随 batch 变化);MLA 模型(MLA decode 的 KV split 按 batch size 决定);非 gfx950 GPU(未测试);speculative decoding(verify/extend kernel 未验证);sliding-window、`SGLANG_USE_AITER_UNIFIED_ATTN`、5D KV layout;`--enable-mixed-chunk`(任意 backend,mixed batch 会把 decode token 走 prefill kernel,由 #44701 在确定性模式下单独关闭)。确定性模式关闭、不设 `--decode-attention-backend aiter` 的所有配置、radix cache(跟随仍是确定性 backend 的 `--attention-backend`)均不受影响。

## 关键改动

```text
python/sglang/srt/arg_groups/overrides.py: _deterministic_attention_backend 按 phase 构建允许列表,decode 在 gfx950/非 MLA/无 spec 时接受 aiter
python/sglang/srt/layers/attention/aiter_backend.py: 确定性模式下对 unified-attention / 5D KV layout 的 decode 路径拒绝启动
test/registered/unit/test_model_overrides.py: 新增 gfx950/MLA/prefill/spec 四维组合的接受/拒绝表测试
```

```python
if (
    is_gfx95_supported()
    and not use_mla_backend(view)
    and view.speculative_algorithm is None
):
    decode_choices.append("aiter")
```

## 性能声明

条件:MI350X、`benchmark.serving` random 数据集(input 1024/8192,output 1024,range ratio 1.0,256 prompts,concurrency 256(1k)/128(8k),request-rate inf,每行同节点上两配置对比,3 轮交错取中位数,轮间差 <0.5%),确定性模式 AITER decode vs 两相 Triton:

- Qwen3-8B TP1:1k/1k 3,691→7,064 tok/s(1.91x),ITL p50 60.6→27.1 ms;8k/1k 737→1,284 tok/s(1.74x)。TP2:1.60x / 1.58x;TP8:1.27x / 1.31x。
- Qwen3-30B-A3B TP1:1k/1k 4,159→6,700 tok/s(1.61x);8k/1k 851→1,303 tok/s(1.53x)。
- Qwen3.8-27B:TP1/TP2 共四格 1.09x-1.13x(增益小因为仅 1/4 层用 attention,其余为 Gated DeltaNet)。
- 相对 normal mode + AITER attention 的参照:确定性模式 Qwen3-8B TP1 从 0.382x→0.731x(1k/1k)、0.329x→0.572x(8k/1k);TP8 从 0.377x→0.476x、0.301x→0.393x。
- TTFT p50 不变(prefill 仍 Triton),仅 Qwen3.8-27B TP1 1k/1k 低 12%(负载下 decode step 更快);deterministic mode 关闭时无变化。

## 验证与回归风险

准确性:8x MI350X(镜像 rocm/sgl-dev:v0.5.21-rocm724-mi35x-20261013,SGLang 27b68303d0,AITER e7d2453f2);`test_deterministic` single/prefix/radix_cache 各 50 trials + prefix logprob;bitwise batch-size 检查(同一 prompt 重复 1-50 次,1,275 条响应的 token IDs 与 selected-token logprobs 与 bs=1 精确相等)覆盖 Qwen3-8B TP1-8、Qwen3.8-27B TP1/2、Qwen3-30B-A3B TP1 及两个 fp8_e4m3 KV 变体,全部 1,275/1,275;长上下文 8k/12k/16k prompts 单独与混批共 15/15 次比较逐位一致;GSM8K(1,319 题,temperature 0)Triton/AITER decode:93.56%/93.48%、94.77%/94.39%、96.06%/95.83%;per-layer hook 检查 TP1/TP2(Qwen3-8B)未发现输入相同而输出不同的 op,decode attention 每秩比较 5,940 次。单测 `test_model_overrides.py -k deterministic`:仅 #44438 时新测试 1 failed 6 passed,本 PR 后 6 passed。风险:MI350X 上确定性模式依赖两个未合并的开放 PR(#43788 AITER GEMM/MoE bypass、#34543 超大输入 all-reduce);`p_vs_d` 模式(prefill vs decode logprobs)有差异,与两相 Triton 相同,不在 AMD deterministic CI 内。

## 关联

#44438(前序:确定性模式 per-phase backend 校验)、#44701(mixed-chunk 在确定性模式关闭)、#43788 / #34543(运行依赖)。无正文列出的同分类其他 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
