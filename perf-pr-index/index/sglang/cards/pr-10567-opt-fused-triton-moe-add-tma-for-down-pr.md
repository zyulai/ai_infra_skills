# PR #10567: Opt fused triton moe: add tma for down proj kernel

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xu-yfei |
| 合并 | 2025-10-28 |
| 分类 | MoE (tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +None/-None);diff 内容可用(3 个文件因超长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/10567 |

## 一句话结论

Fused Triton MoE 的 down proj kernel 用 TMA 重构并独立调优,H20 TP8 prefill 中其计算利用率从 45.20% 提到 81.12%,8K tokens 平均延迟降约 1 ms。

## 动机

H20(96GB)TP8 prefill 性能分析发现:每层第二个 MoE(downsample Fused Triton MoE)的延迟与第一个 MoE(upsample)相当,尽管其权重数据量与计算量只有一半,延迟表现不合理——瓶颈在访存组织与调优配置,而非算力。

## 优化点

四项机制改动:(1) b_scale 读取优化——当 `BLOCK_SIZE_N <= group_n`(block 量化参数)时,每次只需读 1 个 b_scale 元素而非 `BLOCK_SIZE_N` 个,并可先乘 `a_scale * b_scale` 再与 dot 结果相乘,减少计算开销;(2) 第二个 MoE 基于 TMA 重构——输入 A 的访问从离散抓取改为连续组织,第一维从 `(num_tokens * top_k)` 改为 `(num_blocks * block_size_m)`,输入 A 与权重 B 的访问用 TMA 封装;(3) 用真实 expert 分布调优——expert 负载通常不均,原调优基于随机生成 expert 不反映真实情况,改为采录实际推理的 `topk_ids`(在 `deepseek_v2.py` 打点 `torch.save`)作为 tuning 输入;(4) 两个 MoE 分别调优、独立加载配置(down 配置文件带 `_down` 后缀与 `USE_TMA` 字段,小 batch 下 tuning 判定不用 TMA 更快)。新增独立调优脚本 `tuning_fused_moe_triton_sep.py`(支持 fp8_w8a8、block shape、`--topk-ids-dir`)。

## 关键改动

```text
benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py: up/down 分离调优脚本(新文件,支持真实 topk_ids 输入)
python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=257,N=256,..._down.json: H20 down-MOE 独立配置(含 USE_TMA 标志,新文件)
python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: filter_expert 参数与 down kernel 分发
(pack 未收录的 3 个文件): fused_moe_triton_kernels.py 中的 TMA kernel 实现等,见 PR 全 diff
```

## 性能声明

H20 TP8、DeepSeek-R1、`--quantization fp8`、flashinfer attention、chunked-prefill-size 16372:

- 8K tokens 场景 100 样本平均延迟:2.430 ms → 1.435 ms(-0.995 ms);第二个 MoE 计算利用率 45.20% → 81.12%。
- Trace(单请求 context 16372x3+502):before `fused_moe_kernel` 1.525 s / 348 次;after `fused_moe_kernel` 706.7 ms / 174 次 + `fused_moe_down_tma_kernel` 430.0 ms / 174 次。
- 单请求 TTFT(input 512-8192,output 1,`--max-concurrency 1`):Mean TTFT 降幅 2.52%(512)至 9.00%(2560),大多在 7-9%;512 input 时吞吐 8.6 → 8.82 req/s。
- 精度:gsm8k 0.951 / 0.953(Invalid 0.000);mmlu 平均 0.871(57 subject)。

## 验证与回归风险

验证手段为 gsm8k/mmlu 评测与 trace 对比(如上);"Add unit tests" 项在合并时未勾选。风险面:down-MOE 配置为 H20 + DeepSeek-R1(E=257, N=256, fp8_w8a8, block_shape=[128,128])专属,其他硬件/模型需自行重调;USE_TMA 由配置文件逐 batch-size 决定,小 batch 走非 TMA 路径;`filter_expert` 新参数改变 `fused_experts` 签名(tuning 中传 False);新增的 `_down` 配置文件命名约定要求 config loader 按 down_moe 后缀解析。

## 关联

同区域:Fused Triton MoE kernel 与其 tuning 体系(`tuning_fused_moe_triton.py` 的分离版为本次新增);adapted from vLLM `benchmark_moe.py`;pack 内未给出其他具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
