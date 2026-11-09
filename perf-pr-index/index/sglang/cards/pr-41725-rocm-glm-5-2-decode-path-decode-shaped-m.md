# PR #41725: [ROCm] GLM-5.2 decode path: decode-shaped MoE/MLA tiles, split speculative softmax, and bf16 GEMM routing

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jiejingzhangamd |
| 合并 | 2026-10-04 |
| 分类 | Speculative Decoding(major) |
| Label | quant, amd, run-ci, jit-kernel |
| Diff | pack 未提供 diffstat(diff 正文可用,共 10 个文件改动) |
| 链接 | https://github.com/sgl-project/sglang/pull/41725 |

## 一句话结论

在 gfx950 上为 GLM-5.2 decode 路径修正四个 kernel 的 shape 适配,端到端 +5.7%。

## 动机

作者在 profiling GLM-5.2-MXFP4 TP8/EP1、104k context 的 server 时发现,decode batch 规模下四个路径都让 256-CU 的 GPU 大部分空闲:MoE shared-expert append 前后两次 padded-region fill 每层每步多两次 launch;MLA absorb BMM 落入为 square GEMM 调优的 tile,16-32 个 workgroup 饿死整机;speculative softmax 每行只用一个 block 归约 154880 宽的词表且读三遍;aiter 调优表按 `(N, K)` 索引且不回退到邻居条目,MLA q up-projection `(2048, 2048)` 落到 torch(8.4 us,而 weight-bandwidth roofline 是 1.05 us)。这些都不是靠减少算术量能变快的,是 shape/调度问题。本 PR 是 #40358 的重新提交。

## 优化点

四个改动同一主题:让 kernel 的 launch 形状匹配 decode 的实际行数(1-32 行)。(1) MoE:把 `_post_process_topk_ids` 中两次独立的 padded-row fill 折进 aiter shared-expert append kernel——该 kernel 本来就写这些元素,launch 数从 3 降到 1,要求 EPLB remap 关闭(`pad_fill_id=0` 与 `remap(0)=0` 可交换),不满足前置条件时走原路径。(2) MLA:为 absorb BMM 增加 decode 专用 tile 配置 `_absorb_bmm_config`,按 `heads * cdiv(M,BM) * cdiv(N,BN)` 覆盖 CU 数即止,不无限切分(kernel 按 token group 自行量化 A tile,多余 workgroup 会重复该工作,M=200/N=512 时无界切分反而慢 3.7x)。(3) spec:新增 `temperature_softmax`,行数 ≤32 时把 vocab 宽归约按 CTA 切分(partial max+sumexp、online-softmax 合并、normalize),并把温度除法折进第一遍;超过行数上限交还 torch。`fast_sample` 的 argmax 路由到已有 `row_argmax`。(4) linear:以 `K` 为门控(`_prefer_triton_a16w16`),把 aiter 调优表缺失的 bf16 decode shape 路由到 aiter 自带 Triton a16w16 kernel,限制 M 行数以避开 prefill 区间(prefill 上 Triton kernel 反而慢 1.25-1.44x)。

## 关键改动

```text
python/sglang/kernels/ops/sampling/temperature_softmax.py: 新增 split-row temperature softmax(ROW_LIMIT=32,超限回退 torch)
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py: _absorb_bmm_config decode tile 选择
python/sglang/srt/layers/quantization/unquant.py: _prefer_triton_a16w16 门控 bf16 GEMM 路由
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py: append kernel 增加 HAS_PADDING 分支,自行填 padded rows
python/sglang/srt/layers/moe/topk.py: 折叠 padded fill 的条件与传参
```

## 性能声明

Kernel 级(MI355X,vocab 154880,fp32,CUDA graph replay):softmax 1 行 8.7x、32 行 3.1x,48/84 行回退 torch 为 1.0x;argmax 1 行 2.6x、32 行 3.9x。MoE append fold 单次 4.74 → 1.61 us 且随行数平坦;MLA absorb `w_vc`(N=256,K=512)M=48 时 18.1 → 3.6 us,`w_kc` 12.9 → 3.1 us。E2E(GLM-5.2-MXFP4,TP8/EP1,concurrency 1+2 几何均值,fp8 KV,104k context,EAGLE 5/6,`--dsa-decode-backend triton`):三 kernel 改动 +3.94%,加 linear routing 后 +5.67%(concurrency 1 单独 +4.95%,concurrency 2 +6.39%);aiperf trace replay(仅三 kernel 改动)per-user p90 +3.55%,per-GPU 总吞吐 +1.62%。PR 明确说明这是单配置 paired A/B,不与 leaderboard 比较。

## 验证与回归风险

新增测试:`test_temperature_softmax.py`(对 torch 参照 + grammar mask 全掩 split 用例)、`test_aiter_a16w16_gate.py`(路由门控 shape 表)、`test_rocm_mla_absorb_fusion_gate.py` 增加 tile 配置断言、append kernel padded rows 用例与 `test_topk_padded_region.py` 权重掩码用例;另随 PR 附 `benchmark/kernels/bench_bf16_linear_routing.py`(ABBA 交错测时)。风险面:MoE fill 折叠依赖 EPLB remap 关闭与 `pad_fill_id=0` 前置条件,条件不满足时回退原路径;linear routing 门控按 K 判定,若 prefill shape 误入路由会变慢(以 M 行数上限防护);`temperature_softmax` 仅在 HIP 下启用,CUDA 路径不变。

## 关联

#40358(原始提交,重写 SHA 后重新提交);同区域:ROCm aiter 相关 #4053、#39273;speculative 采样 softmax/argmax 相关 #28451。
