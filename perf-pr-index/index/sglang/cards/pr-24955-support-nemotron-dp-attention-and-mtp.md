# PR #24955: Support Nemotron DP attention and MTP

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Zhichenzzz |
| 合并 | 2026-06-12 |
| 分类 | Speculative Decoding |
| Label | run-ci, deterministic, run-ci-extra, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计);diff 有省略(3 个文件未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/24955 |

## 一句话结论

让 Nemotron-H 混合模型支持 DP attention 与 MTP,削减 KV/Mamba 状态在 DP 组内的重复。

## 动机

Nemotron-H 是 Mamba2 + full attention + MoE 的混合模型,而 DP attention 此前只支持标准 attention + MLP 的 transformer:Nemotron-H 的 attention 和 Mamba 只能跑全局 TP,DP-local(attn-TP)布局与 full-TP MoE 布局之间没有 gather/scatter 路径;同时 DP attention 为 collective 对齐引入的 per-rank token padding 在 Mamba 递归路径中未被处理——padding 行会写 KV cache、更新 Mamba 状态并污染 residual。

## 优化点

开启 DP attention 后,MambaMixer2 与 attention 改在 attention-TP 子组(attn_tp_rank/size)上运行而非全局 TP,gated RMSNorm 的方差归约同样走 attn-TP 组,从而削减 KV 与 Mamba 状态的跨 DP 重复;MoE/MLP 仍保持 full TP。每层通过 LayerCommunicator 把 DP-local、attn-TP-partial 的输出 gather 成 full-TP 布局供 MoE/MLP 使用,再 scatter 回来(TP_ATTN_FULL 与 FULL 之间转换)。padding 行被端到端处理:按真实(未填充)batch size 与 prefill token 数裁剪/置零,确保不写 KV cache、不更新 Mamba 递归状态、不泄漏进 residual,并把真实 batch size 穿入 Mamba metadata 与 CUDA-graph replay。o_proj/out_proj 以 reduce_results=False 加按 chunk 的全局 TP 模拟 fp32 累积,使 deferred reduction 与非 DP 数值一致。另外为 hybrid-SSM 模型引入 idle rank 的 MAX_LEN fabricated-row 转换,覆盖 MTP target_verify 与 draft worker 场景。

## 关键改动

```text
python/sglang/srt/models/nemotron_h.py: decoder layer 重构,attn/mamba 层接入 layer_communicator 的 DP 路径
python/sglang/srt/layers/attention/mamba/mamba.py: MambaMixer2 按 attn-TP 组切分,padding 行裁剪
python/sglang/srt/layers/communicator.py: force_layernorm_before_dp_gather 与 dp_gather_replicate
python/sglang/srt/layers/dp_attention.py: hybrid-SSM 启用 MAX_LEN padding 模式(idle rank)
python/sglang/srt/model_executor/forward_batch_info.py: idle rank fabricated-row 转换与真实 batch size 透传
```

## 性能声明

PR 正文的 "Speed Tests and Profiling" 一节为空,未提供吞吐/延迟数字;素材中也未检测到 perf claim。仅有准确性数据(见下)。

## 验证与回归风险

正文给出 gsm8k smoke test(Nemotron 模型,tp=8,eager,parallel=32,200 题):no DP/no MTP 0.955、DP attention only 0.970、MTP only 0.955、DP attention + MTP 0.955,invalid 均为 0.000,作者据此说明无准确性回退、无 hang。回归面:改动触及 DP attention 通用路径(dp_attention.py 的 padding 模式选择、communicator 的 gather 逻辑、forward_batch_info 的 prepare_mlp_sync_batch),影响所有开启 DP attention 的模型;flashinfer backend 的 CUDA graph max_bs 现按 capture 对齐放大;hybrid 模型的 idle 转换逻辑是新增分支。

## 关联

无(素材未提及其他相关 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
