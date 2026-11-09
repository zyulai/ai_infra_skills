# RFC #61761: [RFC]: DFlash2 speculative decoding performance on ROCm (MI355X)

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61761](https://github.com/vllm-project/vllm/issues/61761) |
| 作者 | zhentaocc |
| 创建日期 | 2026-10-15 |
| 状态 | 🟢 open |
| labels | kimi, rocm, speculative-decoding |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Speculative Decoding |

## 一句话结论
追踪 DFlash2 投机解码在 ROCm(MI355X)上的性能修复:汇总各在评审 PR 的实测收益与后续优化计划。

## 提案动机
DFlash2 在 ROCm 上比应有水平慢,高并发下 main 甚至低于不开投机解码。单卡 MI355X、Qwen3.8-27B(z-lab/Qwen3.8-27B-DFlash2)、7 个 draft tokens、MRV2、ShareGPT 512 输出 token、temperature 0 实测输出 tok/s(c=1/16/64):main 无投机 91.7/940.8/2855.4;main DFlash2 239.4/1829.9/2641.3(c=64 反而更慢);DFlash2 + #61433 为 266.8/2392.4/4065.3;再加 #61434(--no-enable-prefix-caching)为 262.2/2511.4/4456.2。各 DFlash2 行接受长度 4.0–4.3。GPU 时间占比(rocprofv3 内核 trace,c=64、含 #61433):GEMM(hipBLASLt)约 61%、GDN recurrent 更新 18.3%、Inductor 融合 norm/激活/RoPE 约 8%。

## 提出的方案
- #61433:spec decode 或 hybrid 模型时把 ROCM_ATTN 排到 TRITON_ATTN/ROCM_AITER_UNIFIED_ATTN 之后(draft 暂停评审,等 vLLM 的 aiter pin 升级包含 #62606 修复)
- #61434 与 #64325:GDN spec-decode 状态延迟写——每步只写一份全量状态加原始 per-token 输入,下一步开头重放已接受 token(fp32 SSM 状态下重放精确);#64325 是同一思路在 Triton fused GDN MTP kernel(#61855 后 ROCm 默认)中的实现,两者作用于不同 kernel、相互独立
- #61760:Triton unified_attention 为非因果投机行加 split-K(3D),并把修剪后的 sliding window 跨段切分
- #61429:精确 Triton vocab top-k 替代 torch.topk
- GLM-5.3-Flash 系列:#63070 可中断 CUDA graphs、#63071 KDA 延迟状态、#63073 Gluon KDA kernel、#63072 预计算 chunk indices
- 后续计划(前三项按序):① mamba_cache_mode=align(默认前缀缓存)下补 GDN 延迟状态——当前默认配置下 GDN 更新仍占 18% GPU 时间;② 每请求单一 GDN 状态槽(参照 ROCm/ATOM#1883 ReplaySSM;Qwen3.8-27B 按配置推算每槽约 144 MiB、7 draft tokens 约 1.15 GiB/请求,为估算非实测);③ ROCM_AITER_UNIFIED_ATTN 小 batch 长上下文 split-KV;④ GEMM 调优配置贡献给 aiter

## 优化点与收益
正文数字(单卡 MI355X,除注明外为 Qwen3.8-27B + DFlash2):#61433 +11%/+31%/+54%(c=1/16/64);#61434 +5.6%(c=16)/+10.6%(c=64);#64325 +6.1%/+8.2%(c=16/64),kernel 382 → 213 µs(64 序列);#61760 drafter attention B=1 超 2048 窗口时 4.7×,Gemma-4-26B-A4B 16k 输出 +32%/+28%/+9%(c=1/4/16);#61429 top-k kernel 快 4–6×(端到端在噪声内);#63070 +14–17%(无投机);#63071/#63073 步时 −4.5%/−4.7%(c=16/64)、再 −4.5%(c=64)。

## 现状与落地
🟢 open,反馈期一周。正文按"在评审 / 已关闭 / 下一步 / 不做(含理由)"组织:#62381 因被 #61855 取代而关闭,#62415 因 NaN 已由 #60767 消除而关闭;后续前三项计划按序推进。

## 关联
- #61433、#61434、#64325、#61855、ROCm/ATOM#1883

---
> 摘要由 LLM 基于 RFC 正文生成,未经人工核实;准确信息以原帖为准。
