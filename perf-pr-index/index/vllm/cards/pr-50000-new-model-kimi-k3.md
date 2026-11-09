# PR #50000: [New model] Kimi K3

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ZJY0516 |
| 合并 | 2026-07-30 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, kv-connector, nvidia, quantization, rust, kimi, k3 |
| Diff | diffstat 缺失(pack 中为 +None/-None,None files;diff 仅保留部分文件,约 14 个,其余被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/50000 |

## 一句话结论

为 vLLM 增加 moonshotai/Kimi-K3 模型支持,并随附 MTP/DSpark 投机解码、MNNVL 多节点 custom collectives、Situ 激活 MoE 等配套性能路径。

## 动机

Kimi K3 是 Moonshot 的新一代 MoE 模型(文本+图像输入),vLLM 尚不支持。PR 正文指出优化细节见 vLLM 官方博客(2026-07-27 K3 博客),运行需 FlashInfer v0.6.16rc5,并依赖 FlashInfer 的 SITU trtllmgen MOE(#4180)与 MLA(#4178)两个外部改动。该 PR 实际是聚合入口,kernels/model files(#50089、#50090)、python frontend(#50093)、rust frontend(#50104)拆分为独立 PR。

## 优化点

除模型实现外,diff 中包含多条性能相关改动:(1) MTP 投机解码:注册 `kimi_k3_mtp` 方法,K3 的 text_config 内嵌 `num_nextn_predict_layers`,hf_config_override 把 model_type 改写为 `kimi_k3_mtp` 并生成 KimiK3MTPModel 配置;(2) DSpark 投机解码:新增 `K3DSparkModel`(MLA draft),checkpoint 权重映射到 fused_qkv_a_proj/gate_up_proj,复用目标模型 embed/lm_head 并跳过 confidence_head,Markov head 全 TP replica 不走集合通信;MLA DSpark 明确不支持 decode context parallelism;(3) MLA 非因果多 token decode:metadata builder 支持 `non_causal_multi_token_decode`,非因果块走 decode 路径且不依赖 CPU 长度,要求显式声明且 query 块长度一致;(4) MNNVL 多节点 custom collectives:CustomAllreduce 支持 world_size=16 与 all-gather/reduce-scatter,跨节点时经 torch symmetric memory multicast buffer + Lamport 帧实现,取消"多节点一律禁用 custom all-reduce"的限制;(5) ROCm aiter fused_moe 新增 Situv2 激活类型与 beta/linear_beta 参数。

## 关键改动

```text
vllm/config/speculative.py: 注册 kimi_k3_mtp 方法与 K3DSparkModel 架构校验、DCP 不支持检查
vllm/distributed/device_communicators/custom_all_reduce.py: MNNVL multicast all-gather/reduce-scatter,world_size 16,symm_mem rendezvous + Lamport epochs
vllm/distributed/device_communicators/cuda_communicator.py: 暴露 custom_all_gather/custom_reduce_scatter 入口
vllm/_aiter_ops.py: ROCm aiter fused_moe 支持 Situv2 激活(beta/linear_beta 透传)
vllm/config/vllm.py: K3 相关架构纳入 V2 model runner 默认集合;ROCm 上排除 K3(V2 runner profile run 会 fault)
```

关键代码(K3 MTP 配置改写):

```python
if hf_config.model_type == "kimi_k3":
    text_config = getattr(hf_config, "text_config", hf_config)
    n_predict = getattr(text_config, "num_nextn_predict_layers", None)
    hf_config.model_type = "kimi_k3_mtp"
    hf_config.update(
        {"n_predict": n_predict, "architectures": ["KimiK3MTPModel"]}
    )
```

## 性能声明

PR 正文未给出 benchmark 数字(regex 未检出性能声明),仅指向 https://vllm.ai/blog/2026-07-27-k3 的优化细节博客;投机解码提供 DSpark 权重 Inferact/Kimi-K3-DSpark。PR 未注明测试条件。

## 验证与回归风险

新增 `tests/models/test_dspark_mla.py`(权重映射、Markov head replica 不调用 TP collectives、无 compile wrapper 入口)、`tests/v1/attention/test_mla_noncausal.py`(非因果 decode 块构建、显式支持与均匀 query 块校验、CUDAGraph 尾部 padding),扩展 registry 的 K3/K3-DSpark/K3-MTP 条目与 deepgemm permute padding scales 测试。风险面:custom_all_reduce 初始化路径大改(单节点/多节点分支、共享缓冲尺寸),影响所有 TP 通信;V2 model runner 判定逻辑重构涉及 hybrid/MoE 模型;parallel.py 移除多节点自动禁用 custom all-reduce 的行为变更;Mamba cache align 模式删除了两个 assert。

## 关联

拆分子 PR:#50089、#50090(kernels and model files)、#50093(python front end)、#50104(rust frontend);外部依赖 FlashInfer #4180(SITU trtllmgen MOE)、#4178(MLA)、v0.6.16rc5 release。同区域:DSpark 投机解码(PR #46995)、MTP 投机解码(PR #12755)、FlashMLA 集成(PR #13747)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
