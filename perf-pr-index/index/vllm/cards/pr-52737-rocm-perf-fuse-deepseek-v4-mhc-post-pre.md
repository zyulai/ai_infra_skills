# PR #52737: [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-08-20 |
| 分类 | Attention / Kernel |
| Label | rocm, ready, deepseek, DSv4, dflash |
| Diff | diffstat 缺失(pack 未提供 +x/-y 与文件数统计,diff 内容有保留) |
| 链接 | https://github.com/vllm-project/vllm/pull/52737 |

## 一句话结论

ROCm 上把 DeepSeek-V4 的 mHC post/pre 与 RMSNorm 融合为 AITER 单内核,mHC 段快 38.35%,e2e 吞吐 +0.66~1.81%。

## 动机

DeepSeek-V4 的 mHC(multi-hyper-connection)路径在 ROCm 上此前是一串分离内核:`mhc_post` + `mhc_pre_gemm_sqrsum`、`mhc_pre_big_fuse` + `add_rmsnorm_quant` 分别独立启动。AITER 已提供融合实现(`mhc_fused_post_pre_gemm_sqrsum`、`mhc_pre_big_fuse_rmsnorm`)但 vLLM 未接入,且旧 `use_fused_mhc` 门控在 AITER 可用时反而禁用融合。

## 优化点

两层融合:其一,`MHCFusedPostPreOp.forward_hip` 在 `HAS_AITER_MHC && hidden_size%256==0 && hc_mult==4` 时走新注册的 custom op `mhc_fused_post_pre_aiter`(包装 AITER 的 `mhc_fused_post_pre`,一次完成 residual 混合 + 下一 sub-layer 的 pre,返回 vLLM 顺序的 residual/post_mix/comb_mix/layer_input);`mhc_pre`/`mhc_post` 也统一加 hc_mult==4 门控。其二,RMSNorm 融合:`mhc_pre` 与 fused op 增加 `norm_weight/norm_eps` 参数,decoder layer 里 `attn_norm`/`ffn_norm` 的权重直接传进 pre kernel,hidden size 属于 `{1280, 2560, 4096, 7168}`(AITER fused-rmsnorm 支持集)时跳过 standalone RMSNorm,否则保留 fallback。`use_fused_mhc` 改为 "AITER 融合可用或 TileLang 可用"。

## 关键改动

```text
vllm/model_executor/kernels/mhc/aiter.py: 新增 mhc_fused_post_pre_aiter custom op(+fake impl)
vllm/_aiter_ops.py: mhc_pre 加 norm_weight/norm_eps,新增 mhc_fused_post_pre 包装
vllm/model_executor/layers/mhc.py: forward_hip 分派 AITER 融合路径(hc_mult==4 门槛)
vllm/models/deepseek_v4/amd/model.py: fuse_mhc_rmsnorm 按支持集折叠 attn/ffn RMSNorm
```

## 性能声明

AMD MI350X、deepseek-ai/DeepSeek-V4-Pro,TP=8、async-scheduling、kv-cache-dtype fp8、moe-backend aiter、FULL_AND_PIECEWISE compilation,SA InferenceX `8k1k` workload(8192-in/1024-out random,并发 1/4/16/64/256):output token throughput 全并发 +0.66%~+1.81%,mean TPOT -0.73%~-1.81%,mean TTFT 混合(conc=1 +9.86%、conc=64 +1.80% 变差;conc=4 -0.60%、conc=16 -1.15%、conc=256 -4.59% 改善);mHC 融合段 profile 显示 38.35% faster(PR 附图)。GSM8K:30-shot conc=256 下 0.8461 → 0.8681,默认 5-shot conc=64 为 0.9386。

## 验证与回归风险

`tests/kernels/test_mhc_kernels.py` 扩展:ROCm fallback 路径(含 RMSNorm)与 AITER fused-norm 路径的数值对拍。风险面:仅 ROCm/AITER 路径,CUDA/XPU 不变;AITER mHC 内核要求 hc_mult==4 与 hidden_size%256==0,RMSNorm 融合再叠加 hidden size 白名单,不满足时逐级回退(tilelang/standalone norm);TTFT 在低并发略升,吞吐收益随并发下降(conc=64 仅 +0.66%)。

## 关联

同区域:DeepSeek-V4 mHC 内核线(tilelang 融合路径为前序);同批 PR #53906(GLM-5.3-Flash)复用同一 mhc layers 测试与 ROCm mHC fallback;dflash label 指向 DFlash 投机解码在同模型上的联动。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
