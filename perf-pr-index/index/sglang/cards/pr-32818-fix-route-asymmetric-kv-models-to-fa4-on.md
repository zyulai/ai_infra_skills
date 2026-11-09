# PR #32818: [Fix] Route asymmetric-KV models to fa4 on SM100 and pin MiMoV2 FP8 MoE to flashinfer_trtllm

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-07-29 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/32818 |

## 一句话结论
SM100 上把非对称 KV 模型自动分派到 fa4、并把 MiMoV2 FP8 MoE 钉到 flashinfer_trtllm,bs=1 decode 吞吐 +11.8%。

## 动机
MiMo-V2.5 无法在 SM100/SM103 用默认 flag 部署(#31243):SM100 auto-dispatch 选中 `trtllm_mha`,其 paged-KV kernel 要求 K/V 行宽相等,而 MiMoV2 是非对称 KV(`head_dim` 192、`v_head_dim` 128),decode CUDA-graph capture 直接报错 `Check failed: key_cache.size(i) == value_cache.size(i)`;用户只能退回 triton 并禁用 CUDA graphs。另外 MiMoV2 FP8 的 `moe_runner_backend=auto` 在非 DeepEP/Mooncake/NIXL a2a 后端时会落到 triton fused-MoE runner,比 flashinfer_trtllm 慢约 12%。

## 优化点
三处改动:(1) `ModelConfig` 新增 `has_asymmetric_kv` property(`head_dim != v_head_dim or swa_head_dim != swa_v_head_dim`),做成 property 是因为 MLA 特判仍会改写 `v_head_dim`;(2) `_get_default_attn_backend` 在 SM100 上把非对称 KV 的 MHA 模型解析到 `fa4`(fa4 能携带 `v_head_dim`),这是能力检查而非模型特例,MLA 模型走独立分支不受影响;(3) `_mimo_v2_overrides` 在 SM100 + FP8 checkpoint + `moe_runner_backend` 仍为 `auto` 时钉到 `flashinfer_trtllm`(FP4 checkpoint 走 `flashinfer_mxfp4`,故按 fp8 门控)。同时更新 cookbook 与部署 snippet。

## 关键改动

```text
python/sglang/srt/configs/model_config.py: 新增 has_asymmetric_kv property
python/sglang/srt/server_args.py: SM100 默认 attention backend 对 asymmetric-KV MHA 返回 fa4
python/sglang/srt/arg_groups/overrides.py: _mimo_v2_overrides 在 SM100+FP8+auto 时 pin moe_runner_backend=flashinfer_trtllm
docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx 及 snippet: 补充 Blackwell 必需 flag
```

## 性能声明
MiMo-V2.5 (310B) TP=4、bs=1 decode、`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle`:triton(原 `auto` 结果)315.7 tok/s、accept_length 2.88;flashinfer_trtllm 353.0 tok/s、accept_length 2.94。+11.8% output throughput(扣除 2.1% 的 accept length 提升后约 +9.6%)。

## 验证与回归风险
`test_model_overrides.py` 67 passed(含新增 SM100 FP8 pin、FP4 不 pin、显式用户选择不被覆盖等用例);端到端验证:不带任何 backend flag 的启动命令现在解析为 `attention_backend='fa4'`、`moe_runner_backend='flashinfer_trtllm'`、`page_size=128` 并正常服务,生成 token 与显式传 flag 完全一致。风险:改的是默认 backend 解析逻辑,影响所有 SM100 上的非对称 KV MHA 模型路由;CI 状态显示 PR Test (Base)/(Extra) 为失败标记。

## 关联
#31243(该问题的来源报告);同模型部署方向见 cookbook 更新。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
