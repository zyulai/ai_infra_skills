# PR #50272: [Bugfix] Fix speculative decoding for short_conv (LFM2) models

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zwischenraum |
| 合并 | 2026-08-22 |
| 分类 | Speculative Decoding(tier: major) |
| Label | bug, ready, v1, mrv2 |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数) |
| 链接 | https://github.com/vllm-project/vllm/pull/50272 |

## 一句话结论

修复 short_conv(LFM2/LFM2.5)模型 speculative decoding 在首个被拒 token 后输出损坏:conv 状态形状加 num_spec 余量并接入 spec-aware conv 更新。

## 动机

short_conv 模型上的 spec decode(ngram、EAGLE-3 等任何方法)在第一个 rejected token 后静默损坏输出:attention KV 会回滚,但 conv 滑窗保留了被拒 token 的激活。`mamba2`/GDN 已通过 spec-aware `causal_conv1d_update` 路径处理,short_conv 从未接入:①`short_conv_state_shape` 按 `conv_kernel - 1` sizing(无 num_spec 余量,镜像:`mamba2_state_shape` 会加 `num_spec`);②`GPUModelRunner._build_attn_group_metadata` 未向 `ShortConvAttentionMetadataBuilder` 注入 `num_accepted_tokens`。

## 优化点

`short_conv_state_shape` 增加 `num_spec` 参数,conv 状态形状改为 `conv_kernel - 1 + num_spec`;short_conv 层构造时读取 `vllm_config.num_speculative_tokens`;`forward_cuda` 的 decode 路径在 `num_accepted_tokens` 非空时改走 spec-aware `causal_conv1d_update`(此时 `state_indices_tensor_d` 形为 (num_decodes, 1+num_spec_tokens),同时传 `num_accepted_tokens` 与 `query_start_loc_d`,镜像 mamba_mixer2),否则保持非 spec 路径;`_build_attn_group_metadata` 与 `get_extra_attn_kwargs` 把 `ShortConvAttentionMetadataBuilder` 纳入 spec metadata 注入名单。

## 关键改动

```text
vllm/model_executor/layers/mamba/mamba_utils.py: short_conv_state_shape 增加 num_spec
vllm/model_executor/layers/mamba/short_conv.py: spec-aware causal_conv1d_update 分支
vllm/v1/worker/gpu/model_states/mamba_hybrid.py 与 vllm/v1/worker/gpu_model_runner.py: ShortConvAttentionMetadataBuilder 纳入注入名单
tests/kernels/mamba/test_short_conv_state_shape.py(新): spec 加宽 conv 状态形状回归测试
```

## 性能声明

无吞吐数据。模型供应商(Liquid AI serving team)E2E:LFM2.5-230M fine-tune + 训练的 EAGLE-3 drafter(~0.9/step acceptance),ROCm MI325X,v0.26.0 + 本补丁:修复前低并发 spec decode 在首个 rejection 后发散(doubled/fused tokens、偶发输出翻转;默认 cudagraph 下从 ~token 2 起退化为乱码);修复后 `--enforce-eager`、`PIECEWISE`、默认 `FULL_AND_PIECEWISE` 三种模式产出一致且正确的输出(固定种子内容探针)。

## 验证与回归风险

pre-commit(含 mypy-3.10)与新形状回归测试通过。风险:conv 状态每层增加 num_spec 行,改变 mamba/conv cache 形状与显存占用(缓存分配随之变化);`state_indices_tensor_d` 的非空断言路径仅在 spec decode 时触发。取代 #44296(rebase 保留原 commit 与作者署名,修复其 mypy 失败);与 #50146(同 target 修复 + 6 项 drafter 侧改动的打包)互补,本 PR 为最小、可独立合并的 target 侧修复。

## 关联

同分类 Speculative Decoding:#43091(MRV2 draft model)、#20401、#38174;取代 #44296,与 #50146 互补。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
