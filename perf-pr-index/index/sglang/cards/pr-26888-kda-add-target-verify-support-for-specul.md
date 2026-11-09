# PR #26888: [KDA] Add target_verify support for speculative decoding

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yyq0210 |
| 合并 | 2026-07-25 |
| 分类 | Speculative Decoding |
| Label | documentation, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 含 6 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/26888 |

## 一句话结论

给 KDA 内核与 dispatcher 增加 target_verify 路径,使 KimiLinearForCausalLM 首次支持 EAGLE 投机解码,补齐与 GDN 的功能对等。

## 动机

姊妹线性注意力后端 GDN 已支持完整 EAGLE 投机解码,而 KDA 缺少 target_verify 支持,Kimi-Linear 系列模型无法受益于投机解码加速。本 PR 为 KDA 做针对性适配:fused sigmoid recurrent kernel 增加 `is_kda=True` 门控;convolution state 需要从 `(conv_width, qkv_dim)` 转置为 `(qkv_dim, conv_width)` 以匹配 `causal_conv1d_update` 期望;完整 conv 权重作用在合并的 `mixed_qkv` 上(与 KDA decode 行为一致)。

## 优化点

三层改动。内核层:`target_verify` 方法写入 KDA kernel 与 dispatcher,基于 `fused_sigmoid_gating_delta_rule_update`(is_kda=True),每 draft token 的中间状态写入 `intermediate_states_buffer`,committed SSM state 不被原地修改(`disable_state_update=True`),供验证后按 accept 长度回滚提交。后端层:`KDAAttnBackend.forward_extend` 增加 `is_target_verify` 分支,处理 conv state 转置与中间 SSM 状态缓存。模型层:`kimi_linear.py` 中 gate 的 unflatten 逻辑放宽——decode 与 target_verify 时 gate 激活留在内核内做,仅 prefill 才需要预展开。此外在三个 eagle worker 变体中注册 KDA(kimi_linear_config)到 mamba verify 检测;`SpeculativeState` 增加 `_detect_conv_window_axis` 处理 GDN/KDA 两种 conv-state 轴布局(混布时不共用重叠 conv-window buffer),并修复 KDA 的 window-major 重叠视图(旧实现把 draft-step 轴别名到 channel 轴)。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kda_backend.py: forward_extend 增加 target_verify 守卫(extend_prefix_lens 非空断言)与 verify 分支
python/sglang/srt/models/kimi_linear.py: gate unflatten 仅在非 decode 且非 target_verify 时进行
python/sglang/srt/mem_cache/memory_pool.py: SpeculativeState._detect_conv_window_axis 支持轴检测与混布报错
test/manual/test_kda_target_verify.py: 内核级等价性测试(fp32/bf16 对 N 次顺序 decode)
test/registered/unit/spec/test_ngram_mamba_verify_update.py: ngram mamba verify 更新与 conv-window 布局回归测试
```

## 性能声明

PR 未注明端到端性能数字(无吞吐/延迟 benchmark)。正确性数字:内核级 target_verify 与 N 次顺序 decode 等价——fp32 max diff 0.0(精确匹配)、bf16 max diff 1.862645e-04 / mean diff 9.756498e-06;中间状态 fp32 全步全请求精确匹配;SSM state 原地变化为 0.0。E2E(Kimi-Linear-48B-A3B-Instruct,TP=2)normal inference / prefix caching(cached 64 tokens)/ 8 请求 batch 推理全部通过。

## 验证与回归风险

Mock fixture 单测 7 项通过(含 `test_runner_mode_eagle_verify_cases`:chain topk=1、tree topk=2、`frozen_kv_mtp`、`dflash`、`ngram`;CUDA graph verify;draft extend)。新增 conv-window 布局回归测试明确保留旧 buggy 视图作对照,防止复发。风险面:KDA conv-state 轴约定与 GDN 不同,重叠 conv-window 布局曾出现轴别名 bug(本 PR 已修,但该区域在后续 PR #30113 中再次被确认需 KDA 保持 dense 布局);target_verify 只覆盖 Triton 路径,FlashInfer/其他内核后端的 verify 由后续 PR 引入。

## 关联

同区域:PR #30113(FlashInfer SM100 KDA decode + MTP target_verify 后端,声明本 PR 是 Triton 参考)、PR #33722(FlashInfer KDA MTP fused-accept state advance)。GDN 对应能力为 KDA 的对齐目标。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
