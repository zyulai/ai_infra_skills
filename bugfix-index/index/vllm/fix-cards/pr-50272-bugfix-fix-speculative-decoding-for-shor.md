# PR #50272: [Bugfix] Fix speculative decoding for short_conv (LFM2) models

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#50272](https://github.com/vllm-project/vllm/pull/50272) |
| 作者 | zwischenraum |
| 合并日期 | 2026-08-22 |
| labels | bug, mrv2, ready, v1 |
| 热度 | 👍 4 · 💬 34 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
short_conv(LFM2/LFM2.5)投机解码首个被拒 token 即静默损坏输出:conv 滑窗没回滚;接上 spec-aware `causal_conv1d_update` 路径并按 num_spec 加宽 conv state。

## 问题现象
对 `short_conv`(LFM2 / LFM2.5)目标模型跑投机解码(ngram、EAGLE-3、任意方法)时,输出在**第一个被拒 token 处**开始静默损坏:attention 的 KV 正确回滚,但 conv 滑动窗口保留了被拒 token 的激活。Liquid AI(LFM2.5-230M fine-tune + 训练的 EAGLE-3 drafter,~0.9/step 接受率,ROCm MI325X,v0.26.0 + patch)实测:不修时低并发 spec decode 在首个拒绝后发散(token 翻倍/融合,偶尔输出翻转;默认 cudagraphs 下从 ~token 2 起退化为垃圾)。

## 根因
正文披露两处:`mamba2` / GDN 早已通过 spec-aware 的 `causal_conv1d_update` 路径处理回滚,而 `short_conv` 从未接入:
1. `MambaStateShapeCalculator.short_conv_state_shape` 把 conv state 尺寸定为 `conv_kernel - 1`、不含 `num_spec`,窗口没有回滚余量(对照:`mamba2_state_shape` 会加 `num_spec`)。
2. `GPUModelRunner._build_attn_group_metadata` 从不为 `ShortConvAttentionMetadataBuilder` 注入 `num_accepted_tokens`,layer 永远看不到 accepted-token 计数。

## 修复方案
1. `mamba_utils.py`:`short_conv_state_shape` 增加 `num_spec: int = 0` 参数,conv state 改为 `conv_kernel - 1 + num_spec`;`short_conv.py` 的 `__init__` 记录 `self.num_spec = vllm_config.num_speculative_tokens` 并传入 `get_state_shape`。
2. `short_conv.py` 的 `forward_cuda`:decode 分支当 `num_accepted_tokens is not None` 时改走 spec-aware `causal_conv1d_update`(传 `num_accepted_tokens`、`query_start_loc=query_start_loc_d`、`max_query_len=state_indices_tensor_d.size(-1)`,镜像 `mamba_mixer2`),否则维持非 spec 路径。
3. `gpu_model_runner.py` `_build_attn_group_metadata` 与 `mamba_hybrid.py` `get_extra_attn_kwargs` 的 builder 判 isinstance 元组中加入 `ShortConvAttentionMetadataBuilder`,使其同样注入 spec 元数据。关键 diff 摘录:
```python
-        conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1)
+        conv_state_shape = cls._orient_conv_shape(conv_dim, conv_kernel - 1 + num_spec)
```
相对 #44296:rebase 到当前 main、修掉阻塞 pre-commit 的 mypy 失败(`assert state_indices_tensor_d is not None`)、新增 spec 加宽 conv state shape 的回归测试 `tests/kernels/mamba/test_short_conv_state_shape.py`。

## 影响与回归风险
- 影响窗口:正文未列具体版本范围;影响 short_conv(LFM2/LFM2.5)模型的投机解码。
- 回归面:conv state 形状变化会使该类模型的 KV/state 缓存尺寸增大(每层多 `num_spec` 列),涉及 `short_conv_state_shape` 的所有调用方与缓存分配;`forward_cuda` 仅在 spec 元数据存在时走新分支,非 spec 解码路径不变;`mamba_hybrid`/`gpu_model_runner` 的 isinstance 列表扩展只影响 ShortConv builder。E2E 验证:`--enforce-eager`、`PIECEWISE`、默认 `FULL_AND_PIECEWISE` 三种模式在固定 seed 内容探针上输出一致且正确;pre-commit(含 mypy-3.10)通过。

## 关联
- #44296(被本 PR 取代的前一 PR,原提交与作者署名保留)
- #50146(捆绑同一 target 侧修复 + 6 个 drafter 侧改动的另一 PR;本 PR 是最小可独立合并的 target 侧修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
