# PR #58655: [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ahmed-bsod |
| 合并 | 2026-09-29 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 5 个文件（_aiter_ops.py、kernels/mhc/aiter.py、layers/mhc.py、deepseek_v41/amd/model.py、测试，约 +250/-15 行） |
| 链接 | https://github.com/vllm-project/vllm/pull/58655 |

## 一句话结论

DeepSeek-V4.1 ROCm 路径的 delayed mHC seam 改走 aiter 新的 fused Triton kernel（2 次 launch 替代一串独立 kernel + 独立 RMSNorm），32k ISL 吞吐 +4.7%。

## 动机

ROCm DeepSeek-V4.1 的每个 mHC seam 此前由一串独立 aiter kernel 组成，RMSNorm 还要对 collapse 再做一次独立 pass。aiter 上游（ROCm/aiter#5824）新提供了整个 seam 的 fused kernel：split-K 主 kernel（post-mix、bf16 residual 写、gate projection、square sums）+ 每 token reduce（含 Sinkhorn 循环的 gates、RMSNorm 原地施加到 collapse）共 2 次 launch。gfx950 上值得把 vLLM 接上去。

## 优化点

(1) `_aiter_ops.py` 新增 `mhc_fused_post_pre_delayed_rms_norm` 包装：调用 aiter 的 `mhc_fused_post_pre_delayed_rmsnorm`，把整个 delayed seam（post block 折叠 sublayer_out、delayed pre block 的投影 + Sinkhorn 产生下一 seam 的 post/comb/pre gates、collapse 与携带的 pre-mix）连同 collapse 的 RMSNorm 一次完成，返回的 `layer_input` 已归一化。(2) `kernels/mhc/aiter.py` 注册 custom op `mhc_fused_post_pre_delayed_rms_norm_aiter`（mutates `residual_out`，附 fake impl 供 torch.compile）。(3) `layers/mhc.py` 的 `MHCPreDelayedOp.forward_hip` 增加首优先分支：`HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM`（gfx950 才为 True，能力探测 `_has_aiter_mhc_fused_post_pre_delayed_rms_norm`）且 `x is None` 且 `norm_weight` 非 None 且形状支持时走 fused op；不满足则回退原 unfused AITER/TileLang/参考路径。(4) `models/deepseek_v41/amd/model.py` 设 `fuse_seam_norm` 标志：fused kernel 覆盖的 seam 把 `attn_norm`/`ffn_norm` 权重以 `norm_weight/norm_eps` 传入 seam 调用，跳过独立 norm 调用；例外是 layer 0 的 attention seam（2-D embedding 广播入口，fused kernel 只收 4-stream residual，故保留独立 attn_norm，`fuse_attn_norm` 判断 `residual is None and x.dim() == 2`）。

## 关键改动

```text
vllm/model_executor/layers/mhc.py: forward_hip 增加 fused seam 首选分支；HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM 能力标志（仅 gfx950）
vllm/model_executor/kernels/mhc/aiter.py: 注册 mhc_fused_post_pre_delayed_rms_norm_aiter custom op
vllm/_aiter_ops.py: 包装 aiter mhc_fused_post_pre_delayed_rmsnorm
vllm/models/deepseek_v41/amd/model.py: fuse_seam_norm 时把 attn_norm/ffn_norm 折进 seam 调用并跳过独立 norm
tests/kernels/mhc/test_mhc_kernels.py: 新增 fused RMS norm seam 对拍（3 种 seam 形态 × split-K bucket）与路由 spy
```

模型侧 norm 折叠：

```python
residual, post_mix, res_mix, x, attn_pre = self.mhc_pre_delayed(
    residual, *pre_args, pre_mix=pre_mix,
    norm_weight=self.attn_norm.weight if self.fuse_seam_norm else None,
    norm_eps=self.attn_norm.variance_epsilon,
)
```

## 性能声明

测试条件：MI355X 类 gfx950（image `vllm/vllm-openai-rocm:nightly-rocm100`）、DeepSeek-V4.1-Flash TP4、`--moe-backend aiter`、DSpark 投机（5 tokens、probabilistic draft、block rejection）、`vllm bench serve` random dataset。结果：ISL 32k/OSL 32/并发 32 吞吐约 **+4.7%**；ISL 350k/OSL 350/并发 32 吞吐约 **+2.94%**。lm-eval（GSM8K 本地 yaml，num_concurrent 16，max_length 32768）"within range"。具体数字以 PR 内截图呈现，正文未给表格数值；未注明服务器轮次/重复次数。

## 验证与回归风险

`test_mhc_pre_delayed_rocm_aiter_fused_rms_norm`：num_tokens 1/2/7/128/1024 覆盖 split-K 配置 bucket，三种 seam 形态（常规 seam、Engram seam 已应用 post、draft 入口 seam 的 identity pre-mix）对 torch 参考对拍——gates 用 bf16 MFMA vs fp32 参考容差 5e-4/1e-3，collapse 的 layer_input 用 atol 2e-2/rtol 1e-2（与 aiter UT 同界，因 kernel 投影的是一次舍入后的 R'）。`test_mhc_pre_delayed_rocm_aiter_declines_unsupported` 更新路由断言：broadcast seam 不进 AITER/fused，fused norm 在 gfx950 上必须进 fused op（`took_fused == HAS_...`）。风险面：(1) fused kernel 仅在 gfx950 验证，能力探测按 `on_gfx950()` 门控，gfx942 走原路径；(2) collapse 路径数值容差较宽（2e-2），精度敏感场景需留意；(3) layer 0 广播入口与 fused kernel 输入形状不匹配，`fuse_attn_norm` 分支遗漏时会产生错误 norm；(4) 依赖 aiter 上游 #5824 的 kernel 存在与正确性，版本耦合；(5) 正文 Test Plan/Result checklist 未勾选，性能数字未附轮次设计。

## 关联

上游依赖：ROCm/aiter#5824（fused seam kernel）。同区域（DSv4.1 ROCm decode 优化）：#60731（fused_markov gfx942 移植）、#63891（fused Markov 放开 gfx950）、#63885（SWA decode metadata 单 launch）、#63181（Markov GEMV+sample epilogue 融合）。素材未列其他直接关联。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
