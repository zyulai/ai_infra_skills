# PR #31041: [Spec] Add LFM2 and LFM2-MoE DSpark speculative decoding support

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @tugot17 |
| 合并 | 2026-08-31 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | +422/-23,7 files(PR 正文注明;pack 未含 diffstat) |
| 链接 | https://github.com/sgl-project/sglang/pull/31041 |

## 一句话结论

为 LFM2/LFM2-MoE hybrid 目标接入 DSpark 投机解码,单 H100 上数学/代码类任务加速最高 2.42x,接受长度近 block-7 上限。

## 动机

DSpark 投机解码(#30261)合入 main 后,尚不支持 LFM2 家族(ShortConv + attention 混合架构)目标。LFM2 draft checkpoint 以 interleaved RoPE 导出,而现有 fused RMSNorm+RoPE KV 写出内核只处理 neox 旋转,这是接入的硬阻塞——PR 实测用 neox 方式 serving 会使接受长度从约 5.1 塌缩到约 1.9。本 PR 使 dense LFM2 与 MoE LFM2-MoE 均可作为 DSpark 目标运行。

## 优化点

两个提交:(1) fused-KV 内核增加 `IS_NEOX` constexpr,在 neox 配对 `(i, i+D/2)` 与 interleaved/GPT-J 配对 `(2i, 2i+1)` 之间选择,cos/sin cache 布局不变,对既有 neox draft 无影响;(2) 目标侧支持:注册 `Lfm2DSparkDraftModel`(薄子类,便于未来加 ShortConv 层);在 `lfm2.py`/`lfm2_moe.py` 增加 aux-hidden 捕获与 ShortConv block 的 verify/rollback(TARGET_VERIFY 下走 triton `causal_conv1d_update`,把每步 conv 窗口记入 speculative tape 以便回滚);DSpark worker 在 verify 后做 hybrid-state commit(镜像 DFlash worker,解码保持无损);draft KV injector 改走 fused-KV 路径;将 LFM2 架构标记为可用 spec-v2 回滚所需的 `extra_buffer` mamba radix 策略。

## 关键改动

```text
python/sglang/kernels/ops/speculative/dspark/fused_kv_write.py: _fused_kv_norm_rope_write_kernel 增加 IS_NEOX 旋转配对
python/sglang/srt/models/dspark.py: 放开 neox-only 门,按 rotary 配置传 is_neox_style
python/sglang/srt/models/lfm2.py: shortconv_target_verify 与 register_shortconv_verify_buffers;aux hidden 捕获
python/sglang/srt/models/lfm2_dspark.py: 新增 Lfm2DSparkDraftModel 架构
python/sglang/srt/models/lfm2_moe.py: MoE 目标同样的 verify/capture 支持
```

关键片段(interleaved RoPE 配对):

```python
if IS_NEOX:
    off1 = half_ar
    off2 = HALF + half_ar
else:
    off1 = 2 * half_ar
    off2 = 2 * half_ar + 1
cos = tl.load(cos_sin_ptr + pos * D + half_ar).to(tl.float32)
```

## 性能声明

测试条件:1xH100 80GB,bs=1,temperature 0,block size 7,对比同一 server 去掉 spec flags;数字为 accept 长度 / 加速比。

| dataset | LFM2.5-1.2B + 5L | LFM2.5-8B-A1B + 3L |
|---|---|---|
| math500 | 4.82 / 2.10x | 5.42 / 2.21x |
| aime | 4.36 / 1.95x | 6.20 / 2.42x |
| humaneval | 4.41 / 1.93x | - |
| mbpp | 4.45 / 1.95x | 4.80 / 1.96x |
| gsm8k | 3.73 / 1.49x | 3.51 / 1.24x |
| mtbench | 2.35 / 1.05x | 5.87 / 2.32x |

下限是小 dense 目标上的短对话(约 1.05x),上限是 MoE 上的长推理(2.3-2.4x,接受长度接近 block-7 极限)。所有可评分 bench 准确性与 baseline 持平。interleaved RoPE 非可选:neox 下接受长度约 1.9 vs interleaved 约 5.1。

## 验证与回归风险

在 1xH100 上按上述 bench 做 dense + MoE、draft 深度 2L/3L/5L 的 serving 验证。MoE ShortConv decode 复用与 dense 相同的 CUDA `causal_conv1d_update` 内核(曾试 triton 版,确认接受数完全相同 5.936 后移除)。公开 draft checkpoint 发布于 HF(dense 5L 与 MoE 3L),作者表示愿在 `test/registered/spec/dspark/` 添加门控 CI 测试——PR 本体未含新自动化测试。风险路径:既有 DSpark/DSparkDraftModel 用户的 neox 行为(IS_NEOX 默认 True 保持兼容)、LFM2 hybrid 状态(ShortConv tape + mamba `extra_buffer`)的 rollback 正确性、DFLASH aux capture 路径。

## 关联

前序:#30261(DSpark 本体)、#30776(本 PR 取代的旧分支 PR)。配套:#30780(LFM2-MoE serving parity,与 DSpark 无关,建议搭配使用)。同分类:DSpark/DFlash 投机解码相关 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
