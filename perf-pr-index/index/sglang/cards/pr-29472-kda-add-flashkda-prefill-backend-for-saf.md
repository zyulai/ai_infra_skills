# PR #29472: [KDA] Add FlashKDA prefill backend for safe-gate KDA linear attention

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2026-07-02 |
| 分类 | Attention / Kernel (tier: major) |
| Label | dependencies, run-ci, run-ci-extra, linear-attention, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None/None files);diff 可用,涉及 5 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29472 |

## 一句话结论

引入 MoonshotAI FlashKDA 融合 CUTLASS 内核作为 safe-gate KDA 线性注意力的可选 prefill 后端,较 Triton chunk_kda 快 1.3–4.6×,opt-in 且透明回退。

## 动机

KDA(Kimi Delta Attention)是 per-channel 门控 delta-rule 线性注意力。参考实现 FLA 的 Triton `chunk_kda` 把分块 delta rule 跑成一串内核(chunk=64、intra-chunk rescaling、高精度累加),launch 与带宽开销大。FlashKDA(MoonshotAI,2026-04 开源)将整个分块递推融合为单个 CUTLASS 内核:chunk=16、recurrent state 存 bf16、16×16 intra-chunk 逆用 fp16 Neumann 级数、base-2 exp、tensor-core tiles,换来 1.3–4.6× 加速。本 PR 只落地内核后端与 dispatch/回退,是 safe-gate KDA 支持的前置基础设施:opt-in、默认关、依赖可选包、不能跑时透明回退 Triton,对现有部署零风险。

## 优化点

1. **safe gate 是硬前提**:FlashKDA 的速度来自低精度(bf16 state、fp16 逆)小 chunk 无 rescaling 递推,仅在门的 per-step log-decay 有下界时数值有效。safe/有界门 `a = lower_bound·sigmoid(exp(A_log)·(g+bias))`,`a ∈ (−5, 0]`,chunk=16 时 `cumsum ∈ [−80, 0]`,`exp(cumsum)` 落在 bf16 可表示范围内(无需 rescale),`(I+A)` 良态(逆元素 ≤1,f fp16 安全)。标准无界门 `a = −exp(A_log)·softplus(...)` 无下界,只能走 Triton。有界门烙在 checkpoint 里(safe-gate 模型 `A_log ≈ 0`),推理期不可切换;代价是记忆不能完全归零(保留 ≥ e^−5 ≈ 0.67%)。
2. **接入方式**:`--linear-attn-prefill-backend flashkda` 选择;`FlashKDAKernel.extend` 把 raw 张量 + `A_log`/`dt_bias`/`lower_bound` 传给 `flash_kda.fwd`(L2 norm/beta sigmoid/门激活全部在内核内融合;beta 需 `torch.logit` 反演回 logits);状态布局 `[N, H, V, K]` 与现有 KDA pool 一致,最终状态经 `ssm_states[cache_indices]` 原地写回。
3. **回退条件**:`lower_bound is None`(无界门)、最短序列 < 64(chunk size)、最长序列 > 2048(Triton 反超的 crossover)、以及投机解码(target_verify/draft_extend_v2——FlashKDA 原地提交状态,会破坏 draft rollback);序列长度优先读 CPU 侧 `extend_seq_lens_cpu` 避免逐层 GPU→CPU 同步。
4. **防误配**:显式 `--linear-attn-decode-backend flashkda` 直接报错(prefill-only);从 `--linear-attn-backend flashkda` 继承时 decode 自动落到 triton。`flash_kda` 懒加载,未选后端不依赖。

## 关键改动

```text
python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py: 新增
  FlashKDAKernel(prefill/extend-only),包装 flash_kda.fwd,含回退判定
  _should_fall_back 与 Triton fallback(穿透 A_log/dt_bias/lower_bound)
python/sglang/srt/layers/attention/linear/kda_backend.py: prefill dispatch
  增加 flashkda 分支;forward_extend 透传 extend_seq_lens_cpu 与
  is_spec_decode 门
python/sglang/srt/layers/attention/linear/utils.py: 新增
  LinearAttnKernelBackend.FLASHKDA 与 is_flashkda()
python/sglang/srt/server_args.py: flashkda 加入
  LINEAR_ATTN_KERNEL_BACKEND_CHOICES;decode 侧守卫
test/registered/attention/test_kda_prefill_flashkda.py: 对 Triton 参考的
  正确性测试(注册但在公共 CI 中 disabled)
```

## 性能声明

PR 正文,未独立核实。FlashKDA vs Triton `chunk_kda`,raw kernels、输入 fresh clones:

- **H20-3e (sm90)**:B=1/T=128 **5.2×**;B=1/T=512 2.2×;B=1/T=2048 1.24×;B=8/16/32、T=512 分别 2.0×/2.1×/2.05×。
- **B200 (sm100)**:B=1/T=128/H=32 **4.61×**;B=1/T=512 2.60×;B=1/T=2048 1.27×;B=16/T=512 2.69×;B=32/T=512 2.65×;B=1/T=8192/H=64 1.73×、H=96 2.23×。
- 总体声明 1.3–4.6×;B200 T=8192 结果(1.73×/2.23×)与 MoonshotAI 公开的 `BENCHMARK_GB200.md`(1.70×/2.31×)交叉吻合。

## 验证与回归风险

- 正确性对生产 Triton `chunk_kda`(safe-gate 路径,bf16,H=32,K=V=128,varlen,每路径克隆输入——`chunk_kda` 会原地改 g/v):single-token 输出 cos 1.0000、max|diff| 3e-4;全序列输出 cos 0.985;最终 recurrent state cos 0.9999。残差归因于 bf16 跨实现累加(非 bit-exact,与 FLA 对该内核的处理一致)。
- 新增注册测试 `test_kda_prefill_flashkda.py`(safe-gate 匹配、无 lower_bound 回退、spec verify 回退),但在公共 CI 中 disabled(flash_kda 不在公共 runner 镜像,仅 Ant 内部 PyPI);本地/内部 CI 可跑。CI 状态(pack 记录):Base 与 Extra 均通过。
- 要求:SM90+、可选 `flash_kda` 包(`pip install sglang[flashkda]` 或源码装)、bf16、K==V==128、HV==H(无 GVA)、safe gate;仅 prefill/extend。
- 回归面:默认关闭、命中不了条件即回退 Triton,对现网影响小;风险点在回退判定漏网(spec decode 门已由内核层与 server_args 双重防护)与 beta logit 反演的 clamp 精度。

## 关联

同 batch 的 #30169(GDN/KDA SM100 CuteDSL prefill state I/O 融合,同属线性注意力 prefill 内核方向)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
