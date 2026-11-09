# PR #65874: [Kernel][Perf] Fused GDN MTP: Support DSpark K=8 and use idle SMs at small batch sizes

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tomasruizt |
| 合并 | 2026-11-06 |
| 分类 | Speculative Decoding（tier: major） |
| Label | ready, dflash |
| Diff | 素材未提供 diffstat（增删行数与文件数缺失） |
| 链接 | https://github.com/vllm-project/vllm/pull/65874 |

## 一句话结论

fused GDN MTP decode kernel 支持 9 token（DSpark K=8）并加 Split-V 用满空闲 SM：Qwen3.6-35B K=8 单请求吞吐 B300 +18%、H100 +13%，kernel 延迟减半。

## 动机

DSpark K=8 每请求验证 8 个 draft 加 1 个 bonus token，`state_indices` 宽度为 9，而 `gdn_decode_post_conv_mtp_kernel` 的 `kMaxMtpTokens=8`，导致 K=8 在 main 上每个 GDN 层静默回退到慢的 Triton delta-rule + gated RMSNorm 路径。同时小 batch 下 kernel 每个 (request, value head) 只发一个 CTA（如 bs=1 时 32 个 CTA 落在 148 个 SM 上），大量 SM 闲置。另一点：单纯提高 token 上限不够——kernel 原来一个 warp 处理一个 token、只有 8 个 warp，第 9 个 token 会被跳过。

## 优化点

- 9-token 支持：`kMaxMtpTokens` 8→9，warp 以 stride 方式跨 token 迭代（`kTokenPasses` 限界、编译器可展开）；Python 侧 `MAX_FUSED_GDN_MTP_TOKENS` 同步为 9；9 token + fp32 state 的静态共享内存约 45.6 KB，仍在 48 KB 静态限制内。
- Split-V（SM90+）：当 split 后的 grid 仍能在一个 wave 内放下（2 CTA/SM）时，把一个 value head 的 V 行按 `blockIdx.z` 拆到 2 或 4 个 CTA 的 cluster，各 CTA 加载/更新自己的 state slice；gated RMSNorm 的平方和经 DSMEM（`cluster.map_shared_rank`）跨 CTA 求和并做 final cluster barrier。`choose_split_v` 是 host 侧启发式（SM 数、无调参表、每次 launch 查两个 device attribute 无 GPU sync）；HV=32 时 B300/H100 上 ≤2 请求 split=4、4 请求 split=2、≥8 请求 split=1（保持单 pass epilogue 不变）。SM80–89 编译为 trap stub 以省二进制。
- 副产物：不 split 时 kernel 也略快（PR head 对 token 循环做了 unroll）；K=7 在 c1 也吃到 Split-V 收益。

## 关键改动

```text
csrc/gdn/fused_gdn_decode_kernel.cu: kMaxMtpTokens=9 + warp stride + SplitV 模板（DSMEM 归约、cluster launch）
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py: MAX_FUSED_GDN_MTP_TOKENS=9
tests/kernels/fla/test_fused_gdn_post_conv.py: 9-token 与 4/16 请求（split=2/1 路径）用例
```

```cpp
constexpr int kMaxMtpTokens = 9;
constexpr int kTokenPasses = (kMaxMtpTokens + kWarps - 1) / kWarps;
...
for (int pass = 0, t = warp; pass < kTokenPasses && t < num_tokens;
     ++pass, t += kWarps) {
```

## 性能声明

E2E（Qwen3.6-35B-A3B，AIPerf，SPEED-Bench throughput 1k，median；B300 用 bf16、H100 用 FP8 checkpoint；main=`5225a1f17f`）：DSpark K=8 输出吞吐 c1 +18.0%（476.0→561.5 tok/s，TPOT 2.01→1.72 ms）、c2 +16.8%；c8 起 +6.4% 至 +8.5%（B300，至 c128），H100 c1 +12.7%、c8-c32 +1.9% 至 +5.9%。K=7（仅 Split-V 生效）c1 +7.9%，c8 起 -1.1% 至 -0.3%。MTP3 不回退：c1/c2 +2.4%/+2.5%，c4 起在会话方差内（H100 c16 -2.2%）。

Kernel microbench（`gdn_decode_post_conv_mtp_kernel`，9 token/请求，fp32 state，CUDA graph replay，median of 3）：B300 c1 22.3→10.2 µs（-54.2%）、c2 22.6→13.5 µs；H100 c1 26.2→14.3 µs（-45.3%）；c8 起 0.4–0.9% 更快。MTP3（4 token）microbench：B300 c1 12.1→6.7 µs（-44.6%）、H100 c1 13.7→9.3 µs（-32.1%）。

正确性：GSM8K 与 acceptance length 与 main 一致（DSpark K=8 B300 0.795/0.799 vs 0.799/0.792；acceptance 2.83–2.85 不变）。

## 验证与回归风险

- 测试：B300(sm103)/H100(sm90) 上 `tests/kernels/fla/test_fused_gdn_post_conv.py`、`tests/kernels/mamba/test_gdn_fused_mtp.py`、`tests/model_executor/test_qwen_triton_warmup.py` 共 139 passed / 2 skipped；compute-sanitizer memcheck/racecheck/synccheck 0 错误；新增 4/16 请求用例覆盖 split=2/1 路径；CI 中 kernels/fla 跑 B200、kernels/mamba 跑 H200 MIG，均会走 split 路径。
- 风险与限制（PR 自述）：split 选择是 host 启发式（无调参表）；SM80–89 永不 split；SM100/SM120 走同一路径但未测试；每个 kernel 实例化现在有 3 个 split 变体——7 个目标 arch 单次 nvcc 编译从 36 s 增至 71 s，目标文件 0.41→0.90 MB；混合 prefill+decode batch 仍走 unfused 路径；K=7 在 c8+ 有 -1.1% 以内的轻微回退。

## 关联

- 相关 PR：#53070（同为 9-token 支持，采用独立 width-9 模板特化的另一方案，8/31 后无活动；本 PR 原地提升上限，可按需 rebase）；#54612（加速 Qwen3-Next 的 Triton spec-decode recurrence，即 K=8 在 main 上的回退路径）；benchislett/vllm#8（draft，SM107 上 12/16 token 宽 kernel，与本 PR 互补）。
- 同分类：本批 PR #60695（Mooncake EAGLE proof）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
