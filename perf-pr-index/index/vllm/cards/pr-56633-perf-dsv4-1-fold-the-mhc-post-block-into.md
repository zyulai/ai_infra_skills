# PR #56633: [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zyongye |
| 合并 | 2026-09-14 |
| 分类 | Speculative Decoding \| tier: major |
| Label | ready, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾说明 4 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/56633 |

## 一句话结论

把 DSv4.1 的 mHC post 折进 delayed pre projection，GB300 小 batch decode 约 1.3x。

## 动机

DeepSeek-V4.1 每个子层接缝依次跑 mHC post block、重读 post 刚写过的 residual stream 的 split-k projection、以及下一子层的 delayed pre epilogue，共三次开销。DSv4 的 `mhc_fused_tilelang` 已在寄存器内算出更新 residual 并直接喂给 pre-norm GEMM，省掉中间那次读；但 V4.1 用上一子层的 pre-mix 折叠本层输入并交还下一层，而 DSv4 的 fused wrapper 既不接收 incoming pre-mix 也不返回它，故 V4.1 无法复用。

## 优化点

新增 `mhc_fused_post_pre_delayed_tilelang`：跑已有 fused kernel 后接 delayed pre epilogue（`MHC_PRE_NORM_KERNEL`，`use_pre_mix_in` / `save_pre_mix`），返回 `(residual, post_mix, comb_mix, layer_input, next_pre_mix)`；超出 fused kernel 的 token 范围回退到原 `mhc_post_tilelang` + split-k GEMM，只改小 batch 路径的形状。Attention seam 仅在注入 Engram 时不融合（Engram 需读 post 与 pre 之间的 residual），FFN seam 总融合，DSpark draft 复用 `DeepseekV4DecoderLayer` 因此两者都继承。`MHCPreNormKernel.get_warmup_keys` 增加 `extra_splits` 供 decoder 注册，避免首次 decode 在 capture warmup 内 JIT。继承未测的 tile/split-k 经 sweep 改为两段 band（`tile_n` 在 16 以下取 2、到 32 取 6，`n_thr` 128），cutoff 定在 32——因 `compute_mhc_pre_num_splits` 把估计分桶到 {1, 4, 16}，128 token 以下比最优差 1.14–1.19x，正文把该修正明确留到独立 PR。另两个 commit 用 `capture_previous_aux` / `write_aux` 把 aux hidden states 的 `mean(dim=1)` 折进 collapse pass，去掉重复的 standalone post。

## 关键改动

```text
vllm/model_executor/kernels/mhc/tilelang.py: 新增 mhc_fused_post_pre_delayed_tilelang 与 tuned split config 的 dispatch
vllm/model_executor/kernels/mhc/tilelang_kernels.py: fused kernel 返回 5-tuple，write_aux 在 collapse pass 内累积 mean
vllm/model_executor/kernels/mhc/warmup.py: get_warmup_keys 接收 extra_splits，注册 JIT warmup key 并拒绝不能整除的形状
tests/kernels/test_mhc_kernels.py: 新增 fused post+pre bit-identical、opcheck、CUDA-graph 及 aux 用例
tests/kernels/test_mhc_jit_warmup.py: 固定 tuned band 的 compile key 与 launch config 一致
```

## 性能声明

GB300、`hc_mult=4`、`hidden_size=5120`、CUDA-graph device time per seam：tuned fused 在 1/2/4/8/16/24/32 token 为 8.02/8.00/8.83/8.84/9.85/10.47/11.47 µs，unfused 5120 为 12.74/12.52/12.72/12.92/13.32/13.32/13.37 µs。对"按自身估计分 split"的修正 baseline，fused 在 1–32 token 依次 1.31x/1.35x/1.28x/1.29x/1.17x/1.10x/1.03x，48 token 0.86x，正文结论"1.3x at decode widths, 1.17x at 16, break-even at 32"。与 Mega-mHC（#56255）对比（DeepGEMM 2.8.0、GB300、hc_mult=4、hidden 5120）：1–16 token 快 1.08–1.27x，48+ 慢 1.18–1.20x。folded mean 在 GB300 `[T,4,5120]` bf16、4096 token 处省 68.70 µs（209.32→140.63 µs）。端到端 DeepSeek-V4.1-Flash、TP4 GB300、gsm8k 5-shot、n=1319：本 PR 0.9242/0.9325/0.9204，unfused baseline 0.9348/0.9249，差异落在噪声内。

## 验证与回归风险

`pytest tests/kernels/test_mhc_kernels.py -v` 得 `164 passed, 23 skipped in 53.80s`（GB300），`pre-commit run --all-files`（ruff / ruff-format / mypy-3.10）clean。新测试断言 residual 与 collapsed layer input 与分离 kernel bit-identical、系数对 FP32 参考在 1e-5 内，含 `torch.library.opcheck` 与 CUDA-graph capture/replay；DSpark 用真实验证、8 个固定 prompts 得到 byte-identical 文本与一致 acceptance counter。数值上 fused TileLang 投影寄存器内的 FP32 post 结果，而 Mega-mHC / unfused 投影其 BF16 舍入，8 token 对 FP32 参考的系数 max abs error 为 9.25e-4 / 1.78e-3 对 1.98e-4 / 3.46e-4，属继承 DSv4 的差异。风险集中在 Engram 注入接缝、每 rank 最后一层、不能整除 tile 的形状（改为 fallback）。

## 关联

正文明确列出：#56255（Mega-mHC，DeepGEMM，大 batch 更快，互补）、#56513（ROCm AITER 同思路）、#56611（系数 GEMM 与 attention/FFN 重叠，env-gated）、#55575（DeepSeek-V4 aux hidden states 前序，本 PR 借用其思路）。作者建议后续写一个按 token 数在两者间选择的 dispatcher。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
