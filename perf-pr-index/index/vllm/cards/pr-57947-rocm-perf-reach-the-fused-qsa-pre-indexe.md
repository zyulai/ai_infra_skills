# PR #57947: [ROCm][Perf] Reach the fused QSA pre-indexer from the AMD path

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-10-05 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, ci/build, qwen |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾标注 "1 file(s) dropped from diff for size"(见 PR 完整 diff),diff 保留 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/57947 |

## 一句话结论

ROCm 上把 QSA pre-indexer 的 37 次 launch 换成单次,MI355X TP2 output throughput +3.1%~+9.3%。

## 动机

ROCm 路径上 Qwen4Exp 的 QSA indexer 此前走 unfused 的 Gemma RMSNorm / MRoPE / compress / cache-write 链,indexer prepare 需多次 kernel launch。PR 正文给出 8k/1024 c4、TP2 的 decode trace:prepare 相关 kernel launch 密集。review 要求把 NVIDIA 的 fused kernel 复制进 AMD 路径而非共享到 `common/`,以便两家厂商各自改 kernel 不必重新验证对方。

## 优化点

把 NVIDIA `amd/ops/qsa_pre_indexer.py` 的 fused pre-indexer kernel 复制为 AMD 路径实现,让 `QSAIndexer` 在配置满足 `supports_fused_pre_indexer`(RoPE 风格、head dim、KV heads、compress ratio 的形状判定)时走 fused 分支,不合格配置继续走 unfused 链(兼作测试的数值参照)。融合后 indexer prepare 每个 QSA layer 每 step 只 1 次 launch,替代 37 次;8k/1024 c4 decode trace 下全部 kernel launches 由 195,384 降到 156,228(−20.0%)、device time −6.1%。与 NVIDIA kernel 仅一处差异:用两个互补 masked loads 选择 state 来源,替代 ROCm Triton 拒绝的 pointer-typed `tl.where`(`'tt.addptr' op failed to verify...`),内存流量不变;两处都保留 `tl.multiple_of` 以保证 pooling reduction 顺序与 unfused 路径逐位一致。tile/warp 常数沿用 NVIDIA 32-lane 调优,在 MI355X 上扫过 TILE_T_Q/TILE_H_Q/num_warps 未找到更快配置。eligibility 谓词仅在 init 跑一次,热路径无 CPU/GPU sync;NVIDIA 文件未改动。

## 关键改动

```text
vllm/models/qwen4_exp/amd/ops/qsa_pre_indexer.py: 新增 AMD 路径 fused pre-indexer kernel(自第一个 @triton.jit 起与旧 head 的 common/ops/qsa_pre_indexer.py 字节一致),含 supports_fused_pre_indexer 判定
vllm/models/qwen4_exp/amd/indexer_qsa.py: QSAIndexer 接入 fused 分支,向 fused kernel 传 raw_key_cache.kv_cache(全宽,使 MRoPE position tail 被写出)
tests/models/qwen4_exp/test_qsa_pre_indexer_amd.py: 新增,8 个 bf16 配置对比 fused kernel 与 AMD unfused 链(FP8 用例因 AMD indexer cache 仅 bf16 跳过)
tests/models/qwen4_exp/test_qsa_amd.py: 覆盖 fused 分支替换、ineligible 配置回退、5 配置 eligibility 谓词
.buildkite/test_areas/models_basic.yaml: AMD models_basic 步骤加入新测试文件
```

## 性能声明

Serving(MI355X、TP2、`amd/Qwen3.8-Flash-Next-Quark-MXFP4`,paired A/B,control → candidate,`vllm serve ... --tensor-parallel-size 2 --max-model-len 262144 --no-enable-prefix-caching --trust-remote-code`,`vllm bench serve --dataset-name random`):ISL/OSL/Conc = 60,000/600/4 时 output tok/s 118.19 → 121.86(+3.1%)、mean TPOT 19.34 → 18.39 ms(−4.9%);60,000/600/16 为 152.13 → 154.28(+1.4%);8,192/1,024/4 为 367.23 → 401.53(+9.3%)、TPOT 9.55 → 8.62 ms(−9.7%);8,192/1,024/16 为 846.90 → 898.96(+6.1%)。PR 正文称 +3.1% 到 +9.3% output throughput,最大在 8k/1024 c4;TTFT 基本持平(decode-only 改动,如 1,385 → 1,378 ms)。Accuracy GSM8K 5-shot(MI355X):0.9636 ± 0.0052 → 0.9689 ± 0.0048,称无精度变化;NVIDIA 路径 unchanged(diff 中无 NVIDIA 文件)。

## 验证与回归风险

单元测试(gfx950):`pytest -v -s tests/models/qwen4_exp/test_qsa_amd.py tests/models/qwen4_exp/test_qsa_pre_indexer_amd.py` 24 passed、8 skipped,跑在 AMD `models_basic` step。fused kernel 对 8 个 bf16 配置(MRoPE/text RoPE、1-D/2-D positions、有/无 position cache、state size 4/8、37-token 新请求、跨 tile switch 的 4097-token batch)与 unfused 链对比:query 与 compressed cache 在 bf16 容差内、raw state cache 逐位一致;重构成 AMD 路径后 12 组输入逐位一致、时间比 0.98–1.00。风险面:仅 ROCm/Qwen4Exp 路径,AMD indexer cache 为 bf16-only 故 FP8 用例跳过;launch 常数在 MI355X 上扫过,正文未列其他 gfx 的逐一验证。

## 关联

PR 正文明确:后续 #59732 将把 #57097 的 main-attention fusion 移植到这份 AMD 副本。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
