# PR #37413: [AMD][DSV4] feat: enable fp8 two-pool unified_kv on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103 |
| 合并 | 2026-09-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | quant, deepseek, sgl-kernel, run-ci, jit-kernel, run-ci-extra, memory-pool |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(16 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/37413 |

## 一句话结论

gfx950 上 DeepSeek-V4 unified_kv 的 nope 半改存 fp8(E8M0 per-tile scales)、rope 保持 bf16,每 token 每层 KV 行 1024 B→640 B,同显存 KV 容量 1.50x。

## 动机

`unified_kv` 布局下 DSv4 的整条 latent(`qk_nope_head_dim` 448 + `qk_rope_head_dim` 64)都以 bf16 存储,每 token 每层 1024 B;而 nope 半并不需要这个精度。把 nope 存成 fp8 加 per-tile E8M0 scales、rope 留 bf16,行宽降到 640 B(512 B fp8 nope 定长 stride + 原 128 B bf16 rope),同显存可驻留约 1.5 倍 token,更长上下文与更高并发能装进同一节点。默认关闭,由 `SGLANG_DSV4_UNIFIED_KV_FP8` 开启,要求 gfx950 且隐含 `is_unified_kv_triton()`;其他架构告警并回退 bf16,现有路径不变。

## 优化点

单池变双池(nope 池 512 B fp8 行含 inline E8M0 scales、pad 到 2 的幂 stride;rope 池 128 B bf16),随后改造池的每个写者与读者:layout/sizing 层新增 `DSV4_FP8_NOPE_ROW_BYTES=512` 常量与 `check_two_pool_pair` 前置配对校验(两个 kernel 都不检查行索引越界,配对检查必须在 launch 前);写者覆盖 SWA scatter、fused QK norm+RoPE store(两池由一个行索引寻址)、compressor norm+rope store 与 prefill/target-verify 使用的 packed ring store,JIT 模板新增 `forward_fp8_2buff` 入口与 `kFp8TwoPool` 编译期分支;读者侧 decode 走 aiter `mla_a8w8` v4 nm asm kernel、prefill 走 aiter opus kernel。另做 decode 侧开销削减:去掉空段 `masked_fill_`、缓存只依赖 token 数的 `qo_indptr`、跳过不需要的 `kv.contiguous()`、并在 >40 token 时把 `num_kv_splits` 封顶为 4(aiter 占据率启发式会过度拆分)。PD disaggregation 与 HiCache offload 暂被拒绝(两者都假设每层一个池)。

## 关键改动

```text
python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/layout.py: 新增 512B fp8 nope 行布局常量与 check_two_pool_pair
python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py: decode_fp8_2buff(aiter mla_a8w8 v4 nm)/prefill_fp8_2buff(aiter opus)
python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh: forward_fp8_2buff 两池 fused norm+rope+fp8 量化 store
python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py: is_unified_kv_fp8 门控(gfx95 否则回退)
```

```python
+DSV4_FP8_NOPE_ROW_BYTES = 512   # 448 B latent + 14 B E8M0 scales(写两遍) + 50 B pad
```

## 性能声明

MI355X、radix cache off、同机同 CLI 仅差 env 开关;TP=8 DeepSeek-V4-Pro:

- KV 容量:四种配置 full_token 比值 1.5004x–1.5023x(如 TP8:16,126,208 → 24,223,744 token @135.59 GB);`bytes_per_full_token` 8918→5945(0.6666x,因 indexer 项不随行宽缩放);另一 mem-fraction 下 1.5094x。
- 吞吐(ttt ratio = fp8/bf16):多数 cell 在 4.3% 以内,并发 ≥16 的 cell 在 3.2% 以内;68k/200 与 8k/1k DP8 场景基本持平(0.98–1.00)。
- 最差 cell 需标注:8k/1k、c=8、TP8 下吞吐 -7.6%、ITL +1.18 ms。作者归因于 decode 读者:`mla_a8w8` asm kernel 为 64 q head 调优,TP8 每 rank 仅 16 head,occupancy-bound,加载更少(640 B vs 1024 B)却更慢;DP8 下差距落入噪声;Triton 重写每个 shape 都更慢已弃;缩小 TP8 差距需 aiter 侧 16-q-head 变体(另立 PR)。
- GSM8K 1319 题、prefix cache 开、四组配对实验:fp8 0.946–0.949 vs bf16 0.940–0.952,无回归。

## 验证与回归风险

- 单测:对 `git diff` 涉及的 test/ 目录跑 pytest;pair check/layout 断言覆盖 nope/rope 池行数、行宽、dtype、连续性与 extend 行匹配。CI Base 与 Extra 通过,AMD ROCm 10 失败。
- 回归面:DSv4 unified_kv 路径为 ROCm-only;唯一 common-code 改动是 JIT norm+rope-store 模板里一个新 `elif`,既有分支不变。fp8 需要 OCP e4m3 + E8M0,gfx94x 的 e4m3fnuz dtype 会误读池内容,故非 gfx95 直接回退。依赖 #37140 的 AOT 重建(否则 6 个相关测试失败)。
- 后续:#38901 支持 DSpark、#37778 HiCache、#39166 PD、#39923 PCP,HiSparse 计划中。

## 关联

- 依赖/上下文:#37140(AOT 重建);相关 #45498(aiter pin)。后继:#38901、#37778、#39166、#39923。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
