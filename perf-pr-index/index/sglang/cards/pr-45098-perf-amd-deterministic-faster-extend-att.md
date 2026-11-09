# PR #45098: [Perf] [AMD] deterministic: Faster extend attention launch on gfx950

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @aarnetalman |
| 合并 | 2026-10-20 |
| 分类 | Attention / Kernel(tier: major) |
| Label | jit-kernel |
| Diff | pack diffstat 缺失;diff 为 extend_attention.py 的 tile 启动改动加一个新 bitwise 单测 |
| 链接 | https://github.com/sgl-project/sglang/pull/45098 |

## 一句话结论

gfx950 确定性模式的 extend attention 改用 256 行 query tile,KV 读取量降 4x,64k+ 输入吞吐最高 +130%,输出 bitwise 不变。

## 动机

确定性推理(`--enable-deterministic-inference`)的 prefill 使用 unified extend-attention kernel(归约顺序与 batch 无关)。gfx950 上 head_dim ≤ 128 一律以 64 行 query tile + 4 warps 启动,每个 workgroup 为自己的 query 行流式读取整个 prefix,长 prompt 下 kernel 被 KV 读取限制:64k-token 输入占确定性模式 GPU 时间的 82%(Qwen3-8B on MI350X),确定性吞吐降至正常模式约 0.2x。

## 优化点

在 `extend_attention_fwd_unified` 中,gfx950 且 head_dim ≤ 128、默认 `BLOCK_N = 64` 时改以 `BLOCK_M = 256`、8 warps 启动:一个 workgroup 为 4 倍数量的 query 行各读一次 KV block,单位 query 行的 KV 字节读取降 4x。`BLOCK_N` 不变意味着每行 softmax 与求和的累加顺序不变,输出与现有 launch bitwise identical。其他 GPU、其他 head dim、共享 block-size 表与非确定性 extend kernel 均不变。PR 还尝试过 pipelined 变体(128 行 tile、2 stages):孤立测试更快,但在 ROCm Triton 编译器中 abort(LLVM `iota_range` assertion)针对 server 的 kernel specialization,故未采用。

## 关键改动

```text
python/sglang/kernels/ops/attention/extend_attention.py: gfx950 + head_dim<=128 + BLOCK_N=64 时 BLOCK_M=256、num_warps=8
test/registered/attention/test_triton_attention_kernels.py: 新增 test_extend_attention_unified_gfx950_tile_bitwise
```

```python
if _is_gfx95 and Lq <= 128 and Lv <= 128 and BLOCK_N == 64:
    # a 256-row query tile cuts the KV bytes read per query row 4x.
    # BLOCK_N, and so the reduction order, is unchanged: outputs are
    # bitwise identical.
    BLOCK_M, num_warps = 256, 8
```

## 性能声明

`sglang.benchmark.serving --dataset-name random --random-range-ratio 1.0 --request-rate inf`,deterministic mode,before/after 在同一 GPU 上交替两次(每对 within 0.5%),output throughput(tok/s):

- Qwen3-8B(Triton prefill + AITER decode):1k/1k conc256 7,070 → 7,234(+2.3%);8k/1k conc128 1,211 → 1,469(+21%);15k–32k/256 47.9 → 86.2(+80%,32k);64k/256 17.8 → 35.2(+98%,YaRN factor 4、128k context);120k–126k/256 5.9 → 13.7(+130%,YaRN)。
- Qwen3-30B-A3B:8k/1k 1,262 → 1,664(+32%)。Phi-4:8k/1k 778 → 924(+19%);15k 95.4 → 139.8(+47%)。gpt-oss-20b:8k/1k 1,644 → 1,877(+14%);32k +77%;120k 13.0 → 27.5(+112%);TP2 8k/1k +11%、32k +59%。gpt-oss-120b:8k/1k 1,031 → 1,169(+13%);32k 59.6 → 105.3(+77%)。
- TTFT 同幅度下降(Qwen3-8B 32k:41.4 → 22.3 s);ITL 不变(decode 不用该 kernel)。gpt-oss 使用 ROCm/aiter#6583 的 deterministic MoE。
- Profile(64k 输入,Qwen3-8B,8 prompts、40 forward steps):`_fwd_kernel_unified` GPU 时间 79.0 → 30.7 s;总 GPU 时间 96 → 48 s。
- 无 `--enable-deterministic-inference` 或非 gfx950 无变化。测试镜像 `rocm/sgl-dev:v0.5.22-rocm724-mi35x-20261018`(MI350X)。

## 验证与回归风险

bitwise 一致性三重验证:kernel level 500 个 launch 配置(8k chunk after 0–56k prefix、4-sequence 混合 batch,head_dim 128 与 64,`BLOCK_N=64`)`torch.equal`;新单测比较新旧 tile 输出;e2e `test_deterministic --test-mode {single,prefix,radix_cache} --n-trials 50`、`--return-logprob`、batch 1–50 的 bitwise 检查(1,275 responses token ID 与 logprob 完全一致),覆盖 Qwen3-8B(TP1/2/8)、Qwen3-30B-A3B、Phi-4、gpt-oss-20b(TP1/2)、gpt-oss-120b;2026-10-19 nightly 复测通过。风险:同文件两个 pre-existing 失败与=`test_extend_attention_unified_vs_regular` 的 head_dim-80 case(max diff 0.167,tol 0.15/0.17),在改动前后同样失败,前者由 #45099 修复(与本 PR 可干净叠加)。

## 关联

#45099(恢复 #34461 加入、#35770 收窄的 tile,修复 `test_extend_attention_block_sizes`,独立 PR);#34461、#35770(tile 历史)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
