# PR #60731: [ROCm][Perf][DSv4.1] Mono decode on MI300X and MI325X (gfx942)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @maeehart |
| 合并 | 2026-10-20 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | rocm, speculative-decoding, ready, ci/build, deepseek, mrv2, dflash, DSv4.1 |
| Diff | diffstat 缺失(pack 未记录);diff 极大,pack 保留 7 个测试/CI 文件,37 个文件因篇幅省略 |
| 链接 | https://github.com/vllm-project/vllm/pull/60731 |

## 一句话结论

把 DSv4.1-Flash 的 mono decode 内核从 gfx950 移植到 gfx942(MI300X/MI325X),标准 decoder 层从 32 个内核并为 2 个,AgentX 交互性最高提升 2.4×。

## 动机

PR #60397 已在 gfx950(MI355X)实现 DSv4.1 mono decode(整层合为少量内核),但 gfx942 用户(MI300X/MI325X)仍走 vLLM 逐算子路径:一个标准 decoder 层 32 个内核、320 µs。gfx942 与 gfx950 能力差异明显:64 KB LDS(对 160 KB)、无 scaled FP4/FP8 MFMA、FNUZ FP8,需要独立的内核实现与权重布局。

## 优化点

把 mono decode 移植到 gfx942:`--monokernel` 开启后,每个 backbone 层整层运行(与 #61082 在 gfx950 的层计划一致),标准层从 32 kernel 降到 2 kernel。gfx942 专属 helper ops(位于 `vllm/models/deepseek_v41/amd/ops/`):稀疏 indexer 的 decode top-512(`topk512_gfx942.cu`,模型创建时用 hipcc 即时编译,失败则告警回退 vLLM top-k)与其 candidate logits、SWA decode metadata 与 top-k packing 各一次 launch、DSpark Markov 采样每 draft 步一次 launch;draft lm_head 把 5 行 logits GEMM pad 到 6 行以避开 wvSplitK(32320×5120 shard 上 96 µs 对 295 µs)。权重加载时按 gfx942 GEMV 字节序复制 routed experts(无 padding 列)、dense linear 按 e4m3 复制,故 TP4 下每 rank 权重 170.11 GiB(原 98.88 GiB);reload 原地写入副本。限制:仅 TP4、decode step ≤42 行(7 请求 × 5 草稿,gfx942 LDS 使 GEMV 每次 6 token);更大步/prefill/混合步/PIECEWISE graph 回退 vLLM 路径。同时合入 #62131 的 draft runner scratch 修复(`scratch_layout` 按 runner 分组数计算,`run_ffn` 校验 MoE 区域与布局)。

## 关键改动

```text
vllm/models/deepseek_v41/amd/ops/: gfx942 helper ops(topk512、cand_logits、
  swa_decode_meta、topk_pack、fused_markov、draft_lm_head 等,文件在省略 diff 中)
vllm/models/deepseek_v41/amd/sparse_indexer.py: gfx942 indexer 层封装
vllm/v1/worker/.../dspark/speculator: _sample_sequential 先询问 draft 模型的
  sample_sequential 的 8 行 hook
tests/kernels/models/amd/、tests/models/test_dsv41_mono_{layers,numerics}.py: 内核/层/回放测试
.buildkite/test_areas/{kernels,models_basic}.yaml: 新增 MI300 单卡与 4 卡 CI job
```

关键机制摘录(tests/kernels/models/amd/test_dsv41_gfx942_decode_meta.py 的等价性语义):

```python
# vLLM's ops run first; then every output is overwritten with -7, the fused
# launch runs, and every output a reader uses must be equal.
bad = [k for k in want if not torch.equal(want[k], got[k])]
assert not failed
```

## 性能声明

DeepSeek-V4.1-Flash,MI325X TP4,DSpark 5 drafts(AgentX 用 InferenceX 结果代码评分;main 臂为 ROCm nightly 镜像):

- AgentX 并发 1(3600s/臂,vs main):interactivity P50 216.85 → 503.53 tok/s/user(2.32×),P90 182.25 → 411.26(2.26×),每 GPU 总吞吐 4,024 → 6,546 tok/s(1.63×);
- 并发 2/4/8(vs InferenceX 公布的 MI325X TP4 基准):P90 153.22→368.64(2.41×)、142.75→316.28(2.22×)、81.52→162.61(1.99×);吞吐 1.58×/1.41×/1.35×;
- 单层:115 µs vs 320 µs;单请求 decode period 7.2 ms vs 17.9 ms;并发 8 时仅 0.14% decode 时间回退 vLLM 路径。

精度:GSM8K strict(1319 题,block rejection)并发 7 池化 +0.23 点(p 0.49,不显著);GPQA-Diamond 85.0%、AIME25 99.0%、RULER NIAH 9 格全部 40/40。gfx950 回归:MI355X TP2 `--monokernel` 下 GSM8K 与各时延指标在两次运行波动内一致(TTFT 差 <1 ms),76 个 mono 内核构建的寄存器/LDS/指令计数与 main 相同。

## 验证与回归风险

新增 9 个测试文件:top-512 对照 vLLM `top_k_per_row_decode`(上下文 100-300k)、candidate logits 与 AITER dense logits 逐位一致、gfx942 indexer op 与 vLLM op 对照(层 20 写候选块等)、SWA metadata/topk packing 对照 800/500 随机 batch、fused Markov 每 draft token 与 vLLM `_sample_logits` 精确一致、draft lm_head 的 wvSplitK 计数、K2 key list/GEMV/FP4-FP8 转换对 torch 参考、TP4 层输出界与 300 次 graph replay 逐位一致;新 CI job(MI300 单卡 + 4 卡)在 ROCm 7.2 与 10 nightly 下通过。调试性逐层对照(481,944 个 top-512 条目全部匹配)未发现非有限值。风险:mono 内核把激活量化为 MXFP8 而 vLLM gfx942 路径保持 bf16(随机权重下 FFN 输出差 4.65%、残差流至 2.5%,路由近平局可翻转专家);权重双份副本占显存(KV cache 受限,MI300X 192 GiB 不太放得下,数字来自 MI325X);ROCm 10 nightly 下一次启动曾卡在 AITER custom all-reduce;与 #61311/#61482 的目录迁移有合并冲突,后合者需移植。

## 关联

- 前序:PR #60397(gfx950 mono decode)、#61082(gfx950 层计划/开关)、#62131(draft 层整层运行 + scratch 修复,已并入本分支)
- 流程:RFC #60904(模型内核目录约定);冲突相关 #61311(amd/mono → amd/specialized_decode/mono)、#61482(FlyDSL 迁移)、#61869(fused Markov sampler)
- 同区域:DSv4.1 系列(本索引 PR #56935、#59327、#56441、#63921)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
