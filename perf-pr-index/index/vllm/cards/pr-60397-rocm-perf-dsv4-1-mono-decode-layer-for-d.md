# PR #60397: [ROCm][Perf][DSv4.1] Mono decode layer for DeepSeek-V4.1-Flash on MI355X

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Fangzhou-Ai |
| 合并 | 2026-10-10 |
| 分类 | MoE |
| Label | rocm, ready, ci/build, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供,正文 diff 另有 30 个文件被裁剪) |
| 链接 | https://github.com/vllm-project/vllm/pull/60397 |

## 一句话结论

MI355X 上 DSv4.1-Flash decode 的 opt-in 单内核路径:K1+K2 两个 persistent launch 替代每层约 40 个 kernel,kernel 级 1.10-2.18x,同请求 decode 快 7-9%。

## 动机

DeepSeek-V4.1-Flash 在 MI355X(gfx950)的 decode 路径由大量小 kernel 组成:每个 standard 层约 40 个 launch,TP all-reduce 走独立通信、MoE(router、top-6、shared expert、MXFP4 experts)各自单独 launch,launch 开销与中间往返主导小 batch decode 延迟。

## 优化点

opt-in(`VLLM_ROCM_MONO_DECODE=1`,定位为所有模型 mono decode 内核的总开关)后,decode step 的每个 backbone 层跑在 persistent FlyDSL launch 上:K1(30 个 standard 层的 mHC attention seam + attention 前段:wqkv、q/kv norms、KV insert、wq_b)、K2(attention 后段 + attention TP all-reduce + FFN seam + MoE 及其 all-reduce),其余 10 层(首层、Engram 层、compressor/indexer 层)走 FFN launch(接 wo_b 未归约输出)。所有 TP all-reduce 在 kernel 内基于 symmetric peer memory 完成。权重原地读取(attention 线性层用 vLLM row-major MXFP8 32x32 E8M0 scales,routed experts 用 AITER shuffled MXFP4);CTA 间 hand-off 用 per-step-width 的 tagged mailbox scratch(3-16 MB,DSpark 下最多 8 个 width),代替每步 mailbox 清空,保证 graph-safe。代码改编自 ROCm/ATOM 的 DeepSeek-V4.1 mono decode(MIT,commit 97359d6),按 vLLM 数值与 seam 特化。生效范围:仅 decode、≤48 行、TP2/TP4、eager 或 FULL graph、causal SWA 窗口与 `fp8_ds_mla` KV;PIECEWISE capture 一律拒绝;不满足部署条件(CDNA 版本、TP、FlyDSL、expert/DP/EPLB、MoE backend、<256 CU)显式报错。

## 关键改动

```text
vllm/models/deepseek_v41/amd/mono_decode.py: 每层/每步 eligibility,vLLM 张量到 kernel 输入(在裁剪部分)
vllm/models/deepseek_v41/amd/mono/: runner.py、layer.py(K1/K2/FFN launch)、attention/、stages/、common/(在裁剪部分)
tests/models/test_dsv41_mono_decode.py: 路径选择测试(whole/ffn/拒绝条件)
tests/models/test_dsv41_mono_numerics.py: 与 vLLM ops 逐 stage 数值对比
tests/models/test_dsv41_mono_layers.py: 五层真实模型 mono on/off 输出一致性
```

## 性能声明

Kernel 级(MI355X,96 个点:TP2/TP4 x concurrency 1-8 x 1-6 token/request x 两类层):全部快于 vLLM 现路径,1.10-2.18x;FFN launch 48 点全部更快或持平(TP2 1.01-2.14x,TP4 1.12-2.32x)。AgentX e2e(InferenceX#3690 recipe,a8w4 MoE,DSpark acceptance 3.51):对同节点 mono-off 基线(TP2),p90 interactivity 1.14x-1.35x、总吞吐 1.04x-1.16x;FFN launch(40 层)vs K1+K2(30 层)同请求 decode 每点快 7-9%。GSM8K(全量 1319,5-shot,TP2):current 0.9742 / K1+K2 0.9750 / 全 40 层 0.9735。精度:GPQA-Diamond 85.3 → 85.1(Δ -0.2)、AIME25 99.4 → 98.3(Δ -1.0);RULER NIAH 32K/128K/256K 全部 40/40。数值测试:余弦 ≥ 0.9993,相对误差 0.3-0.5%,300 次 graph replay bit-exact;100K 次 race stress 无 mismatch。注意 kernel/GSM8K/AgentX 数字来自开发期 AITER 托管版本,FFN launch 与 TP4 GSM8K 来自 in-tree 代码。

## 验证与回归风险

新增三个测试文件(CPU 路径选择、4x MI355 numerics CI step、模型级 layer 对比)并接入 Buildkite models_basic 与新增 AMD numerics step。风险:TP rank 必须 lockstep 同路径;wo_b 在 FFN-launch 层改为 `reduce_results=False`(顺带修复 main 上 sequence parallel + FP8-block checkpoint 时 wo_b 被 all-reduce 后又 reduce-scatter 的 bug);依赖 `flydsl` wheel 与两个 AITER helper;仅 CDNA4/gfx950、TP2/TP4、≤48 行 decode 生效,其余路径不变;RFC #59665 的精度协议执行。

## 关联

同区域 mono-decode 姊妹 PR:#59705(MiniMax-M3)、#60216(GLM-5);来源 ROCm/ATOM(aiter#6196 承载开发版内核)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
