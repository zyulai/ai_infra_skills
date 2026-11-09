# PR #62712: [ROCm][Perf][MiniMax-M3] MonoKernel decode for M3 MXFP4 on MI355X

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @micah-wil |
| 合并 | 2026-10-21 |
| 分类 | Quantization (tier: major) |
| Label | performance, rocm, ready, ci/build, multi-modality, cpu, mrv2, kimi, k3, minimax |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 为 3 个测试文件 + CI step,另有 24 个文件(实现本体)未展示 |
| 链接 | https://github.com/vllm-project/vllm/pull/62712 |

## 一句话结论

为 MiniMax-M3-MXFP4 在 gfx950 增加可选 --monokernel decode 路径:layers 3-59 的 FFN 半程合并为单个 persistent FlyDSL launch,MI355X TP4 上 conc 1 的 TPOT 降 16.6%、吞吐 +19.6%。

## 动机

MXFP4 MoE 的 decode 步骤由一串小 kernel 组成(TP all-reduce、residual+RMSNorm、FP32 路由 top-4、shared/routed experts、再一次 all-reduce),每次 launch 开销与跨 kernel 同步在小 batch decode 下占大头;1-16 行的 FULL CUDA graph decode 是收益最明确的场景。延续 RFC #60904 / Kimi-K3 的 monokernel 布局(DeepSeek-V4.1 已有先例)。

## 优化点

新增 vllm/models/minimax_m3/amd/specialized_decode/ 路径(实现文件在 pack 未展示的 24 个文件中):每步一个 plan(),层用 step.layer(i) 选择;layers 3-59 的 FFN 半程在单个 persistent FlyDSL launch 内完成 TP all-reduce、residual add、post-attention Gemma RMSNorm、FP32 router top-4(含 correction bias)、shared + routed BF16xMXFP4(A16W4,per_1x32)experts、最终 TP all-reduce;CTA 间跨 XCD 用 mailbox/hand-off 同步(mono/sync.py,附 _xcd_stress.py 压测协议)。Attention、dense layers 0-2、prefill 与混合 batch 走原路径。支持 eager 与 FULL CUDA graphs,1-16 decode rows;更大 batch、PIECEWISE、prefill 自动回落 stock。启动 default-deny 拒绝不支持配置并报告全部原因;kernel fault 在所有 rank fail-stop,失败步不返回 token。硬性要求:TP4 + gfx950、--kv-cache-dtype fp8、indexer_kv_dtype fp8、--moe-backend aiter、VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1、VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1。临时依赖 DeepSeek-V4.1 的 mono.common FlyDSL helpers(等公共 AMD runtime 落地后迁移)。

## 关键改动

```text
vllm/models/minimax_m3/amd/specialized_decode/(: 实现在未展示的 24 个文件中,含 mono/{ffn,sync,residency,runner} 与 runtime/executor)
tests/models/minimax_m3/amd_specialized_decode/test_model.py: 5 层 mini 模型、真实 post-load MXFP4/AITER 路径、TP4 精度与 fail-stop
tests/models/minimax_m3/amd_specialized_decode/test_ops.py: stage parity/路由 tie/padding/residency/graph replay 等 op 级测试
tests/models/minimax_m3/amd_specialized_decode/_xcd_stress.py: 跨 XCD CTA 间数据传递压测 kernel
.buildkite/test_areas/models_basic.yaml: 新增 MI355 MiniMax M3 MonoKernel CI step
```

## 性能声明

MI355X,TP4,FULL_DECODE_ONLY graphs,基线为推荐 serve 命令,--monokernel 为基线加所需 flags。Random 1024 in/1024 out:conc 1 TPOT 5.12 -> 4.27ms(-16.6%),输出吞吐 192 -> 230 tok/s(+19.6%);conc 2 -12.5%(+13.4%);conc 4 -8.7%(+9.3%);conc 8 -7.8%(+7.8%);conc 16 与 32(stock fallback)parity。Long context 8192/1024:conc 1 TPOT -14.9%(5.25 -> 4.47ms)、conc 4 -9.0%、conc 16 -3.4%。MT-Bench(greedy):conc 1 TPOT -17.0%(5.00 -> 4.15ms,196 -> 235 tok/s)、conc 4 -8.6%、conc 16 parity。Kernel-only(同 server 同 flags 去 --monokernel):TPOT conc 1 -19.1%、2 -14.3%、4 -11.1%、8 -8.8%、16 parity。GSM8K 5-shot 全量 1319:0.8969 -> 0.9052(paired +0.83pp,95% CI [-0.83,+2.50],McNemar p=0.37)。限制:spec decoding 被拒(stock 路径配合 spec 解码约比 --monokernel 快 1.6x);仅 TP4;16 rows parity。

## 验证与回归风险

新增三层测试:op 级(stage parity cos>0.9999/rel<1e-2、精确 tie 归低 id、NaN padding 行 inert、全 CU residency、graph replay 逐位一致、fail-stop 注入)、model 级(TP4 dummy 权重 + 真实 MXFP4 post-load 转换、teacher-forced logprobs 对拍、FULL graphs 下 epoch reset、late rank fault 不返回 token)、跨 XCD 压测;接入 mi355_4 CI step。回归面:默认关闭(--monokernel opt-in),不影响 stock 路径;新 CI step 依赖 4x gfx950;对 DeepSeek-V4.1 mono.common 的临时耦合需在公共 runtime 落地后迁移;仅覆盖 TP4,TP2/TP8 是 follow-up。

## 关联

前序:RFC #60904 与 Kimi-K3 monokernel 布局、DeepSeek-V4.1 mono decode(mono.common helpers、test_dsv41_mono_decode 系列已在其 CI step 中)。同区域:PR #63076(ROCm MoE 路由 top-k,本索引库同批次)。
