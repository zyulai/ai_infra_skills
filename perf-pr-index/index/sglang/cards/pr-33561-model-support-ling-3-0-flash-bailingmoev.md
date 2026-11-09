# PR #33561: [Model] Support Ling-3.0-flash (BailingMoeV3)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-08-27 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, quant, amd, deepseek, sgl-kernel, blackwell, npu, run-ci, jit-kernel, run-ci-extra |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 超长仅保留源码类文件,66 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/33561 |

## 一句话结论

Day-0 接入 Ling-3.0-flash(KDA+MLA 混合 MoE),用融合 KDA 链式 verify kernel、router matvec 与全链 PDL 压低投机解码开销。

## 动机

SGLang 此前不支持 `BailingMoeV3ForCausalLM`(`model_type=bailing_hybrid`):一种 KDA 线性注意力(带 safe-gate lower bound)与 MLA 交替的混合 MoE 架构,512 路专家,需要 MTP(NEXTN)与 DSPARK 两套投机解码路径同时可用。PR 提供 Day-0 模型支持,并把该模型依赖的 KDA verify 热路径一并做了内核级优化。

## 优化点

核心是新增融合 KDA chain-verify Triton kernel:把 MTP `target_verify` 热路径中的 `causal_conv1d_update`(chain 模式)与 `fused_sigmoid_gating_delta_rule_update`(T 步递推)融合为单个 kernel,消除每层每次 verify 的一次 kernel launch、conv 与递推之间 mixed_qkv 的 HBM 往返、以及喂给 conv kernel 的两次 transpose 拷贝;数值上与 unfused 路径逐行对齐(conv 输出显式 round 回 bf16)。其次,新增 `router_gate_matvec` kernel,小 M 场景下用单次 launch 替代库 GEMM/eager upcast 链,BLOCK_K 一次覆盖 K 以避免冷权重下的串行依赖加载链(~94MB/layer 专家流量会冲掉 L2)。同时把 PDL(`gdc_wait`/`gdc_launch_dependents`)接入 fused_moe_kernel、moe_sum_reduce、causal_conv1d、fused_sigmoid_gating 等多个 kernel,实现调度级重叠(声称 bit-exact)。DSPARK 侧新增 `MarkovGreedyStep` 融合 draft 步:对 (vocab x rank) 权重单 pass 完成 argmax,不再物化全词表 bias 或分多个 launch。KDA safe gate(lower_bound)传播进 decode/target-verify kernel。

## 关键改动

```text
python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py: 新文件,融合 conv+delta-rule 的 KDA 链式 verify kernel(BV=4 tile,num_warps=4,PDL)
python/sglang/kernels/ops/moe/router.py: router_gate_matvec 小 M router logits kernel(M<=8 走 matvec,否则回库 GEMM)
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py: fused_moe_kernel/moe_sum_reduce 接入 PDL(GDC_EARLY 按 A 行数选择触发时机)
python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py: MarkovGreedyStep 融合 greedy draft 步(CUDA 走 Triton,CPU 回退 torch)
```

## 性能声明

服务吞吐(80 条 exact-length 随机请求,ISL/OSL 8192/1024,c1/c16,吞吐含输入输出 token/s/GPU,全部开启 decode CUDA Graph):H200 INT4 1,159/7,184;H200 MXFP4 709/5,215;B200 INT4 482/1,344;B200 MXFP4 785/7,933。对应 GSM8K(1,319 题):95.30% / 96.29% / 96.74% / 96.29%。

投机解码(Ling-3.0-flash BF16,4x B200 TP4,temperature=1.0/top_p=0.95,thinking on,full GSM8K,max_tokens=8192):DSPARK 96.66%,MTP(NEXTN) 96.44%,错误率均 0.00%。

diff 内注释:融合 verify kernel num_warps=4 时在 graph 内比 unfused 对约快 1.3x;B200 T=5 逐层扫描 BV=4 为 11.56 us/layer;router matvec 冷缓存(H20-3e,E=513,K=2560)M=2 时 bf16 4.5 us vs 库 13.8 us。

## 验证与回归风险

新增 exactness、fusion-gate、quantization、parser 测试;四种量化组合通过 20-example GSM8K 回归门(100% stop rate)。风险点:融合 verify kernel 仅支持 chain speculation(`speculative_eagle_topk == 1`)、conv width 4、K 为 2 的幂;fp32 中间 ssm 回滚缓存存在 reduction-order 差异(T=4 标准门 ~6e-8,safe gate ~1.5e-5,T=8 safe gate ~2e-3)。PDL 改动波及 KDA/GDN recurrent 与 fused_moe 广泛路径(调度 only、声称 bit-exact,但属全局行为变化)。

## 关联

同区域(卡片库中):#29472(FlashKDA prefill backend)、#30169(GDN/KDA CuTeDSL prefill state fusion)、#43621(DSPARK MarkovSampleStep one-op)。PR 正文引用部署配方讨论 #34363。
