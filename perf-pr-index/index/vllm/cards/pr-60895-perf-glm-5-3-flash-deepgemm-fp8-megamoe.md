# PR #60895: [Perf][GLM-5.3-Flash] DeepGEMM FP8 MegaMoE with fused shared experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gcanlin |
| 合并 | 2026-11-05 |
| 分类 | MoE（tier: major） |
| Label | ready, deepseek, quantization, DSv4, glm |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 含 5 个文件（新增 mega_moe.py 适配器与 test_mega_moe.py，改动 mxfp8_utils.py、glm5next model.py/mtp.py） |
| 链接 | https://github.com/vllm-project/vllm/pull/60895 |

## 一句话结论

GLM-5.3-Flash 的 MoE 层改走 DeepGEMM SM100 FP8×FP8 MegaMoE（routed+shared experts 单 kernel 且含 EP dispatch/combine），AgentX replay 吞吐 1.6 倍于 generic MoE 路径。

## 动机

GLM-5.3-Flash checkpoint 的专家权重为 block FP8（E4M3，每 128×128 block 一个 float32 scale），generic MoE 路径走 FLASHINFER_TRTLLM。DeepGEMM SM100 MegaMoE 把 routed 与 shared experts、EP dispatch/combine 融进一个 kernel 并把 scale 直接喂给 block-scaled MMA，但要求 UE8M0（2 的幂）scale、每行 32 列粒度，与 checkpoint 格式不匹配——需要一次性 requant 桥接。开启方式：`--kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'` + `--tensor-parallel-size 1 --data-parallel-size N --enable-expert-parallel`。

## 优化点

(1) 新增 `Glm5NextMegaMoEExperts`（继承 `DeepGemmMegaMoEExperts`，`MEGA_MOE_MXFP8` 格式）：E4M3 shard 直接载入 kernel 布局，float32 block scale 暂存于 `*_weight_scale_inv`；load 完成后 `requantize_block_fp8()` 将新载 shard 原地圆整为 UE8M0 1×32 scale（以 8 个 expert 为一批），主模型/多模态 wrapper/MTP drafter 三处均处理。(2) shared expert 同样融合进 experts 权重，模型不再构建 shared MLP。(3) 路由留在 kernel 外：`fused_topk_bias`（sigmoid + score correction bias），`routed_scaling_factor` 折进 top-k 权重；DeepGEMM 自带的 Mega Gate 仅支持 sqrtsoftplus 故不采用。(4) 权重 reload（layerwise/direct）在 kernel tensor 上保持 storage 不变（依赖 #64357/#64336 的 reload 钩子），派生 UE8M0 scale 标记 `weight_loader_numel = 0`，layerwise reload 在 checkpoint tensor 齐后逐层重放 requant。(5) requant 有损：非 2 的幂 scale 会圆整，代码中加 warning 提示换 moe_backend。限制：仅 SM100、`tensor_parallel_size=1`（DP+EP）、需 block-FP8 checkpoint、单 expert group、无 sharded RDT/sparse NCCL updates。

## 关键改动

```text
vllm/models/glm5next/nvidia/mega_moe.py: 新增，DeepGemmMegaMoEExperts 适配器（分片加载、UE8M0 原地 requant、shared expert 融合、reload 配对校验）
vllm/model_executor/layers/quantization/utils/mxfp8_utils.py: 新增 requant_block_fp8_to_ue8m0（block-FP8 → UE8M0 1x32）
vllm/models/glm5next/common/model.py: use_mega_moe 分支（_init_mega_moe_experts、forward 走 fused_topk_bias + experts、load_weights 处理 shared_experts 权重）
vllm/models/glm5next/common/mtp.py: MTP drafter 同步 MegaMoE 权重路径
tests/models/glm5next/test_mega_moe.py: 新增 5 个用例（requant 精确性、跨调用分片加载、reload storage 不变等）
```

requant 核心：

```python
def requant_block_fp8_to_ue8m0(weight, scale):
    n, mn, k = weight.shape
    _, blocks_m, blocks_k = scale.shape
    real = weight.to(torch.float32).view(n, blocks_m, -1, blocks_k, k // blocks_k)
    real.mul_(scale[:, :, None, :, None])
    return _mxfp8_e4m3_quantize_torch(real.view(n, mn, k))
```

## 性能声明

测试条件：4×B200、nightly `8cbd5d03` + 本卡、MTP 5（`synthetic_acceptance_length` 3.61 固定），三臂同节点同容器、各跑一次，仅并行布局与 moe_backend 不同。AgentX replay（aiperf，semianalysis traces，concurrency 96，1 小时）：DEP4+MegaMoE output 3,332 tok/s vs DEP4 generic 2,063（**1.6x**）vs TP4 3,152；TTFT avg 0.81 s vs TP4 3.93 s（约为其五分之一）；总吞吐 402,628 tok/s；preemptions 2（TP4 35/generic 53）。Prefill（8192/1 workload）115,081 tok/s：**1.8x 于 DEP4 generic（63,900）、2.5x 于 TP4（46,334）**；1024/1024 workload output 11,691 tok/s。精度：GSM8K 5-shot 0.9735 vs TP4 0.9757（±0.0042/0.0044，一个标准误内）；AIME 2026 Pass@1 0.934 vs generic 0.939（131K output cap），1M context 下 0.977±0.005。已知有损 requant 代价：每权重张量约 2.4% 相对 L2、单 MoE 层输出约 3.9%（layer 15）。weight reload 双模式均恢复 byte-identical 权重、地址不变、greedy 输出 4/4 匹配。单次运行（one run each），TP4 对比亦为单次。

## 验证与回归风险

单元测试 `pytest tests/models/glm5next/test_mega_moe.py` 5 passed，覆盖 power-of-two scale 精确 requant、权重与 scale 分两次 load_weights 的配对校验（缺一半时报 ValueError）、layerwise/direct reload 后 storage 地址不变、UE8M0 scale 不计为 missing load。风险面：requant 有损且不可逆（对非 2 的幂 scale），精度敏感场景需权衡；`requantize_block_fp8` 假设 load 完成时 weight/scale 成对到位；NATIVE_MEGA_MOE_BACKENDS 判定耦合 kernel_config；限制条件多（SM100 only、TP=1、需 EP、block-FP8 checkpoint），配置不满足会 NotImplementedError。对其他 moe_backend 无改动。

## 关联

同区域：#55275（GLM 的 BF16/MXFP4 MegaMoE，当时 open）、#54049（FlashInfer NVFP4 mega kernel）、#64357（layerwise reload 钩子）、#64336（direct reload 模式）。模型前缀 `--prefix-match-unit` 与 KDA state page 的交互在本卡正文被注明为与本 PR 无关但影响该模型。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
