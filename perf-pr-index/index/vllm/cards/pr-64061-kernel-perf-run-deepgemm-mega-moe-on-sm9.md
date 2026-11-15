# PR #64061: [Kernel][Perf] Run DeepGEMM Mega MoE on SM90 with MXFP4 experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @simon-veitner-redhat |
| 合并 | 2026-11-12 |
| 分类 | MoE(tier: major) |
| Label | ready, ci/build, deepseek, DSv4, dflash |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/64061 |

## 一句话结论

在 SM90(H200)上以 MXFP4 routed experts 启用 `deep_gemm_mega_moe`——DeepGEMM 新的 SM90 FP8 x FP4 Mega MoE kernel 在一个 persistent kernel 内融合 dispatch、L1 GEMM、SwiGLU+FP8 quant、L2 GEMM 与 combine(NVLink symmetric memory):DeepSeek-V4.1-Flash 单节点 DP8+EP8 output 吞吐比 main 默认高 17–26%、比 main+deepep_v2 高 12–19%。

## 动机

`deep_gemm_mega_moe` 此前要求 SM100;SM90 用户(H200)只能走 main 默认(HUMMING + block-FP8 activations)或 deepep_v2。DeepGEMM vllm-project/DeepGEMM#44 提供了 SM90 FP8 x FP4 Mega MoE kernel;本 PR 在 vLLM 侧接入(依赖该 PR,`cmake/external_projects/deepgemm.cmake` 的 pin 需在其落地后 bump,在此之前 SM90 测试 case skip)。

## 优化点

- **激活 staging 格式**:`deep_gemm_mega_moe.py` 新增 `MEGA_MOE_MXFP4_SM90` 权重格式——SM90 上激活按 FP8 dispatch、每 128 channels 一个 FP32 scale(SM100 保持 UE8M0/每 1x32);`prepare_megamoe_inputs` 以正确取整的 `amax * (1/448)`(`tl.math.div_rn`)计算,scale 与值和 DeepGEMM 的 `per_token_cast_to_fp8` bit-for-bit 一致,fp32 scale 的 bits 存入 int32 `x_sf` buffer 与 shared-expert scale 布局。相对两 PR 的更早版本,该格式在负载下快 1–2%,GSM8K 不变。
- **权重 loader**:relayout routed FP4 shards、按 DeepGEMM 的 SM90 transforms 重排 scales(scale row 超出 kernel 范围则 fail load 并指名 shard);fused FP8 shared experts 在全部 shard 载入后经 `process_weights_after_loading` requantize 一次(`requantize_fp8_weights_for_sm90_mega_moe`,weight/scale 必须成对)。
- **BLOCK_M hint**:每次 launch 的每个 rank 传 `num_tokens_for_block_m`(该 step 每 EP rank 平均 token 数),所有 rank 选相同 BLOCK_M,避免 idle DP rank 用小 block 陪跑;无 hint 时 ragged step 在最慢 rank 慢 1.5–1.9x。
- **Mega Gate**:DeepSeek V4 在非 SM100 上用 `GateLinear` 且 `fused_topk_bias` 关(无 `bf16_mega_gate`),warmup 在 SM90 跳过这些 config;`use_mega_gate` 属性控制 warmup。
- SM100 行为不变。

## 关键改动

注:pack 的 diff 被截断(注明 9 个文件略去),以下仅覆盖已展示部分。

```text
vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py: MEGA_MOE_MXFP4_SM90 格式、SM90 support check 与 per-format gate、SM90 shard relayout(_sm90_fp4_shard)、shared expert requantization(process_weights_after_loading)、forward 传 num_tokens_for_block_m(取 DP-wide 每 rank 均值)
vllm/model_executor/layers/fused_moe/prepare_megamoe.py: fp32 scale 用 div_rn 正确取整匹配 DeepGEMM cast;x_sf 写入 bitcast 后的 fp32 bits;shared scale TMA 布局支持 fp32 per-128K
vllm/model_executor/warmup/deep_gemm_warmup.py: mega gate m-values 需 module.use_mega_gate
tests/kernels/moe/test_deep_gemm_mega_moe.py: SM90 gate 与 fp32/fp8_128 staging bit-exact 测试、scale-row range 拒绝、shared shard requantize 测试
tests/models/test_deepseek_v4_mega_moe.py: mega-moe 测试改为 SM90 或 SM100
tests/model_executor/test_deep_gemm_warmup.py / tests/kernels/mhc/test_mhc_kernels.py: use_mega_gate 更新
```

```python
+MEGA_MOE_MXFP4_SM90 = replace(
+    MEGA_MOE_MXFP4,
+    hidden_quant=QuantKey(
+        FP8_DTYPE,
+        ScaleDesc(torch.float32, False, GroupShape(1, 128)),
+        symmetric=True,
+    ),
+    activations=frozenset({"swiglu"}),
+)
```

## 性能声明

(全部 8× H200,DeepSeek-V4.1-Flash,MXFP4 experts;吞吐表在 FP32 激活 scale 之前的 revision 测得,最终 revision 对比该 revision:+1.2% output tok/s @ c=1024、+1.9% total tok/s、−2.2% mean TTFT(prefill-heavy 4096 in/16 out, c=256)、+0.5% TPOT @ c=1;GSM8K 92.93 vs 92.72)

- **单节点 DP8+EP8**(random 1024/256,cells 为 output tok/s / mean TPOT ms,每 cell 两节点各 1 run 的均值,两 run 差 ≤7%):c=1 115 vs main 默认 92(+26%)、vs deepep_v2 97(+19%);c=16 1359 vs 1102(+23%)/1156(+18%);c=64 4029 vs 3402(+18%)/3513(+15%);c=256 9424 vs 7640(+23%)/7988(+18%);c=512 13380 vs 11478(+17%)/11953(+12%);c=1024 18566 vs 14733(+26%)/16013(+16%)。TPOT 相对默认 −15% 至 −23%,相对 deepep_v2 −9% 至 −16%。
- **两节点**(2×8 H200,各跑一个 DP8+EP8 replica):c=256 13191 vs 10938(+21%)/11439(+15%);c=512 20238 vs 16565(+22%)/17503(+16%);c=1024 28716 vs 24847(+16%)/26004(+10%);c=2048 39740 vs 31253(+27%)/34297(+16%)。
- **AgentX P/D,2 节点**:比 main HUMMING+deepep_v2 在 c=128/256/512 分别 +1.5%/+4.3%/+4.0% tok/s/GPU;ITL p50 降 10–11%(6.06/7.56/9.54 vs 6.76/8.51/10.69 ms),TTFT p50 降 12–17%(0.68/0.90/1.31 vs 0.77/1.08/1.51 s);agentic trace 的 idle gaps 限制吞吐。
- **Kernel 时间**(8 ranks,最慢 rank,3 run 中位):T=8192 约 675 TFLOPS/rank;decode 调优相对未调优 kernel 在 T=1–128 降 2.0–2.7%、T=512 降 8.5%,prefill sizes 不变;FP32 激活 scale 的最终 revision 再快 0.5–0.8%(T=1–512)与 4.1%(T=8192)。
- **正确性**:五条 greedy prompts(T=0)Mega MoE 与 main 默认输出连贯、内容一致(仅措辞/细节差异;一条 multi-turn 上 Day 3 vs Day 2 各称对方为 close second);最终 revision 上另 10 条 prompts 均正确。

## 验证与回归风险

- `tests/kernels/moe/test_deep_gemm_mega_moe.py`、`tests/models/test_deepseek_v4_mega_moe.py`(最终 revision):120 passed,12 skipped(skips 为 SM100-only NVFP4 cases),含 FP32 staging cases(与 DeepGEMM `per_token_cast_to_fp8` bit-exact);`tests/model_executor/test_deep_gemm_warmup.py` 与 FlashInfer EP 测试 29 passed。
- CI 风险:这些文件只在 B200 "DeepSeek V4 Kernels" step 运行,CI 仅覆盖 SM100;SM90 cases 需 DeepGEMM SM90 kernel(pin bump)与 H200 CI step,目前结果来自本地 H200。
- 风险:Mega MoE 需要 EP 在一个 NVLink domain 内(两节点场景每 replica 各自 EP8);scale row 超出 SM90 kernel 范围会 fail load;fused FP8 shared expert 的 requantization 要求 weight/scale 成对载入;吞吐表数据来自较早 revision(最终 revision 更快 0.5–4.1% kernel 层);DeepGEMM pin 待 bump。

## 关联

- 依赖 vllm-project/DeepGEMM#44(SM90 FP8 x FP4 kernel,落地后 bump pin);follow-up:pin bump。正文另提供 kernel visual walkthrough(按需)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
