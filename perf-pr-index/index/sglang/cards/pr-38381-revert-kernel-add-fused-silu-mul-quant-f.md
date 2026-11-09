# PR #38381: Revert "[kernel] add fused silu mul quant fp8"

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-09-08 |
| 分类 | Speculative Decoding (tier: major) |
| Label | quant, sgl-kernel, jit-kernel |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用(4 个文件,以删除为主) |
| 链接 | https://github.com/sgl-project/sglang/pull/38381 |

## 一句话结论

回退 fused silu+mul+FP8 quant 内核:恢复两 kernel 路径,修复其导致的投机解码接受率与速度回归。

## 动机

#37376 在 block-wise FP8 MoE 路径上启用 silu+mul+FP8 量化的单 kernel 融合,但它与被替换的两 kernel 路径数值不等价,差异足以拉低 GLM-5.2-FP8 EAGLE MTP 的投机解码接受率:`test_dsa_glm52_tp_mtp.py::test_bs_1_speed` 断言 `accept_length > 4.0`,在 main 上持续失败。因此整体回退。

## 优化点

本 PR 是纯回退,删除融合路径并恢复原机制:原融合 kernel 直接产出 FP8 down-GEMM 输入,跳过 bf16 `intermediate_cache2` 的 round-trip,单次 launch 省一次 kernel 调度,但 `_can_use_fused_silu_mul_quant_fp8` 仅凭 shape/dtype 自动开启、没有任何 flag 或环境变量可关闭,受影响模型无法 opt out。而 #37376 的单测容差本身就声明了非等价:`fp8_match > 0.95`(至多 5% 的 FP8 字节可不同)、`rtol=0.02, atol=0.02`。精度类 eval(cosine_sim > 0.9999)能吸收该误差,但 acceptance length 是 draft/target token 逐 token 一致的离散比率、在 5 个 draft 步上复合,且 target 与 draft 共享同一 MoE,扰动不断累积而不是平均掉。回退后 block-wise FP8 MoE 重新走 silu_and_mul + per_token_group_quant_fp8 两 kernel 路径。正文说明选择 revert 而非 gate 是因为当下不存在开关;reland 可加 opt-in flag 或在投机解码启用时禁用。

## 关键改动

```text
python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py: 删除 _fused_silu_mul_quant_fp8_kernel 与 fused_silu_mul_quant_fp8
python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py: 删除 _can_use_fused_silu_mul_quant_fp8 与自动融合分支,恢复 intermediate_cache2 两 kernel 路径
test/registered/kernels/ops/quantization/test_fused_silu_mul_quant_fp8.py: 删除该融合内核的容差型单测
python/sglang/kernels/aot/benchmark/bench_silu_quant_fp8.py: 删除对应 benchmark 脚本
```

## 性能声明

Bisect 定位:在 `8-gpu-h200` pool 上对 #37376 及其父提交各跑两次 `Rerun Test`。父提交 `503511c963`:Acc Length 4.727、1092 tokens、323.28 token/s(pass);#37376 `a8b2f36dee`:Acc Length 3.949、778 tokens、265.40 token/s(fail,三次重跑均复现)。即该 recipe 下 **-16.5% accept length、-18% bs=1 decode speed**。测试为 GLM-5.2-FP8 EAGLE MTP(`test/registered/e2e/models/test_dsa_glm52_tp_mtp.py::test_bs_1_speed`)。4.727 是 #37376 落地前所有 scheduled main run 的稳定值。本 PR 回退后预期恢复该水平,PR 正文未附回退后的新基准数字。

## 验证与回归风险

验证依赖既有 e2e 测试 `test_dsa_glm52_tp_mtp.py::test_bs_1_speed`(断言 accept_length > 4.0)。回退恢复两 kernel 路径,影响所有命中原融合条件的 block-wise FP8 MoE 调用(CUDA、bf16/fp16、silu gated、无 gemm1_alpha/limit、无 hooks);#37376 引入的容差型单测与 benchmark 脚本一并删除。若后续 reland,正文建议改为 opt-in flag 或在投机解码场景禁用。

## 关联

回退 #37376([kernel] add fused silu mul quant fp8);正文未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
