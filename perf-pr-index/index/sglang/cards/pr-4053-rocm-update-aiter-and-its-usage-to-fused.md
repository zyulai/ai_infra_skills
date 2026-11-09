# PR #4053: ROCm: update aiter and its usage to fused moe (bloat16, fp8, fp8 block-quant)

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @HaiShaw |
| 合并 | 2025-03-04 |
| 分类 | MoE(major) |
| Label | (无) |
| Diff | pack 未提供 diffstat(diff 正文可用,4 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/4053 |

## 一句话结论

在 ROCm 上把 CK_MOE 路径接入 aiter 的 fused MoE(bf16/fp8/fp8 block-quant),PR 声明 DSv3 提速 25-35%。

## 动机

PR 正文动机一栏只有一句:"Performance boost (DSv3 +25~35%)"。当时 ROCm 上 `CK_MOE` 环境变量下的 MoE 走 aiter 的 `fused_experts_ck` 且仅支持 silu 激活、无 fp8 block-quant 支持;本 PR 更新 aiter 依赖并改用其更完整的 fused MoE 入口(ck_moe / asm_moe),覆盖 bf16、fp8、fp8 block-quant 三类权重。

## 优化点

定义 AITER fused MoE 支持矩阵:bfloat16(SILU/GELU)、float8(仅 SILU,GELU 待加)、float8 block quant(仅 SILU,对应 DSv3/R1);INT4-FP8 为后续计划。支持矩阵之外回退 Triton 实现。机制上:(1) bf16 路径(`fused_moe_triton/layer.py`)从 `fused_experts_ck` 换成 `aiter.ck_moe`,activation 参数透传,去掉仅 silu 的 assert;(2) fp8 路径(`fp8.py`)从 `fused_experts_ck` 换成 aiter 的 `asm_moe`(`fused_moe_bf16_asm`):block-quant 情况传 `weight_scale_inv` 与 `weight_block_size`(DSv3/R1),per-tensor 情况使用新增的列向 scale `w13_weight_scale1`/`w2_weight_scale1`(把 per-tensor scale 广播成 per-column,创建时置 1、加载后乘上原 scale),且仅当 `activation == "silu"` 才走此路径;(3) fp8 权重加载后用 `shuffle_weight` 做 (16,16) 预 shuffle 以匹配 aiter 的布局要求;(4) Dockerfile.rocm 的 AITER_COMMIT 从 `dev/testx` 更新为 `testx`,pyproject 的 ROCm 基础镜像升级到 `rocm/vllm-dev:20250114`。

## 关键改动

```text
python/sglang/srt/layers/quantization/fp8.py: CK_MOE 下 asm_moe 接入 + 列向 scale + 权重预 shuffle
python/sglang/srt/layers/moe/fused_moe_triton/layer.py: bf16 路径改用 aiter ck_moe
docker/Dockerfile.rocm + python/pyproject.toml: aiter commit 与 ROCm 基础镜像更新
```

## 性能声明

PR 正文给出 `bench_one_batch` 数字,条件为 DeepSeek-V3(本地权重),TP8,`--quantization fp8`,`RCCL_MSCCL_ENABLE=0 CK_MOE=1`:batch 64 / input 2048 / output 256 时 prefill 16,129.57 tok/s,decode 中位 1,397.96 tok/s,总吞吐 7,452.11 tok/s;batch 128 时 prefill 19,493.80 tok/s,decode 中位 1,983.23 tok/s,总吞吐 9,862.72 tok/s。声明为 "DSv3 +25~35%",但 PR 正文未附基线(不开 CK_MOE)的对照数字,提高幅度的基准无法从 pack 中核实。未提供精度评测结果。

## 验证与回归风险

PR 未附单元测试或精度数据(checklist 各项为模板勾选)。风险面:`CK_MOE` 开启时 fp8 的 per-tensor 路径依赖新参数 `w13_weight_scale1`/`w2_weight_scale1` 的加载与广播正确性;权重 (16,16) shuffle 是破坏性原地替换(`process_weights_after_loading`),只应执行一次;GELU 激活、fp8 非 silu 场景会静默落回 Triton `fused_experts`;aiter commit 更新影响所有 ROCm 构建的 aiter 行为。`CK_MOE` 未设置时路径不变。

## 关联

同区域(AMD MoE/aiter 线):#48440(aiter opus sort 上限调整)、#39273(AITER MoE runner)、#41725(aiter shared-expert append);本 PR 是 SGLang ROCm aiter fused MoE 的早期接入点。pack 未列出其他具体 PR 编号。
