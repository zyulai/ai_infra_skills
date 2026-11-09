# PR #56226: [ROCm][Perf] Route softmax MoE top-k, including fused shared-expert sigmoid, through AITER topk_gating

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-10-15 |
| 分类 | MoE |
| Label | performance, rocm, ready |
| Diff | diffstat 缺失(素材未记录 +/- 行数与文件数);diff 可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/56226 |

## 一句话结论

ROCm CDNA3/4 上 softmax MoE 路由默认改走 AITER `topk_gating`:小 token 数时 kernel 快 37–47%,单流 decode 吞吐 +2.0~4.4%。

## 动机

此前 ROCm 上 AITER 的 softmax MoE 路由(`FusedTopKRouter` 与 fused shared-expert router)一律使用遗留的 `topk_softmax` launcher。AITER 侧的 `topk_gating` 已具备 in-kernel selected-mass renormalization(ROCm/aiter#4460)、E=512 `prefill_n` dispatch(#5334),并可在同一 launch 内用 sigmoid 给 shared-expert 列打分(#6306),但 vLLM 尚未接入;小 batch 下每层每次 launch 白付约 4–5 µs。

## 优化点

`dispatch_topk_softmax_func` 改为接收本次 launch 的张量与 shared-expert 参数,由 `_aiter_topk_gating_supported` 逐 launch 判定:gating 行须连续且 token 数 ≤ `_AITER_TOPK_GATING_MAX_TOKENS`(4096);带 shared expert 时还要求 sigmoid 打分、数量为 1/2/4/8、`topk_weights`/`topk_indices` 行 stride 一致且 `topk_weights` 宽度容纳 shared 列。满足时在 CDNA3/CDNA4(gfx942/gfx950)返回新注册的 `rocm_aiter_topk_gating` custom op(内部直接调 `aiter.ops.topk.topk_gating`,renorm 在 kernel 内完成,免去二次 pointwise kernel),否则回退 legacy `topk_softmax`;其他架构一律不变。4096 上限来自实测:T≥2048 与 legacy 持平,8K prefill chunk 更慢。无新增 env var,随 `VLLM_ROCM_USE_AITER`(_MOE)默认生效。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/router/fused_topk_router.py: dispatch 门控与 4096-token 上限
vllm/_aiter_ops.py: 注册 rocm_aiter_topk_gating custom op;is_topk_gating_enabled 限 CDNA3/4
vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py: FSE 路由传入 num_fse 与 sigmoid
tests/kernels/moe/test_rocm_aiter_topk.py: 数值/共享专家 suffix/torch.compile 兼容测试
```

```python
if (
    not gating_output.is_contiguous()
    or gating_output.shape[0] > _AITER_TOPK_GATING_MAX_TOKENS
):
    return False
if num_shared_experts == 0:
    return not shared_expert_scoring_func
return (
    shared_expert_scoring_func == "sigmoid"
    and num_shared_experts in (1, 2, 4, 8)
    and topk_weights.stride(0) == topk_indices.stride(0)
    and topk_weights.shape[-1] >= topk_indices.shape[-1] + num_shared_experts
)
```

## 性能声明

- Kernel(AITER op test,MI355X,E=512/top-10/bf16/renorm/1 shared expert):`topk_gating` 在 T≤1024 比 legacy `topk_softmax` 快 37–47%(T=256:6.05 vs 11.42 µs,−47%),T=2048/4096 持平(+0.5%/−2.3%),0 routing mismatch。
- E2E gfx950(MI355X,`VLLM_ROCM_USE_AITER=1`):Flash-Next MXFP4 TP2 六行 output-throughput geomean +0.7%~+1.5%(FSE=1/0 两套);Qwen3-30B-A3B bf16 TP1 四行 geomean +1.3%~+1.9%;1k/1k 并发 1 各 cell +2.0%~+4.4% 收益最大。
- gfx942(MI300-class,较轻验证):Qwen3-30B geomean +0.45%(mbt 2048)/+2.0%(mbt 4096);Flash-Next-FP8 TP2 三行 +1.4%~+4.7%。
- GSM8K(1319 题,5-shot,配对 exact McNemar):9 个 cell 无方向性变化,最低 p=0.122。

## 验证与回归风险

新增 `test_rocm_aiter_topk_gating_matches_softmax_reference`(含 Qwen3.8 E=512/k=10)、shared-expert suffix 测试与 torch.compile opcheck;`test_topk_softmax_dispatch_aiter_topk_gating` 对非连续行、T=4097、不支持的 shared 数量、宽度/stride 不匹配等逐一钉死必须回退 legacy;`test_topk_gating_enabled_by_cdna` 钉死 CDNA 版本门控(MI355X 上 93 passed)。风险:默认行为变更且无 env 开关,影响 CDNA3/4 上所有 softmax 路由 launch(PR 称实际隐式限于 Qwen 系模型);T>4096 或形状不满足时回退 legacy,行为不变;gfx942 验证强度低于 gfx950。

## 关联

依赖上游 ROCm/aiter#4460(in-kernel renorm)、#5334(E=512 prefill_n)、#6306(shared-expert sigmoid);相关 vLLM PR:#57246(fused_topk_bias 的 eager fallback 修复)、#55147(topk_ids buffer 宽度修复)。索引中同区域另有 pr-22211(topk softmax fused kernel)、pr-62172(cooperative topk)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
