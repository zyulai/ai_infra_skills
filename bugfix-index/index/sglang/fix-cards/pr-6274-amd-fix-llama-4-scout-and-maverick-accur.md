# PR #6274: [AMD] Fix Llama 4 Scout and Maverick accuracy issues on MI300X

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#6274](https://github.com/sgl-project/sglang/pull/6274) |
| 作者 | hubertlu-tw |
| 合并日期 | 2025-05-14 |
| labels | (none) |
| 热度 | 👍 1 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
MI300X 上开启 `SGLANG_AITER_MOE=1` 时 Llama 4 Scout/Maverick 精度异常;在 FusedMoE AITER 路径补齐 `apply_router_weight_on_input` 的路由权重处理,mmlu_pro 精度恢复到官方水平。

## 问题现象
AMD GPU (MI300X) 上运行 meta-llama/Llama-4-Scout-17B-16E-Instruct 与 meta-llama/Llama-4-Maverick-17B-128E-Instruct 出现精度问题。触发条件为 `SGLANG_AITER_MOE=1`(如 `SGLANG_AITER_MOE=1 python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct/ --port 30000 --tp 8 --mem-fraction-static 0.8 --context-length 65536`),再用 lm_eval 跑 `mmlu_pro` 评测。影响面是这两个 Llama 4 模型在 AMD MI300X 上的推理精度。

## 根因
正文披露:带 `SGLANG_AITER_MOE=1` 的 `FusedMoE` 路径在 `apply_router_weight_on_input=True` 时存在问题。(由 diff 推断)具体机制是 CK 两段式 MoE kernel 路径没有处理「把路由权重施加在输入上」这一语义:修复前直接把原始 `topk_weights` 传给 `ck_moe_2stages`,路由权重未被施加到输入上,导致输出错误。修复方式为在调用 kernel 前手动将输入 `x` 乘以 `topk_weights`(此时仅支持 topk=1),并把 `topk_weights` 改为全 1 的 FP32 张量传入 kernel(kernel 要求 FP32),避免权重被重复或错误施加。

## 修复方案
改 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的 `forward_cuda`:`_is_hip` 且 `SGLANG_AITER_MOE` 开启时,若 `apply_router_weight_on_input` 为真,先断言 `topk_weights` 形状为 (num_tokens, topk) 且 topk==1,再做 `x = x * topk_weights.to(x.dtype)`,并将 `topk_weights` 替换为 `torch.ones_like(topk_weights, dtype=torch.float32)`,然后照常调用 `ck_moe_2stages`。关键 diff 摘录:

```diff
             if _is_hip and get_bool_env_var("SGLANG_AITER_MOE"):
                 assert not no_combine, "unsupported"
+            if apply_router_weight_on_input:
+                assert (
+                    topk_weights.dim() == 2
+                ), "`topk_weights` should be in shape (num_tokens, topk)"
+                _, topk = topk_weights.shape
+                assert (
+                    topk == 1
+                ), "Only support topk=1 when `apply_router_weight_on_input` is True"
+                x = x * topk_weights.to(x.dtype)
+                topk_weights = torch.ones_like(
+                    topk_weights, dtype=torch.float32
+                )  # topk_weights must be FP32 (float32)
```

修复后 PR 给出 mmlu_pro 评测结果(v0.4.6.post2-rocm630 + 本 PR,8*MI300X):Scout 75.0、Maverick 81.2,对照官方基准 74.3 / 80.5,SGLang 8*H100 为 75.2 / 80.7。

## 影响与回归风险
- 回归面:仅影响 HIP(AMD)上 `SGLANG_AITER_MOE=1` 的 `FusedMoE` 前向路径中 `apply_router_weight_on_input=True` 的模型;新断言把该路径限制为 topk==1,若此类模型 topk>1 会直接报 AssertionError;其他 MoE 路径与 NVIDIA 路径不受影响。

## 关联
- https://github.com/sgl-project/sglang/issues/5402 (正文称本 PR 亦解决该 issue)
- https://github.com/sgl-project/sglang/blob/main/docs/references/llama4.md (Ref)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
