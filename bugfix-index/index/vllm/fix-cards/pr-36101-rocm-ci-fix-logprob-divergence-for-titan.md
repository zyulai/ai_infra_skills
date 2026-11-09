# PR #36101: [ROCm][CI] Fix logprob divergence for TitanML/tiny-mixtral under AITER rms_norm

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#36101](https://github.com/vllm-project/vllm/pull/36101) |
| 作者 | AndreasKaratzas |
| 合并日期 | 2026-03-09 |
| labels | rocm, ready |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | correctness |

## 一句话结论
未训练的 tiny-mixtral 在 AITER bf16 plain rms_norm 下 logprob 测试分叉,测试期间把该路径回退原生 float32 内核。

## 问题现象
两个测试对 `TitanML/tiny-mixtral` 持续 flake/失败:
```
FAILED models/language/generation/test_common.py::test_models[True-True-5-32-TitanML/tiny-mixtral]
FAILED models/language/generation/test_common.py::test_models[False-True-5-32-TitanML/tiny-mixtral]
```
报错形似 HF 与 vLLM 的 token 排名不一致,但不是逻辑 bug,而是只在该模型上暴露的数值精度问题。

## 根因
正文披露:`TitanML/tiny-mixtral` 随机初始化且从未训练,进入 unembedding 层的 hidden states 没有偏好方向,logit 分布在 32k 词表上接近均匀(达最大熵的 98%),top-1/top-2 token 概率差仅约 `O(1/sqrt(V)) ≈ 0.006 nats`。而 AITER 的 plain `rms_norm` 用 bfloat16 而非 float32 累加方差,在典型 post-norm 量级(约 2)下每次调用舍入误差为 `2^-6 = 0.015625`,已大于上述 rank gap;24 步 decode 中误差在 KV cache 里累积,最终翻转 argmax。vLLM 与 HF 输出分叉不是逻辑错误,而是两条路径踩在同一近平局的两侧。训练模型 rank-1/rank-2 gap 约 1-3 nats(约为 AITER 每步误差的 100 倍),不受影响。

## 修复方案
`tests/utils.py` 新增 `disable_aiter_plain_rmsnorm(monkeypatch)`:patch `dispatch_rocm_rmsnorm_func`,使 plain(非 fused)`rms_norm` 路径在测试期间回退原生 float32 内核;fused 路径(`rms_norm2d_with_add`,`with_fused_add=True` 时选用)仍留在 AITER,只重定向 plain 路径。`test_common.py` 在 `use_rocm_aiter=True` 且模型在 `AITER_MODEL_LIST` 时,对该模型显式关闭 AITER RMSNorm:
```python
 if use_rocm_aiter and (model in AITER_MODEL_LIST):
     monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
+    if model == "TitanML/tiny-mixtral":
+        # Untrained model: near-uniform logits make argmax sensitive to
+        # AITER's bfloat16 rounding error in plain rms_norm.
+        monkeypatch.setenv("VLLM_ROCM_USE_AITER_RMSNORM", "0")
```
另附 `test_custom.py` 共 5 个证明性测试,逐步验证因果链:随机权重产生近均匀 logits、AITER plain rms_norm 每步误差非零(~0.016)、原生内核对 float32 参考逐位一致、真实 checkpoint 同样近均匀(near-tie 9.2%)、误差逐层累积至 0.25。

## 影响与回归风险
- 回归面:改动仅限测试代码(`test_common.py`、`tests/utils.py`),不触及生产推理路径;helper 只重定向 plain rms_norm 分发,fused rms_norm 路径与其余 AITER 内核不受影响。

## 关联
正文未列出(仅 cc @kenroche @mawong-amd)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
