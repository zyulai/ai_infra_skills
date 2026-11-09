# PR #31001: Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31001](https://github.com/sgl-project/sglang/pull/31001) |
| 作者 | kpham-sgl |
| 合并日期 | 2026-07-13 |
| labels | high priority, deepseek, post version patch |
| 热度 | 👍 1 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
modelopt_fp4 + flashinfer_trtllm 下 fp32 路由 bias 触发 flashinfer tie 处理 NaN,长上下文输出全为「!」;恢复 bias 的 bf16 cast。

## 问题现象
以 `--quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm` 服务 `nvidia/GLM-5.2-NVFP4` 时,长上下文请求输出全是 `!!!!!!...`(每个输出 token 的 logprob 均为 null,即 NaN);短请求与 FP8 checkpoint 不受影响。在 4×B300(SM 10.3)、flashinfer 0.6.12、torch 2.11.0+cu130 上用 issue 的 `openai/graphwalks` probe 复现:10k/20k/40k/80k chars 全部 COLLAPSED、`null_logprobs=40/40`;修复后全部 ok、`null_logprobs=0`。

## 根因
正文披露:`MoEGate.__init__` 为 `modelopt_fp4` 创建 fp32 的 `e_score_correction_bias`,FP4 moe_runner 未转换就把该 bias 作为 `routing_bias` 传给 flashinfer 的 `trtllm_fp4_block_scale_moe`(其文档约定要求 bfloat16,kernel 不校验 dtype)。fp32 bias 下,当多个 expert 的选择键 `sigmoid(logit) + bias` 按位相等、tie 组跨 top-k 边界时,flashinfer 的 DeepSeekV3 路由 kernel 输出 NaN expert 权重。GLM-5.2-NVFP4 的 bias 只有 196/256 个唯一值(都约 11.13–11.32),长上下文 token 的 256 个 router logits 都很负时,sigmoid 贡献低于 ~11.3 bias 的 fp32 ulp,加法中被吞掉,key 塌缩成重复 bias 值 → top-8 边界按位 tie → NaN,并传播到所有 logits;greedy argmax 退化为 token id 0(GLM 词表中即 `!`)。这是 PR #29783 移除 `modelopt_fp4 + flashinfer_trtllm` 的 bf16 cast 造成的回归;FP8 路径本就 cast 到 bf16,FP4 路径原样传入。

## 修复方案
`python/sglang/srt/models/deepseek_v2.py` `MoEGate.__init__`:当 `quant_config.get_name() == "modelopt_fp4"` 且 moe runner backend 为 `flashinfer_trtllm` 时,把 `correction_bias_dtype` 设为 `torch.bfloat16`,镜像 FP8 路径的 cast。bf16 粗量化(11.3 处 ulp 0.0625)使 bias 值间距远大于任何 sigmoid 贡献,无法形成精确 tie。根因中的 tie 处理与缺失 dtype 校验属于 flashinfer `routingCustom` 的问题,本 PR 未改。

```diff
+                if (
+                    quant_config.get_name() == "modelopt_fp4"
+                    and get_moe_runner_backend().is_flashinfer_trtllm()
+                ):
+                    correction_bias_dtype = torch.bfloat16
```

## 影响与回归风险
- 影响窗口:自 PR #29783 移除该 cast 起至本修复合并前;仅影响 modelopt_fp4 + flashinfer_trtllm 组合(GLM/DeepSeek 等 DeepSeekV3 式路由)。
- 回归面:deepseek_v2.py 中 MoEGate correction bias 的 dtype 选择;该分支仅在 fp4+trtllm 组合下生效,不触碰其他量化/后端组合的 bias dtype;bias 转为 bf16 可能影响使用该 bias 数值精度的路径(与既有 FP8 路径行为一致)。

## 关联
- https://github.com/sgl-project/sglang/issues/30989 (修复对象)
- https://github.com/sgl-project/sglang/pull/29783 (回归来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
