# PR #31065: [Cherry-pick to release/v0.5.15] Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing) (#31001)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#31065](https://github.com/sgl-project/sglang/pull/31065) |
| 作者 | Kangyan-Zhou |
| 合并日期 | 2026-07-13 |
| labels | deepseek, cherry-pick |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
NVFP4 + flashinfer_trtllm 下 e_score_correction_bias 保持 fp32,平局时路由 NaN、长文输出塌缩成 "!!!!";镜像 fp8 路径 cast 为 bf16。

## 问题现象
GLM/DeepSeek 使用 NVFP4(modelopt_fp4)量化并搭配 flashinfer_trtllm MoE 运行后端时,长上下文场景输出塌缩为 "!!!!" 字符串(标题所述 "!!!!" collapse),根因是路由出现 NaN。本 PR 是将 main 上的修复(cherry-pick 自 #31001 的 commit `f49cbbd67dea602f8616892d2a9882c8c30ae942`)回移到 `release/v0.5.15` 分支,由 cherry-pick 工作流自动创建;正文除 cherry-pick 元信息外未再描述症状细节。

## 根因
正文未披露根因分析(仅有标题点出 NaN routing)。

从 diff 可以描述改动所暗示的机制:diff 中新增的代码注释写明「flashinfer trtllm routing requires a bf16 routing_bias; an fp32 bias yields NaN routing on exact ties」,即 modelopt_fp4 量化路径下 `e_score_correction_bias` 未像 fp8 路径那样被 cast 成 bf16,保持 fp32 的 correction bias 传入 flashinfer trtllm 路由后,在打分完全平局(exact ties)时产生 NaN 路由,导致长上下文输出塌缩。(由 diff 推断,注释为 diff 内代码注释)

## 修复方案
改 `python/sglang/srt/models/deepseek_v2.py` 中 `e_score_correction_bias` 的 dtype 选择逻辑:在已有的 quark 等分支之后,新增当 `quant_config.get_name() == "modelopt_fp4"` 且 `get_moe_runner_backend().is_flashinfer_trtllm()` 时,把 `correction_bias_dtype` 设为 `torch.bfloat16`,镜像 fp8 路径已有的 cast。

```diff
+                # NOTE(kpham-sgl): flashinfer trtllm routing requires a bf16
+                # routing_bias; an fp32 bias yields NaN routing on exact ties.
+                # Mirror the fp8 path's cast.
+                if (
+                    quant_config.get_name() == "modelopt_fp4"
+                    and get_moe_runner_backend().is_flashinfer_trtllm()
+                ):
+                    correction_bias_dtype = torch.bfloat16
             self.e_score_correction_bias = nn.Parameter(
                 torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
             )
```

## 影响与回归风险
- 回归面:影响 modelopt_fp4(NVFP4)量化 + flashinfer_trtllm MoE 后端组合下 DeepSeek 系模型的 `e_score_correction_bias` 参数 dtype(由 fp32 变为 bf16),路由打分数值路径随之改变;其他量化方式与 MoE 后端组合不进入该分支,行为不变。本 PR 仅面向 release/v0.5.15,main 分支修复见 #31001。

## 关联
- https://github.com/sgl-project/sglang/pull/31001 (源 PR,main 分支上的原始修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
