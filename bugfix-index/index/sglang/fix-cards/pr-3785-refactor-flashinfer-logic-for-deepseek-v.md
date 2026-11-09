# PR #3785: Refactor flashinfer logic for deepseek v3 and fix accuracy bug

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#3785](https://github.com/sgl-project/sglang/pull/3785) |
| 作者 | Fridge003 |
| 合并日期 | 2025-02-24 |
| labels | high priority |
| 热度 | 👍 4 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论

启用 flashinfer MLA 的 DeepSeek 模型出现精度 bug:MLA 的 attention scaling 未应用 yarn rope scaling 的 mscale;新增独立 `flashinfer_mla_backend` 并在 model_config 中正确计算含 yarn 的 scaling 修复。

## 问题现象

issue #3716 与 #3751 报告:启用 flashinfer MLA(`--enable-flashinfer-mla`)时出现 accuracy bug(表现为模型输出精度下降)。正文未给出具体错误输出示例。修复后正文给出的验证(DeepSeek-R1,TP8,`--enable-flashinfer-mla`):gsm8k 8-shot 精度 0.960、mmlu 精度 0.871,未启用 flashinfer mla 的 baseline 可参考 #3486。

## 根因

正文披露:精度 bug 的原因是启用 flashinfer MLA 时 **rope scaling with yarn 未被正确处理**(在 @yzh119 协助下定位)。机制(由 diff 推断):MLA 模型的 attention softmax scale 应为 `1/sqrt(qk_nope_head_dim + qk_rope_head_dim)`,且当模型配置带 rope_scaling(如 yarn)时还须乘以 `yarn_get_mscale(factor, mscale_all_dim)` 的平方;此前 model_config 对 MLA 分支根本没有计算 `scaling`(以及 `qk_nope_head_dim`、`v_head_dim` 字段),下游 kernel 拿到的 sm_scale 缺少 yarn mscale 因子,长上下文(yarn)模型输出即出错。

## 修复方案

- `configs/model_config.py`:DeepSeek MLA 分支补充 `qk_nope_head_dim`、`v_head_dim`,计算 `self.scaling = 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`;存在 rope_scaling 时按 yarn 公式修正:

```diff
+            # Handle rope scaling with yarn
+            self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
+            if self.hf_config.rope_scaling:
+                mscale_all_dim = self.hf_config.rope_scaling.get(
+                    "mscale_all_dim", False
+                )
+                scaling_factor = self.hf_config.rope_scaling["factor"]
+                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
+                self.scaling = self.scaling * mscale * mscale
```

并新增 `yarn_get_mscale(scale, mscale)`(scale<=1 返回 1.0,否则 `0.1 * mscale * log(scale) + 1.0`)。

- 新建 `layers/attention/flashinfer_mla_backend.py`,从 `flashinfer_backend.py` 中提取 MLA 逻辑定义 `FlashInferMLAAttnBackend`:radix cache 开启时 prefill/decode 均用 `BatchMLAPagedAttentionWrapper`,关闭时 prefill 用 ragged/paged prefill wrapper;移除与 MLA 无关的代码与 magic numbers,使 DeepSeek v2 亦可支持;`backend="auto"` 以使用 flashinfer 最新 fa3 后端。
- `model_runner.py`:`enable_flashinfer_mla` 时 attention backend 由 `"flashinfer"` 改为新的 `"flashinfer_mla"` 并注册。
- `models/deepseek_v2.py`:forward 中 `no_absorb()` 决策重构——flashinfer MLA 仅在 `disable_radix_cache` 且 extend 时走 forward_normal,其余走 forward_absorb(原逻辑在开启 flashinfer mla 且未禁用 radix cache 的 extend 时也走 absorb 分支,与之前不同)。

## 影响与回归风险

- 影响窗口:正文未说明,不写。
- 回归面:按 diff,`--enable-flashinfer-mla` 的 DeepSeek V2/V3 系列 MLA 模型改走全新 backend 文件,attention scaling 计算在 model_config 层变化(无 yarn 配置时 scaling 为标准公式值,yarn 模型的 sm_scale 变大);非 MLA 模型与 triton backend 不受影响。deepseek_v2 forward 分支条件重写影响该架构在 flashinfer_mla/其他 backend 下的 normal/absorb 路径选择。

## 关联

- issue [#3716](https://github.com/sgl-project/sglang/issues/3716)、issue/PR [#3751](https://github.com/sgl-project/sglang/issues/3751)(accuracy bug 报告)
- PR [#3486](https://github.com/sgl-project/sglang/pull/3486)(未启用 flashinfer mla 的 baseline 参考)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
