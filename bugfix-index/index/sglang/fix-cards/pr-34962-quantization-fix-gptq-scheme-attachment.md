# PR #34962: [Quantization] Fix GPTQ scheme attachment broken by LinearBase.scheme default

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34962](https://github.com/sgl-project/sglang/pull/34962) |
| 作者 | mmangkad |
| 合并日期 | 2026-08-16 |
| labels | run-ci, bypass-fastfail, run-ci-extra |
| 热度 | 👍 1 · 💬 3 |
| 分类 | Quantization |
| bug 类型 | regression |

## 一句话结论
#29328 给 LinearBase 加了类级 `scheme = None`,GPTQ 的 `hasattr(layer, "scheme")` 守卫恒真、scheme 永不构建,所有 GPTQ checkpoint 加载即挂;探针改 `is None` 并在其余可量化层基声明默认值。

## 问题现象
#29328 合入后,每个 GPTQ checkpoint 加载即死,报错 `'NoneType' object has no attribute 'create_weights'`——GPTQ 的 per-layer scheme 从未被构建。

## 根因
正文披露:#29328 给 `LinearBase` 加了类级 `scheme = None` 默认值,使 GPTQ 里 `if not hasattr(layer, "scheme")` 这个守卫变成 no-op(hasattr 恒为 True),`create_weights` 因此跳过 scheme 构建,后续调用 `layer.scheme.create_weights` 即 NoneType 报错。修复后探针改为 `layer.scheme is None`,类级默认值因此成为"承重"语义,GPTQ 会被交给的其他两个层基——量化 `ParallelLMHead` 对应的 `VocabParallelEmbedding`、gptq_marlin MoE 对应的 `FusedMoE`——也必须声明同样的默认;`FusedMoE` 两处 `hasattr` 的 weight-loader 读取同步翻转,否则重载时所有未量化与 fp8 MoE 层会拿到 `method = None`。

## 修复方案
- `gptq.py` 四处 `create_weights` 的守卫由 `if not hasattr(layer, "scheme")` 改为 `if layer.scheme is None`(linear 与 moe、marlin 变体一并修改)。
- `linear.py`:更新 `LinearBase.scheme = None` 的注释,说明默认值支撑 `is None` 探针、scheme 必须是普通对象(nn.Module.__setattr__ 会把 Module 值塞进 _modules 被默认值遮蔽)。
- `fused_moe_triton/layer.py`:`FusedMoE` 声明 `scheme = None`;`_weight_loader_impl` 与 `weight_loader_fused` 两处 `if hasattr(self, "scheme")` 改为 `if self.scheme is not None`。
- `vocab_parallel_embedding.py`:`VocabParallelEmbedding` 声明 `scheme = None`。
- 新增回归单测 `test_gptq_scheme_attach.py`:GPTQ linear 层建出 scheme 且有 `qweight`;断言三个层基的类默认均为 None。

关键 diff(gptq.py):

```diff
-        if not hasattr(layer, "scheme"):
+        if layer.scheme is None:
             layer.scheme = self.quant_config.get_linear_scheme(layer)
```

## 影响与回归风险
- 影响窗口:正文未给版本号(回归由 #29328 引入,其合并后至本 PR 前所有 GPTQ 加载受影响)。
- 回归面:量化方法与层基的 scheme 附加协议从 hasattr 探针统一改为 `is None` 探针;`FusedMoE` 的 weight loader 在 `self.scheme is not None` 时用 scheme 作为 method——未量化/fp8 MoE 层 scheme 为 None 时回落 quant_method,行为同修复前(按 diff 与正文陈述);依赖 `hasattr(layer, "scheme")` 判断的其他代码路径不在本 PR 范围内。

## 关联
- https://github.com/sgl-project/sglang/pull/29328

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
