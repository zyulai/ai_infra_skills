# PR #14897: Fix dsv3 dp accuracy issue when using bf16-kv

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14897](https://github.com/sgl-project/sglang/pull/14897) |
| 作者 | Duyi-Wang |
| 合并日期 | 2025-12-11 |
| labels | (none) |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
DeepSeek-V3 开 DP attention 且用非 fp8(bf16)KV 时 aiter MLA 精度出错;此时关闭 _use_mla_ps_kernel/fast_mode/intra_batch_mode 三个加速开关。

## 问题现象
标题指向的问题:DeepSeek-V3(dsv3)在 DP attention 且使用 bf16 KV cache 时出现精度问题。正文未描述具体现象、数值或复现步骤——本 PR 只是把 #14895("Fix dp issue dsv3 bf16 kv")cherry-pick 到 `amd_mori` 分支,Motivation 与 Modifications 节均为空模板。

## 根因
正文未披露根因。(由 diff 推断)`aiter_backend.py` 的 MLA 分支里,`_use_mla_ps_kernel`、`fast_mode`、`intra_batch_mode` 这三个模块级加速开关此前不区分 KV cache dtype 一律保持启用;当 KV cache 不是 fp8(如 bf16)且开启 DP attention 时,这些 kernel/模式与该配置不兼容,产出错误结果。因此修复按条件显式关闭它们。

## 修复方案
改 `python/sglang/srt/layers/attention/aiter_backend.py` 的 `__init__`(`use_mla` 分支):把 `enable_dp_attention` 的赋值提前,并新增判断——KV cache dtype 非 fp8 且启用 DP attention 时,把三个全局标志置 False:

```diff
         if self.use_mla:
+            global _use_mla_ps_kernel, fast_mode, intra_batch_mode
+            self.enable_dp_attention = is_dp_attention_enabled()
+
+            if self.kv_cache_dtype is not fp8_dtype and self.enable_dp_attention:
+                _use_mla_ps_kernel = False
+                fast_mode = False
+                intra_batch_mode = False
+
             self.qo_indptr_ = torch.zeros(
...
-            self.enable_dp_attention = is_dp_attention_enabled()
```

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:aiter 后端 MLA 路径。非 fp8 KV + DP attention 的配置从此退回非 ps-kernel/非 fast/非 intra-batch 的默认(更保守)路径,可能影响该配置下的性能;fp8 KV 或未开 DP attention 的配置判断不成立,行为不变。注意这三个是模块级全局变量,条件满足时会影响后续所有 forward 的模式选择。

## 关联
- #14895(cherry-pick 来源:"Fix dp issue dsv3 bf16 kv")

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
