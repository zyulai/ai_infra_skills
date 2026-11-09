# PR #14936: Fix accuracy issue when using a16w16 mla_decode_fwd

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14936](https://github.com/sgl-project/sglang/pull/14936) |
| 作者 | kkHuang-amd |
| 合并日期 | 2025-12-16 |
| labels | amd, run-ci |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Other |
| bug 类型 | correctness |

## 一句话结论
persistent 版 mla_decode_fwd kernel 不支持 nhead=128 的 a16w16,导致精度崩坏(GSM8K 0.009);在 aiter 后端按条件回退到非 persistent kernel,恢复到 0.935。

## 问题现象
使用 a16w16 的 `mla_decode_fwd`(persistent 版本)时出现严重精度问题,由 PR #13147 暴露。作者给出 GSM8K 对比:修复前 Accuracy: 0.009、Invalid: 0.058、Latency: 133.917 s(输出几乎完全不可用);修复后 Accuracy: 0.935、Invalid: 0.000、Latency: 92.142 s。正文未附具体报错日志。

## 根因
正文披露:"Current persist mla_decode_fwd kernel does not support nhead=128 a16w16."——当前的 persistent(常驻/ps)版 `mla_decode_fwd` kernel 不支持 nhead=128 的 a16w16 场景,却仍被使用,因此产生精度错误;处理方式是回退到非 persistent 版 kernel("We need to fall back to non-persist mla_decode_fwd kernel")。

(由 diff 推断)回退条件为 `kv_cache_dtype is not fp8_dtype and self.enable_dp_attention`,即 KV cache 非 fp8(a16w16 对应的路径)且开启 DP attention 时,将模块级开关 `_use_mla_ps_kernel`、`fast_mode`、`intra_batch_mode` 全部置 False,使后续走非 persistent kernel;`max_split_per_batch` 也随之从 32 变为 None。

## 修复方案
修改 `python/sglang/srt/layers/attention/aiter_backend.py` 的 `__init__`(use_mla 分支):新增条件判断,在不满足 persistent kernel 支持条件的组合下关闭 persistent kernel 相关全局开关;顺带把 `enable_dp_attention` 的赋值上移。附注释 "current persist a16w16 mla_decode kernel does not support head_num = 128, need to fall back to non-persist"。

关键 diff 摘录:

```diff
+            global _use_mla_ps_kernel, fast_mode, intra_batch_mode
+
+            # current persist a16w16 mla_decode kernel does not support head_num = 128
+            # need to fall back to non-persist
+            if self.kv_cache_dtype is not fp8_dtype and self.enable_dp_attention:
+                _use_mla_ps_kernel = False
+                fast_mode = False
+                intra_batch_mode = False
```

作者验证(同上问题现象一节的 GSM8K 数字):Accuracy 从 0.009 提升到 0.935,Invalid 从 0.058 降到 0.000,Latency 还从 133.917 s 降到 92.142 s。

## 影响与回归风险
- 影响窗口:由 PR #13147 引入相关 persistent kernel 使用路径起(正文以此 PR 为问题暴露点;未给出明确版本)。
- 回归面:影响 aiter 注意力后端开启 MLA 且 DP attention、KV cache 非 fp8 的配置:该组合强制回退非 persistent `mla_decode_fwd`,性能特征改变(本例 latency 反而更好);fp8 KV cache 或未开 DP attention 的组合不受影响。修改的是模块级全局变量(`_use_mla_ps_kernel`/`fast_mode`/`intra_batch_mode`),若同进程存在多个 model_runner 实例,任一实例命中条件都会影响全局开关(由 diff 推断)。

## 关联
- #13147(暴露精度问题的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
