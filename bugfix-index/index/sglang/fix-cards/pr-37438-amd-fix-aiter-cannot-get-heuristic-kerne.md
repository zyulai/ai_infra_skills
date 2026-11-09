# PR #37438: [AMD] fix aiter cannot get heuristic kernel regression

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37438](https://github.com/sgl-project/sglang/pull/37438) |
| 作者 | billishyahao |
| 合并日期 | 2026-09-01 |
| labels | amd, run-ci |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Attention / Kernel |
| bug 类型 | regression |

## 一句话结论
#34647 在未选中 gluon kernel 时也禁用 ps,aiter asm 路径取不到 heuristic kernel 而崩溃;改为仅选中 gluon 时禁用。

## 问题现象
AMD aiter 后端回归:使用如下环境组合启动 server 即崩溃——`SGLANG_AITER_MLA_GLUON=0`(关闭 gluon、走 aiter asm)、`SGLANG_USE_AITER=1`、`SGLANG_AITER_K3_OPT=1`、`AITER_FLYDSL_FORCE=1`、`AITER_SITUV2_A8W4=1`,模型 Kimi-K3,`--prefill-attention-backend aiter --decode-attention-backend aiter --kv-cache-dtype fp8_e4m3`、tp-size 8。报错:

```
[AITER] asm_mla.cu:193 get_heuristic_kernel_mla:
    cannot get heuristic kernel! q_type:bf16 kv_type:fp8 gqa:16 ps:0 prefill:0
    causal:1 qseqlen:1 lse:0 cprr:0
  Fatal Python error: Aborted
```

影响面:此前可用的 aiter asm 路径(非 gluon)在 #34647 合并后无法启动。

## 根因
正文披露:回归由 #34647 引入——该 PR 将 gluon 引入 aiter 后端并默认启用 `SGLANG_AITER_MLA_GLUON`,同时错误地(wrongly)在 gluon kernel 未被选中时也禁用了 ps,导致 asm 路径回退到不匹配的 aiter kernel,heuristic 查找失败并 abort。

## 修复方案
`python/sglang/srt/layers/attention/aiter_backend.py`:zero-pad topology(`head_pad_mode == "zero"` 且 kv 为 fp8)分支下,原先无条件禁用 ps(`_use_mla_ps_kernel = False; fast_mode = False; intra_batch_mode = False`),现改为仅当 `prefer_mla_gluon_decode(...)` 判定选中 gluon kernel 时才禁用,避免回退到错误的 aiter kernel:

```diff
-                _use_mla_ps_kernel = False
-                fast_mode = False
-                intra_batch_mode = False
+                # Disable ps only when gluon kernel is selected to avoid falling
+                # back to incorrect aiter kernel
+                if prefer_mla_gluon_decode(
+                    head_pad_mode=self.head_pad_mode,
+                    num_head=self.num_head,
+                    kv_cache_dtype=self.kv_cache_dtype,
+                ):
+                    _use_mla_ps_kernel = False
+                    fast_mode = False
+                    intra_batch_mode = False
```

## 影响与回归风险
- 影响窗口:自 #34647 合并(gluon 默认开启)起至本修复。
- 回归面:aiter attention 后端在 zero-pad + fp8 KV 场景的 kernel 选择逻辑——gluon 未选中时恢复 ps kernel 路径(`_use_mla_ps_kernel` 不再被错误关闭,`max_split_per_batch` 相应取 32);gluon 选中时行为与回归版本一致。作者给出精度验证:gsm8k(1300) Accuracy 0.950、Invalid 0.001、Output throughput 458.604 token/s。

## 关联
- 引入回归的 PR:https://github.com/sgl-project/sglang/pull/34647

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
