# PR #37366: Fix GLM-5.3 Flash CI regressions after latest main merge

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37366](https://github.com/sgl-project/sglang/pull/37366) |
| 作者 | Fridge003 |
| 合并日期 | 2026-09-01 |
| labels | memory-pool |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
修复 latest main 合并后的批量 CI 回归:hybrid 后端可选属性改安全读取,HiSparse KV pool 与 MXFP8 CPU offload 接口对齐。

## 问题现象
latest main 合并后出现一批 CI 测试失败(regression)。正文在「Failures addressed」列出四条失败流水线的 job 链接:Base CPU、Base B200、Extra H200、Extra B200(来自 runs 33474106882 与 33474106554)。正文未给出具体报错文本或日志,也未描述服务侧运行症状;修复内容按 fix 条目列出。验证时「merging current main 后两个 wrapper regression 文件通过:6 tests and 7 subtests」,可见失败集中在 wrapper regression 测试文件。

## 根因
正文未披露根因,仅给出修复摘要。从 diff 可推断:(由 diff 推断)
- `hybrid_linear_attn_backend.py` 中直接以属性访问读取 `linear_attn_backend.supports_overlap_plan_stream_graph_load`,对未暴露该属性的后端或 test double 会抛 AttributeError,对应正文「default the optional hybrid overlap-plan capability to disabled for backends and test doubles that do not expose it」;
- HiSparse KV pool 子类 `__init__` 未接收/未向父类转发 DSA KPool 与 tail-layout 相关新参数(`index_kpool`、`index_kpool_compress`、`tail_extra_slots`、`max_running_requests`、`skip_topk_layers`),构造签名不匹配;
- MXFP8 KV CPU offload 方法 `get_cpu_copy`/`load_cpu_copy` 缺少 `req_pool_index` 参数,与 request-aware pool 接口不一致。

## 修复方案
改动三处(均在 `python/sglang/srt/`):
1. `layers/attention/hybrid_linear_attn_backend.py`:将 `supports_overlap_plan_stream_graph_load` 的直接属性访问改为 `getattr(..., False)`,未暴露该属性时默认禁用该 hybrid overlap-plan 能力。
2. `mem_cache/hisparse_memory_pool.py`:子类 `__init__` 新增 `index_kpool`、`index_kpool_compress`、`tail_extra_slots`、`max_running_requests`、`skip_topk_layers` 参数并转发给父类(对应「forward DSA KPool and tail-layout arguments through the HiSparse KV pool subclass」)。
3. `mem_cache/memory_pool.py`:`get_cpu_copy`/`load_cpu_copy` 增加 `req_pool_index=None` 参数,与 request-aware pool 接口对齐。

关键 diff 摘录(hybrid_linear_attn_backend.py):
```diff
-            linear_attn_backend.supports_overlap_plan_stream_graph_load
+            getattr(
+                linear_attn_backend,
+                "supports_overlap_plan_stream_graph_load",
+                False,
+            )
```
正文另提到「refresh the hybrid graph-metadata test fake with the KV index translator field」,但本卡所含 diff 中未见对应测试文件改动。

## 影响与回归风险
- 回归面:按 diff 实际改动,(a) hybrid linear attention backend 的构造与 overlap plan stream graph load 判定路径——缺属性的后端从报错变为能力默认禁用;(b) HiSparse 内存池构造参数转发,涉及 HiSparse/GLM 相关路径;(c) MXFP8 KV CPU offload 的 `get_cpu_copy`/`load_cpu_copy` 调用方(新参数带默认值,旧调用签名兼容)。正文 CI policy 注明本 PR 为 draft,验证依赖 devbox 与 targeted rerun-test,不加 run-ci。

## 关联
正文未列出(仅给出 CI run/job 链接,无 issue/PR 关联)。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
