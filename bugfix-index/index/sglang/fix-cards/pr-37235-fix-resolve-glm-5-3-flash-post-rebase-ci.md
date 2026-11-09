# PR #37235: fix: resolve GLM-5.3-flash  post-rebase CI regressions

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#37235](https://github.com/sgl-project/sglang/pull/37235) |
| 作者 | Fridge003 |
| 合并日期 | 2026-08-31 |
| labels | (none) |
| 热度 | 👍 0 · 💬 0 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
GLM-5.3 rebase 后调用点未迁移到新 runtime-context API 致 CI 回归;迁移访问点并修复 declare_resolution 参数错位。

## 问题现象
正文未描述具体用户侧现象,仅在 Summary 中列出 rebase 后的 CI 回归:不兼容的 breakable CUDA graph 启动路径会因 `declare_resolution` 参数被交换而崩溃;GLM-5.3 与 DSA 相关调用点、shared-expert fusion 行为未对齐;CPU-only fixtures 因 conditional FP8 symbols 与新增必需的 runtime 字段而失败。共 97 个受影响单元测试受波及,修复后在 RX B300 devbox 上全部通过(pre-commit hooks 以 Python 3.12 通过)。

## 根因
正文 Summary 披露:rebase 之后 GLM-5.3 与 DSA 的调用点仍走旧的 server_args 全局访问方式,未迁移到 resolved runtime-context accessors;GLM-5.3 的 shared-expert fusion 判定与 post-rebase 的 loader 决策不一致;`declare_resolution` 的两个实参顺序被写反,导致 breakable CUDA graph 启动路径崩溃。

## 修复方案
按正文 Summary 与 diff:(1) 将 GLM-5.3 与 DSA 调用点迁移到 resolved runtime-context accessors——`glm5_next.py` 改用 `get_spec()/get_parallel()/get_mm()`,`disaggregation/utils.py` 的 `should_bypass_dsa_cp_prefix_cache()` 去掉 server_args 参数改读 `get_disagg()/get_parallel()`,`dsa_indexer_kpool.py` 改用 `get_device()`,encoder preprocessor 改用 `get_parallel().attn_tp_group` 等;(2) `glm5_next.py` 的 `determine_num_fused_shared_experts` 改为直接依据 `is_shared_experts_fusion_disabled()` 对齐 loader 决策;(3) 修正 `declare_resolution` 实参顺序,并补齐缺失的 runtime 字段等测试 fixture。关键摘录:

```diff
             declare_resolution(
-                "_disable_breakable_cudagraph_if_incompatible",
                 server_args,
+                "_disable_breakable_cudagraph_if_incompatible",
                 cuda_graph_config=with_phase(
                     cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                 ),
```

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围,仅表明为 GLM-5.3 rebase(基于 #36507)后的回归修复。
- 回归面:按 diff,涉及 `arg_groups/cuda_graph_hook.py`(breakable CUDA graph 启动)、`disaggregation/encoder/preprocessor.py`(视频多模态预处理的 attn TP 分片)、`disaggregation/utils.py`(DSA CP 前缀缓存绕过判断)、`layers/attention/dsa/dsa_indexer_kpool.py`(rope device 获取)、`mem_cache/kv_cache_builder.py`、`models/glm5_next.py`(GLM-5.3 初始化与 shared-expert fusion),以及 nixl、fp8 MoE runner、scheduler、tokenizer_manager、eagle worker 等多处单测 fixture。

## 关联
- Stack(Base): [#36507](https://github.com/sgl-project/sglang/pull/36507)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
