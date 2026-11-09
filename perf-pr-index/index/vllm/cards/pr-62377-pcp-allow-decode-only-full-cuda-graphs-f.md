# PR #62377: [PCP] Allow decode-only full CUDA graphs for DSv3.2/GLM-5.2/5.3 sparse MLA with DCP=1

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LucasWilkinson |
| 合并 | 2026-10-19 |
| 分类 | CUDA Graph / Compiler(tier: major) |
| Label | ready, nvidia, mrv2, glm |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 1 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/62377 |

## 一句话结论

解除 MRV2 PCP+DCP=1 下 sparse-MLA 模型的 CUDA graph 禁令,GLM-5.3 GSM8K 评测提速 2.6 倍。

## 动机

MRV2 的 PCP(prefill-context parallel)路径此前对 sparse MLA(`index_topk`)模型一律要求 `cudagraph_mode=NONE`,即使模型运行在 DeepSeek-V3.2 attention 路径上也不能捕获 CUDA graph,decode 阶段只能走 eager,launch 开销显著。但在 DCP=1(decode-context-parallel 关闭)时,decode 请求在 PCP ranks 间 round-robin 分片,#53867 的 pure-decode full-graph 路径已能正确处理该形态,禁令过于保守。

## 优化点

改动集中在 `pcp_manager.validate_config` 的合法性检查:保留"sparse MLA + PCP + DCP=1 + 非 NONE"的拒绝逻辑,但把 `model_type` 为 `deepseek_v32` 或 `glm_moe_dsa`(覆盖 DeepSeek-V3.2 与 GLM-5.2/5.3)的模型从禁令中豁免,允许 `FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE`。纯 FULL 模式仍被拒绝,prefill 在 PCP 下继续走 piecewise;其他 `index_topk` 模型(DSv4/V4.1、MiniMax-M3 等)仍要求 NONE,因为需要模型侧改造(参见 #61555)。零运行时代码新增,纯解锁配置空间。

## 关键改动

```text
vllm/v1/worker/gpu/pcp_manager.py: validate_config 豁免 deepseek_v32/glm_moe_dsa 的 CUDA graph 禁令
```

```python
and getattr(model_config.hf_text_config, "model_type", None)
not in ("deepseek_v32", "glm_moe_dsa")
```

## 性能声明

GLM-5.3-NVFP4,4×B300,`-tp 1 -pcp 4 --enable-expert-parallel --kv-cache-dtype fp8`,`VLLM_USE_V2_MODEL_RUNNER=1`,GSM8K 1319 题、并发 64:`cudagraph_mode=NONE` 精度 0.9075、墙钟 379.5s、330 output tok/s;`FULL_AND_PIECEWISE` 精度 0.9083、墙钟 145.9s、859 output tok/s(2.6× faster,精度持平)。每 rank 捕获 11 个 FULL + 19 个 PIECEWISE graph。

## 验证与回归风险

`pytest tests/v1/worker/test_gpu_pcp_manager.py` 通过。风险点:DeepSeek-V3.2 与 GLM-5.3 共享 attention/indexer 代码但未做端到端验证;MTP 组合尚未验证;豁免按 `model_type` 白名单枚举,新 sparse-MLA 模型默认仍被拒,属保守策略;DCP>1 时 decode 分片形态不同,禁令仍然生效。

## 关联

前序:#53867(pure-decode full CUDA graph 路径,本 PR 依赖其正确性)。相关:#61555(DSv4.1 需要 model-specific 工作,故仍禁用);#47631 同样涉及 MiniMax-M3(sparse attention 模型,此处仍要求 NONE)。
