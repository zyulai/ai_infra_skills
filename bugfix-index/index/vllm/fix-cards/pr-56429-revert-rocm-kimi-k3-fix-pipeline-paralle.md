# PR #56429: Revert "[Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode  (#53664)"

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#56429](https://github.com/vllm-project/vllm/pull/56429) |
| 作者 | shen-shanshan |
| 合并日期 | 2026-09-11 |
| labels | rocm, kimi, k3 |
| 热度 | 👍 0 · 💬 4 |
| 分类 | Model-specific Fast Path |
| bug 类型 | regression |

## 一句话结论
回滚 #53664:其给 MLAAttentionLayer.build 加的 dcp_local_seq_lens 兜底是死代码,保留反而会掩盖 runner 侧未来的回归,删除以恢复单一数据源。

## 问题现象
正文未描述用户可见的故障现象。本 PR 是对 #53664 的回滚:#53664 曾为 Kimi-k3 DCP 模式在 `MLAAttentionLayer.build()` 里加了一个「`CommonAttentionMetadata` 不带 `dcp_local_seq_lens` 时从 `seq_lens` 现场推导」的兜底分支、一个断言和两条 test-registry 条目。回滚的动机是该兜底在所有 in-tree 路径上都不可达,且长期保留有害。

## 根因
正文披露的论证(即回滚依据):#53664 加的 fallback 是死代码——vLLM 的 in-tree 路径中,只要启用了 DCP,`dcp_local_seq_lens` 总是在 MLA decode 路径运行前就被填充:
- 单体 V1 model runner 在 `decode_context_parallel_size > 1` 时无条件填充 `CommonAttentionMetadata.dcp_local_seq_lens`(`vllm/v1/worker/gpu_model_runner.py:2456-2466`),且早于 #53664 编写时即已如此;
- 模块化 runner(MRV2)通过 `maybe_prepare_dcp_local_seq_lens()` 在构建 attention 元数据前、PCP batch 划分后同样填充(`vllm/v1/worker/gpu/model_runner.py:1728`,helper 在 `vllm/v1/worker/gpu/cp_utils.py`);该 helper 仅在 `dcp_size == 1` 时返回 None,此时 MLA 层的 `dcp_world_size` 也是 1,分支根本不会进入,故 fallback 不可达;
- 两个 runner 总会在 `CommonAttentionMetadata` 上设置 `seq_lens`,#53664 新加的断言同样不可达。
此外,保留兜底会掩盖 runner 侧填充逻辑未来的回归:一旦某路径停止填充 `dcp_local_seq_lens`,层会静默自行推导而不是响亮失败。

## 修复方案
回滚 #53664 的改动:
1. `vllm/model_executor/layers/attention/mla_attention.py` `build()`:删除 decode 分支里 `assert seq_lens is not None` 与「`dcp_local_seq_lens is None` 时调用 `get_dcp_local_seq_lens(...)` 现场推导」的兜底,恢复模型 runner 作为 DCP-local sequence lengths 的单一数据源。
2. `tests/models/test_registry.py`:删除 #53664 加的 `KimiLinearForCausalLM`、`KimiK3ForConditionalGeneration` 两条测试注册条目。

关键 diff 摘录(删除部分):
```diff
-                assert seq_lens is not None, (
-                    "MLA DCP decode requires seq_lens on CommonAttentionMetadata"
-                )
-                if dcp_local_seq_lens is None:
-                    dcp_local_seq_lens = get_dcp_local_seq_lens(
-                        seq_lens,
-                        dcp_size=self.dcp_world_size,
-                        dcp_rank=get_dcp_group().rank_in_group,
-                        cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
-                    )
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:恢复到 #53664 之前的行为。MLA decode 的 DCP 路径重新完全依赖 runner 侧填充 `dcp_local_seq_lens`;若未来某 runner 路径漏填,将直接报错(fail loudly)而非静默推导——这正是本 PR 想要的语义。对 DCP 未启用的常规 MLA 路径无影响。

## 关联
[#53664](https://github.com/vllm-project/vllm/pull/53664)(被回滚的 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
