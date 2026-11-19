# PR #67338: [Attention] Decide the sparse MLA prefill route once in the metadata builder

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Pb314314 |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel(tier: major) |
| Label | documentation, performance, rocm, intel-gpu, ready, ci/build, nvidia, verified, dflash |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数);diff 涉及 sparse_utils.py(枚举)、sparse_mla_attention.py、mla_attention.py、benchmarks/attention_benchmarks/*、多个 tests/v1/attention/* 文件,另有 8 个文件被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/67338 |

## 一句话结论

把稀疏 MLA prefill 的三条路由(dense MHA / masked MHA / sparse MQA)决策收拢到 metadata builder 的 `_prefill_route()` 一处,结果存入 `SparseMLAPrefillMetadata.route` 供全部 10 处读者消费;serving 行为不变,GLM-5.3-NVFP4 serving 两树均值差 ≤0.8% TTFT。

## 动机

此前路由决策分散在三处:builder 的 `use_dense_mha`、`MLAAttention._use_sparse_mha`(每层调用时算)、`SparseMLACommonImpl.forward_mha` 内的 `prefill_max_seq_len <= topk_tokens` 长度判断,另有七处各自重推;没有单一值说明当前步 prefill 实际走哪条路。三条路由在精确算术下结果相同(差别在 kernel、舍入与 sparse MQA 可用 fp8 query),带可学习 sink 的模型(如 HY-V4)强制 sparse MQA。

## 优化点

1. 新 `SparseMLAPrefillRoute` 枚举(置于 `sparse_utils.py`)与必填字段 `SparseMLAPrefillMetadata.route`(builder 忘设即在构造期失败,而非静默回退)。
2. `SparseMLACommonMetadataBuilder._prefill_route()` 按序应用 5 条规则(force_mqa / HiSparse / ShardQ → SPARSE_MQA;PCP+DCP>1 → SPARSE_MQA;`max_seq_len <= top-k` → DENSE_MHA;masked MHA 可用(全局 KV dtype 的 mask workspace + 该层组 impl 的 `masked_mha_available`,两者可因 MTP draft 独立 kv_cache_dtype 而不同)且 DCP=1、阈值内、mask 放得下 → MASKED_MHA;否则 SPARSE_MQA)。
3. 读取方改造:FlashInfer SM90 / ROCm planner 与 indexer 短 prefill skip、quant-output fusion check 均改为读 metadata 中的 route;ShardQ split 移到路由决策之前。
4. benchmark 统一用 `BenchmarkConfig.sparse_mla_route` 单一覆盖,替换 `sparse_mla_force_mqa` / `sparse_mla_mha_mode`;生产配置 `AttentionConfig.sparse_mla_force_mqa` 不变。

## 关键改动

```python
# tests/v1/attention/test_flashinfer_mla_sparse_sm90.py(消费方式示例)
-@pytest.mark.parametrize("use_mha", [False, True])
+@pytest.mark.parametrize("route", list(SparseMLAPrefillRoute))
 ...
-    builder._attention_layer = SimpleNamespace(_use_sparse_mha=lambda _: use_mha)
     metadata = object.__new__(sm90_mod.FlashInferMLASparseSM90Metadata)
+    metadata.prefill = SimpleNamespace(route=route)
```

```python
# sparse 路由枚举(由 PR 新增,SparseMLAPrefillRoute)
Route.DENSE_MHA / Route.MASKED_MHA / Route.SPARSE_MQA
```

## 性能声明

数字均照抄 PR 正文(未独立核实):

- No measurable performance change:on GLM-5.3-NVFP4 serving, the mean of two runs per tree differs by at most 0.8% (TTFT), less than the run-to-run spread. The kernels are unchanged; only the host-side route decision moved.
- Route trace(Baseline main `df5d80339b`/`b704146733` vs After PR,GB200,dummy weights,11 请求 300–20000 tokens,TP1/TP4):FlashMLA sparse bf16 KV:16 dense, 40 masked, 416 sparse MQA(逐路径逐形状一致,tokens 11/11 identical,max |logprob diff| 1.9e-6,A/A 2.9e-6);fp8 KV 与 FlashInfer sparse 各行 16 dense, 456 sparse MQA。MTP 行:9 calls per shape = 8 target layers + 1 MTP layer,1.9e-6 (A/A 2.9e-6)。GLM-5.3-Flash 行 2.0e-3(A/A 同为 2.0e-3,噪声来自其 linear-attention 层)。
- GB200 benchmark CLI:9 CLI cases / 18 result rows:15 successful measurements and 3 expected rejections。

## 验证与回归风险

- 测试:`-k 'route or masked_mha or sparse_backend_prefill_correctness or ...'` 50 passed;`test_flashinfer_mla_sparse_sm90` 等四文件 127 passed, 20 skipped;`test_mla_backends.py -k masked_mha` 32 passed;pre-commit 通过;新增 B200 ShardQ metadata CI step。
- Mutation check:从 builder 构造中移除 `and attention_layer.impl.masked_mha_available` 后,`-k masked_route_needs_layer_masked_mha` 如预期 FAILED(`MASKED_MHA is not SPARSE_MQA`)。
- Per-model check:GLM-5.x、DeepSeek-V3.2、GLM-5.3 MTP、GLM-5.3-Flash(含 MoE/bf16+fp8 KV)、HY-V4 均逐路径一致、token 全同;dots3-note 仅代码阅读+单测(其 sparse MQA kernel 是 FA3,Hopper only,未在 GB200 上端到端跑);Kimi DSA、Glm4MoeLite 代码阅读;dense MLA(DSV2/V3/R1、Kimi-K2.5/K3 等)与 DSV4/V4.1、MiniMax-M3、非 MLA 模型不受影响(DSV2-Lite trace bit-identical)。
- 回归风险/限制:CUTLASS_MLA 对比因既有 `'MockLayer' object has no attribute 'q_lora_rank'` 错误未能完成测量;dense FA2 未声明验证;本 revision(代码搬运)未复跑真权重评测;ROCm/XPU/Hopper 硬件在该 revision 不可用;route_diff 是 dummy-weight 路由检查,不构成严格输出相等证明。

## 关联

涉及 baseline commit `b704146733` / `df5d80339b` / `dfebb42d09` / `b3c08a2555`;相关模型族 hy_v4、dots3_note、kimi_k3(dsa_config);benchmark YAML 复现配置见 PR 正文(`/tmp/mla-prefill-check.yaml`)。索引内同区域:sparse MLA 路由/FlashMLA sparse 卡片(pack 未提供具体编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
