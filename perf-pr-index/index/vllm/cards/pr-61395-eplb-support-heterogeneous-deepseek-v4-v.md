# PR #61395: [EPLB] Support heterogeneous DeepSeek V4/V4.1 DSpark draft experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2026-11-13 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, speculative-decoding, qwen, deepseek, mrv2, DSv4, dflash, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/61395 |

## 一句话结论

当 DeepSeek V4/V4.1 DSpark draft 的 expert topology 与 target 不同时也启用 EPLB:draft MoE 注册进 target 的 EPLB state、合并负载统计;支持 `draft_num_redundant_experts` 配置与"target 开 EPLB 即默认 draft 开"。

## 动机

#57549 的 follow-up:target 与 DSpark draft 可有不同的 routed/physical expert 数,而 EPLB 状态此前假设同构拓扑。本 PR 覆盖 DSpark 的 draft model loader。Related:#59611 修复 target 模型的 redundant expert 加载;open 的 #66045 处理 weight loading 时的 draft expert 过滤,与本 PR 互补但未实现 heterogeneous EPLB state 或负载均衡。

## 优化点

1. 支持的 DSpark draft MoE 模型注册进 target 的 EPLB state,合并(一起 reduce)双方负载统计,physical expert 数不同也支持;EPLB state 按 registered model 保存 topology 相关计数并处理不同 physical expert 维度的统计规约。
2. Draft R(r redundant experts)解析三情形:未设置且 target/draft routed-count 集合一致 → 继承已解析且对 draft 有效的 target R;集合不同 → 计算 draft 最小值;显式 draft R 对每个 draft routed count 满足整除则采用,否则 config 校验拒绝并在存在公共可行值时建议最小值;draft layers 需要不同 padding 时报告"无单一 R 可行";target R 为自动且校验时尚未解析时,同构拓扑各自独立算出同一最小值。
3. `--eplb-config` 新增 `draft_num_redundant_experts`(仅 draft override,不改 target 的 `num_redundant_experts`;无 speculative MoE draft 时被忽略);`draft_enable_eplb` 留在 `--speculative-config`(默认 true)。
4. target EPLB 开启时,supported V4.1 DSpark drafts 默认启用 draft EPLB;`draft_tensor_parallel_size=1` 的 DSpark draft 要求 prefill context parallelism 或 data parallelism > 1,否则校验拒绝;DSpark draft EPLB 与 Elastic EP 不兼容,校验拒绝并建议 `draft_enable_eplb=false`。
5. V4.1 DSpark 的 fused-MoE weight mapping 纳入 redundant physical slots(DSpark 自定义 weight loader 与 target 的 `get_expert_mapping` 是两条调用路径,注册 EPLB 后 redundant slots 也必须收到复制的 expert 权重;配合 #59611 才能端到端使用 redundant experts)。
6. target EPLB 校验:未设置的 target R 现在本地解析做整除检查而不改动 `EPLBConfig`,让 ngram 配置能走正常 `VllmConfig` 解析路径。

## 关键改动

```text
vllm/config/speculative.py: draft_enable_eplb / resolved_draft_num_redundant_experts + _verify_draft_eplb_config + apply_draft_eplb_overrides
vllm/config/parallel.py: EPLBConfig.draft_num_redundant_experts + get_min_num_redundant_experts
vllm/config/model.py: 通用 verify_eplb_config / get_eplb_expert_counts(整除校验)
vllm/distributed/eplb/eplb_state.py: 移除同构假设的 validate_ep_configuration,支持异构注册
docs/features/speculative_decoding/README.md: draft_enable_eplb 与 draft_num_redundant_experts 说明
tests/...(test_deepseek_v4_mega_moe / test_dspark_utils / test_gpu_model_runner_v2_eplb 等): V4/V4.1 fixture 化与 EPLB 注册测试
```

(注:pack 的 diff 截断,有 15 个文件未展示,以上仅覆盖已展示部分。)

```python
            minimum_counts = {(-count) % ep_size for count in draft_expert_counts}
            if len(minimum_counts) == 1:
                self.resolved_draft_num_redundant_experts = next(iter(minimum_counts))
            else:
                self.resolved_draft_num_redundant_experts = (
                    -default_num_experts
                ) % ep_size
```

## 性能声明

(perf claim 正则未检测到;以下为正文的 GSM8K 功能性验证,非吞吐 benchmark)

- Server:`vllm serve deepseek-ai/DeepSeek-V4.1-Flash --tensor-parallel-size 4 --enable-expert-parallel --enable-eplb --eplb-config '{"num_redundant_experts":4,"step_interval":100,"log_balancedness":true}' --attention-backend FLASHINFER_MLA_SPARSE_DSV41 --sparse-indexer-topk-backend flashinfer --kv-cache-dtype fp8 --speculative-config '{"method":"dspark","num_speculative_tokens":5}'`(另 `--max-model-len 8192 --gpu-memory-utilization 0.9`)。
- Client:`tests/evals/gsm8k/gsm8k_eval.py --num-questions 1319 --num-shots 5 --max-tokens 256 --temperature 0.0 --seed 42`。
- 结果:accuracy **0.902**(1,319 questions,5-shot);invalid responses 0。
- 该 run 的启动/重排日志(target R=4,自动选定 draft R=0,routed_expert_counts=(128,), ep_size=4):"EPLB is enabled for MoE part of drafter model deepseek-ai/DeepSeek-V4.1-Flash" 与 target 先后生效;profile 重排 1.76 s / 1.81 s;EPLB step 98/99:draft balancedness 0.6526→0.6653(avg_tokens 4500),target balancedness 0.7429→0.7504(avg_tokens 235080→512400)。target 与 draft 都参与 EPLB rearrangement。

## 验证与回归风险

- 回归风险(正文/测试体现):DSV4/DSV41 EPLB 在 ROCm 上禁用(相关测试 ROCm skip,`speculator.eplb_state is None`);V4.1 MegaMoE 测试要求 SM100;端到端使用 redundant experts 依赖 #59611 的 target 侧修复;`draft_num_redundant_experts` 只在 `draft_enable_eplb=true` 且 target EPLB 开启且 MoE draft 时可用,组合不当会 config 校验拒绝(附补救建议)。
- 正文自述的 PR checklist 多项未勾选(Design Fit / Testing and Validation 等),GSM8K 为唯一端到端验证手段。

## 关联

- #57549(本 PR 的 follow-up 对象)、#59611(target redundant expert 加载修复,端到端前置)、#66045(open,draft expert filtering,互补未重叠)。
- 文档引用:expert_parallel_deployment.md 的 "multimodel EPLB with speculative decoding" 小节。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
