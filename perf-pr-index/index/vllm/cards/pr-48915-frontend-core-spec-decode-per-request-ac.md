# PR #48915: [Frontend][Core][Spec Decode] Per-request acceptance stats in OpenAI API responses

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @matthewkotila(co-author @reed-meyerson) |
| 合并 | 2026-08-20 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, frontend, speculative-decoding, ready, v1, verified, rust, scheduler |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files;diff 摘录另有 9 个文件因超长被丢弃) |
| 链接 | https://github.com/vllm-project/vllm/pull/48915 |

## 一句话结论

在 OpenAI API 响应的 `metrics.speculative_decoding` 中按请求报告 spec decode 接受率与接受长度直方图,默认关闭,行为零改动(观测性)。

## 动机

spec decode 的接受情况此前只能在 `/metrics` 看到服务器级聚合均值,无法得知单个请求的表现;AIPerf 等工具需要 per-request 数字,目前只能包一层 engine 才能拿到。把该数字放进响应体,客户端即可逐请求计算 mean acceptance length 与接受分布,作为 Prometheus 聚合计数的个体侧补充。此 PR 延续 #47322 的设计(原作者署名 co-author),收敛到 #46768 引入的 per-request timing `metrics` 对象上。

## 优化点

纯观测性改动,不碰采样/调度/生成 token。数据链路:scheduler 在 `add_request` 时为每个请求预建 `RequestSpecDecodeMetrics` 累加器(默认 `none` 时完全不创建);`update_from_output` 里每个 verify step 调 `observe(num_draft_tokens, num_accepted, detailed)`,其中 proposed 计数先扣除 grammar 失效的 draft(`num_invalid_spec_tokens`,与聚合统计口径一致),`detailed` 级别才记录 per-step 数组;只在请求结束的 final `EngineCoreOutput` 上附带(字段追加在末尾,保持 array_like 位置序列化兼容,Rust 侧 `EngineCoreOutput` 加 `Option<OpaqueValue>` 占位但前端暂不展示)。前端把类型化模型 `SpeculativeDecodingMetrics`(mean_acceptance_length = 1 + num_accepted_draft_tokens/num_spec_steps、draft_acceptance_rate、长度为 k+1 的 `acceptance_histogram`——index j 为恰好接受 j 个 draft 的 verify 步数、不含必中的 bonus token、num_spec_steps、num_accepted/num_draft_tokens、num_spec_tokens)挂进 `PerRequestMetrics`(由 `PerRequestTimingMetrics` 改名扩展)。配置:`--per-request-spec-decode-metrics none|summary|detailed`,默认 none 且响应不变;n>1 抑制(与 timing 一致);streaming 时 rides final usage chunk(需 `include_usage`);config 校验强制要求已启用 speculative decoding。

## 关键改动

```text
vllm/v1/core/sched/scheduler.py: add_request 预建累加器,update_from_output 观察(扣 grammar 失效 draft),final output 附带
vllm/entrypoints/openai/engine/protocol.py: SpeculativeDecodingMetrics 类型 + PerRequestMetrics(原 PerRequestTimingMetrics)扩展
vllm/entrypoints/generate/base/serving.py + openai/{chat_completion,completion}/serving.py: build_spec_decoding_metrics 接入非流式与流式
vllm/config/observability.py + engine/arg_utils.py: --per-request-spec-decode-metrics 配置与校验
docs/features/speculative_decoding/acceptance_metrics.md: 新文档(字段表、与 Prometheus 计数的对账关系)
```

关键摘录(响应示例):

```json
{"metrics": {"speculative_decoding": {
  "mean_acceptance_length": 1.2325581395348837,
  "draft_acceptance_rate": 0.07751937984496124,
  "acceptance_histogram": [39, 1, 0, 3],
  "num_spec_steps": 43, "num_accepted_draft_tokens": 10,
  "num_draft_tokens": 129, "num_spec_tokens": 3}}}
```

## 性能声明

无性能数字;正文声明 "Purely observational — no change to sampling, scheduling, or generated tokens。Output is byte-identical across none/summary/detailed。No accuracy eval applicable。" E2E(H100 80GB,Qwen2.5-0.5B-Instruct,n-gram spec decode,num_speculative_tokens=3)验证 `metrics.speculative_decoding` 符合 schema:单序列请求的计数求和与聚合 `vllm:spec_decode_*` Prometheus 计数完全对账(仅对全 n==1 workload;n>1 请求计入聚合但不按请求上报)。

## 验证与回归风险

测试:`tests/v1/spec_decode/test_request_acceptance.py`;`tests/v1/core/test_scheduler.py -k spec_decod`(test_schedule_spec_decoding_stats 扩展 per-request 直方图断言,新增 detailed per-step、grammar 失效扣除、默认关闭三个用例);`tests/entrypoints/openai/completion/test_completion_error.py -k spec_decode`;`tests/test_config.py -k per_request_spec_decode_metrics`。回归风险:`PerRequestTimingMetrics` 改名 `PerRequestMetrics` 影响响应 schema(字段兼容,新增可选 `speculative_decoding`);`EngineCoreOutput` 追加新字段需保持位置序列化兼容(Rust fixture 同步更新);累加器在 `none` 级别不创建,无默认开销;与 #46768 的 timing metrics 共存但开关独立。

## 关联

延续 #47322;收敛于 #46768 的 per-request metrics 对象;相邻但不同:#43310(native /inference/v1/generate 端点、RL rollout 用)、RFC #48202(自适应 drafting 行为改动,`per_step_drafted` 已兼容)、#44487(Prometheus 侧直方图)。与 PR #46725(INFO 级 acceptance stats)同属 spec decode 可观测性。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
