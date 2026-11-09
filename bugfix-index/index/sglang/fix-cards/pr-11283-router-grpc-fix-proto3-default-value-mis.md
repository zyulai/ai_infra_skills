# PR #11283: [router][grpc] Fix proto3 default value mismatches and cleanup unused fields

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#11283](https://github.com/sgl-project/sglang/pull/11283) |
| 作者 | CatherineSue |
| 合并日期 | 2025-10-07 |
| labels | router, run-ci |
| 热度 | 👍 0 · 💬 1 |
| 分类 | Frontend / Serving |
| bug 类型 | correctness |

## 一句话结论

proto3 数值默认值(0)与 Python `SamplingParams` 语义默认值(temperature=1.0 等)不一致,Python 侧 `or` 兜底又破坏 temperature=0.0 的贪心采样;改为 Rust 显式构造默认值、Python 直接透传。

## 问题现象

走 router gRPC 路径的请求,若 proto 字段被直接使用,会因 proto3 默认值与语义默认值不符而行为错误。正文明确指出的一个具体错误:Python 服务端用 `temperature=grpc_params.temperature or 1.0` 兜底,当客户端显式传 `temperature=0.0` 做贪心采样时,该值被 `or` 短路成 1.0,贪心采样失效。同类问题覆盖 `top_p`、`top_k`、`repetition_penalty` 等所有数值字段;`max_new_tokens or 128` 也会把缺省语义与显式值混在一起。此外 `lora_path`、`token_healing` 是 proto 中从未使用的字段。

## 根因

正文披露:proto3 数值字段的默认值(0)与 Python `SamplingParams` 的语义默认值(temperature=1.0、top_p=1.0、top_k=-1、repetition_penalty=1.0 等)不匹配;Python 服务端又用 `or` 兜底来"修复"缺省值,导致合法的 0 值被覆盖。设计上选定单一事实来源(single source of truth):由 Rust 构造 proto 消息时写入正确默认值,Python 信任并直接使用,不再做防御性兜底。

## 修复方案

- `python/sglang/srt/grpc/sglang_scheduler.proto` 与 `sgl-router/src/proto/sglang_scheduler.proto`:语义上可选的 `max_new_tokens`、`stream_interval` 改为 `optional`;删除未使用的 `lora_path`、`token_healing` 字段,后续字段编号顺延重排;加注释警告不要直接用 `::default()`;
- Rust 客户端 `sglang_scheduler.rs`:构造请求时显式设置全部字段默认值(`temperature: 1.0`、`top_p: 1.0`、`top_k: -1`、`repetition_penalty: 1.0`、`skip_special_tokens: true`、`spaces_between_special_tokens: true`、`n: 1`);
- Python `grpc_server.py` `_convert_sampling_params`:删除所有 `or` 兜底,直接透传;补上缺失的 `logit_bias`、`custom_params`、`stream_interval`、`no_stop_trim` 转换;空 repeated 字段转 `None` 而非 `[]`(如 `stop=list(grpc_params.stop) if grpc_params.stop else None`)。

```diff
-            temperature=grpc_params.temperature or 1.0,
-            top_p=grpc_params.top_p or 1.0,
-            top_k=grpc_params.top_k or -1,
+            temperature=grpc_params.temperature,
+            top_p=grpc_params.top_p,
+            top_k=grpc_params.top_k,
```

`sglang_scheduler_pb2.py`/`.pyi` 同步再生成。

## 影响与回归风险

- 影响窗口:正文未说明。
- 回归面:router gRPC 的整条采样参数链路——proto 定义(字段编号重排,新旧协议两端不再字段对齐)、Rust 客户端默认值构造、Python 服务端转换。Python 侧不再兜底,若上游(旧版 router)未按新约定填默认值,temperature 等字段将直接以 proto3 的 0 进入采样;`max_new_tokens` 缺省时不再回退 128 而是传 `None`。已加单元测试断言 proto3 默认值与语义默认值的差异(Rust 侧 `test_sampling_params_defaults`)。

## 关联

正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
