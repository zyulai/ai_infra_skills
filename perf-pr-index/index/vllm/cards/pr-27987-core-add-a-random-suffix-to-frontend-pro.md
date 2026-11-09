# PR #27987: [Core] Add a random suffix to frontend-provided request IDs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @markmc |
| 合并 | 2025-12-23 |
| 分类 | Frontend / Serving(tier: major) |
| Label | frontend, ready, v1, gpt-oss, kv-connector, ready-run-all-tests |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files;diff 本身可用,但有 3 个文件因超长被截断未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/27987 |

## 一句话结论

解决前端自定义 request ID 在并发下碰撞导致的请求串扰/调度异常问题,保障服务正确性而非直接提速。

## 动机

自 #9550 与 #10968 起,vLLM 支持客户端自定义 request ID,便于将 vLLM 日志与上游服务日志关联。但 request ID 在 vLLM 内部被当作全局唯一键使用,一旦多个在途请求复用同一 ID 就会出问题:近期 `vllm serve bench` 携带固定 request ID、多实例并发时即发生了碰撞(#27723)。现有防线仅是 OutputProcessor 里 `request_id in self.request_states` 时抛 ValueError,在三类场景下失效:abort 竞态(#15326)、异步调度竞态(#29355)、P/D 分离下 prefill 引擎长时间跟踪已完成请求(#20139)。

## 优化点

机制上把"外部 ID"与"内部 ID"分离:在创建 `EngineCoreRequest` 前,`InputProcessor.assign_request_id()` 给前端提供的 ID 追加 8 个随机十六进制字符(32 bit 熵)。按碰撞概率近似 (k^2)/(2N),即使 1 万个请求复用同一客户端 ID,碰撞概率也仅约 1.16%,对防碰撞"足够好"。内部全链路使用带后缀的 internal ID;对外输出(如 `RequestOutput.request_id`)仍返回原始 external ID,保持向后兼容。abort 路径由 `OutputProcessor` 维护 external→internal 映射,支持按任一 ID 取消;`AsyncLLM` 用携带 internal ID 的 `RequestOutputCollector` 在异常/取消时精确 abort;parallel sampling 的 `ParentRequest` 同时跟踪两种 ID 以聚合子请求输出。

## 关键改动

```text
vllm/v1/engine/input_processor.py: 新增 assign_request_id(),生成 internal ID 并落 external_req_id
vllm/v1/engine/__init__.py: EngineCoreRequest 增加 external_req_id 字段
vllm/v1/engine/output_processor.py: RequestState/RequestOutputCollector 携带双 ID;abort_requests 支持 internal 标志
vllm/v1/engine/async_llm.py: abort(request_id, internal=False);generate/encode 异常路径改按 internal ID abort
vllm/v1/engine/llm_engine.py: add_request/abort_request 同步适配双 ID 语义
```

核心代码摘录:

```python
@staticmethod
def assign_request_id(request: EngineCoreRequest):
    """Replace the externally supplied request ID with an internal request ID
    that adds 8 random characters in order to ensure uniquness."""
    if request.external_req_id is not None:
        raise ValueError(
            "The external_req_id field should not be set on EngineCoreRequests"
            " passed to vLLM; use the request_id field.")
    request.external_req_id = request.request_id
    request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

## 性能声明

PR 正文与素材中未检测到任何 benchmark 数字(regex 检测为 none)。本 PR 为正确性/健壮性修复,不声明性能收益;碰撞概率 1.16%(10k 同名请求场景)为理论计算而非压测结果。PR 未注明测试条件。

## 验证与回归风险

PR 大幅扩展了 `tests/v1/engine/` 下测试:`test_output_processor.py`(abort 按 internal/external 双参数化)、`test_async_llm.py`、`test_parallel_sampling.py`、`test_engine_core.py`/`test_engine_core_client.py`(构造函数补 external_req_id)、`test_process_multi_modal_uuids.py`(mm UUID 会继承随机后缀,断言改为前缀/后缀匹配)、`test_fast_incdec_prefix_err.py`。回归风险集中在 ID 语义变化面:依赖 request ID 格式的下游(日志关联、metrics 按 ID 聚合、P/D 连接器、多模态 mm UUID 派生)都会看到带后缀的内部 ID;`AsyncLLM.add_request`/`LLM.add_request` 传入的 request_id 参数与 `EngineCoreRequest.request_id` 不一致时会被忽略并告警。标签含 kv-connector、gpt-oss,提示需跑相关集成测试。

## 关联

前序: #9550、#10968(自定义 request ID 支持);#15326、#29355、#20139(此前的三类失效场景修复尝试);#27723(触发本 PR 的碰撞案例)。本卡批次内无同分类 PR。
