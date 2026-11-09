# PR #45635: [AuxOutput] Add block-keyed storage for routed-expert outputs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xhx1022 |
| 合并 | 2026-09-21 |
| 分类 | Speculative Decoding(tier: major) |
| Label | documentation, frontend, ready, ray, v1, kv-connector, mrv2, scheduler |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 23 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/45635 |

## 一句话结论

为 MoE routed-expert 输出(R3)引入 KV 兼容的 block-keyed 存储,使 R3 随前缀一起复用,不再依赖 request-local GPU buffer。

## 动机

R3(每个执行 token 被选中的 expert ID)在 prefix caching 与 async scheduling 开启时无法留在 request-local GPU buffer:prefix 命中会跳过模型执行、物理 GPU/KV slot 会被复用且不是稳定对象身份、speculative decoding 会执行后来被拒绝的行、一个结果可能同时含 cached block、新 block 与部分 tail。模型捕获 hook 只能拿到当前 forward 的 tensor,需要独立生命周期层决定哪些逻辑 token 被接受、哪些 block 可跨请求复用、何时释放临时 buffer 与存到何处。

## 优化点

引入首个 AuxOutput Connector。Scheduler 侧 connector 持有逻辑进度与 KV 兼容哈希(不是 payload),Worker 侧 connector 持有 accepted R3 行与发布,`BlockObjectStore` 管理不透明不可变 block 与保留策略。R3 block 用 KV 兼容的 block hash 内容寻址,同一前缀的 KV 与 R3 因此可以一起复用。本 PR 提供有界本地 SHM 后台发布后端;R3 GPU-to-CPU 传输移到既有 async output copy stream,避免模型执行关键路径上的同步拷贝。覆盖 async scheduling、chunked prefill、speculative rejection、preemption,streaming 输出仅 terminal 交付;禁用时正常服务路径不创建任何捕获 buffer、Store、每步 metadata 或 R3 拷贝。Mooncake 分布式后端与 DSA/logprobs 等其他辅助输出为 follow-up。

## 关键改动

```text
vllm/config/aux_output.py: 新增 AuxOutputConfig(enable_return_routed_experts、max_bytes LRU 容量),经 --aux-output-config 暴露;enable_return_routed_experts 从 ModelConfig 移出
vllm/config/vllm.py: 挂载 aux_output_config 并计入 compute_hash
vllm/distributed/aux_output_connector/connector.py: AuxOutputSchedulerConnector / AuxRequestOutput(Scheduler 侧逻辑进度与哈希)
vllm/distributed/aux_output_connector/worker.py: AuxOutputWorkerConnector(捕获绑定、accepted-row 组装、block 发布;内部 BlockObjectStore/BackgroundBlockObjectStore,对应 tests/distributed/aux_output_connector/test_store.py)
tests/model_executor/test_routed_experts_capture.py: 捕获输出 uint8/uint16 窄化 profile 与 DP/EP/TP 聚合
```

## 性能声明

无新功能加速比。rename 前 GB200 A/B(2026-09-08,两节点 8 GPU,本地缓存 DeepSeek-V4-Flash-Base FP8 权重,R3+MRV2+TP1/DP8/EP8+async scheduling+prefix caching;64 并发请求各 262,144 输入 token、1 输出 token):总吞吐 82,741.42 → 82,568.30 tokens/s(−0.21%,单次 A/B,非统计显著性);prefix 命中 0,preemption 0。验证:96 passed / 1 skipped。限制:长上下文 256K 全输出 bitwise 等价未通过(首个差异 row 2051,影响 55.38% 元素),PR 明确不能当作全 R3 correctness pass;cached-block R3 SHA 复用逐位一致。

## 验证与回归风险

单测覆盖 config/store/capture/Scheduler/output/Ray;开发验证含 async scheduling、forced preemption、chunked prefill、speculative rejection、cold/hot prefix reuse、streaming aggregation、SHM capacity/LRU、pre-commit。风险:`enable_return_routed_experts` 配置迁移(aux_output_config 计入 compilation hash);`reset_prefix_cache` 新增 in-flight 输出与 pause(mode='keep') 校验(行为变化);R3 存储容量默认从 KV cache 容量推导;长上下文 R3 bitwise 差异原因未定位。

## 关联

同分类 Speculative Decoding:#43091、#50272、#38174、#20401;同批 KV/内存区域:#44890(reset_prefix_cache/sleep 语义)、#52921。既有 `--enable-return-routed-experts` 标志与 `routed_experts` 响应字段保持不变;后续:统一 object/profile 契约、Mooncake 后端、KV+辅助输出联合前缀就绪。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
