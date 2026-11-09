# PR #44890: [Frontend][Core] Add release_kv_cache_memory() API

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @andakai |
| 合并 | 2026-09-17 |
| 分类 | Frontend / Serving(tier: major) |
| Label | documentation, frontend, ready, v1, multi-modality, cpu, rust, scheduler |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;diff 中 2 个文件因超长被省略) |
| 链接 | https://github.com/vllm-project/vllm/pull/44890 |

## 一句话结论

新增 `release_kv_cache_memory()`:只丢弃 KV cache 物理显存、保留模型权重,降低 RL 权重更新等只需临时 GPU 余量场景的峰值显存。

## 动机

sleep level 2 释放 KV cache 时连模型权重一起丢弃,对只需临时 GPU 余量、保持同一模型实例的场景过于昂贵。典型用例是 RL 训练中的模型权重更新:verl#7373 只能用 `sleep(level=2)` + `wake_up(tags=["weights"])` 的 workaround。只释放 KV cache 可降低峰值显存、避免更新期间 OOM,且无需卸载并重映射权重。

## 优化点

KV-only 部分休眠:offline 与 server 入口(LLM API 与 `POST /release_kv_cache_memory`)汇聚到 EngineCore——先校验"已完成 pause 且无 pending 请求/batch、executor 内存全部 resident",否则 fail closed(不清 cache、不 discard);随后清理 prefix cache 并请求所有 worker `discard(("kv_cache",))`。分配器层只 unmap/release 标记 `kv_cache` 的 mapped allocation(无 CPU 备份),per-allocation mapping 状态防止与 sleep/wake_up 混用时重复 unmap/map;后续 `sleep(1)/sleep(2)` 可把状态升级为完全休眠,`wake_up(tags=["kv_cache"])` 恢复并重启调度;executor 以 `sleeping_tags` 跟踪精确状态(`is_sleeping` 变为属性),`wake_up` 返回是否完全恢复;release 同时异步清空 MM processor cache,避免 sender 缓存指向已清空的 receiver。

## 关键改动

```text
vllm/v1/engine/core.py: release_kv_cache_memory 状态校验与 discard 调用;wake_up 返回 fully_awake
vllm/v1/executor/abstract.py: SLEEP_TAGS/sleeping_tags 与 discard(tags) collective_rpc
vllm/device_allocator/sleep_mode_backend.py: discard(tags) 选择性丢弃(默认后端不支持则报错)
vllm/entrypoints/llm.py、vllm/entrypoints/serve/dev/sleep/api_router.py: Python API 与 HTTP 端点
rust engine-core-client/server: release_kv_cache_memory 路由;wake_up 返回 bool 并支持重复 tags
```

## 性能声明

PR 未提供性能数据("no additional model benchmark was run beyond the generation-equivalence checks")。功能等价验证(H200 容器):Qwen3-0.6B 贪心生成 token ID 在"拒绝的 release"与"KV-only release/wake、release 后升级 level 1/2 sleep 再恢复"后保持一致;KV-only release 后 mapped 内存减少量 ≥ kv_cache_memory_bytes 的 99%(测试断言)。

## 验证与回归风险

tests/v1/engine/test_engine_core.py 拒绝场景 5 passed(未暂停、pending requests、pending batches);tests/basic_correctness/test_mem.py 2 passed;tests/multimodal/test_cache.py 新增 release 后 MM payload 重发用例;rust 路由测试。风险:CUDA/ROCm 需 cumem 后端(`enable_sleep_mode=True`),XPU 需 sleep mode,CPU 不支持选择性 discard;旧 release 后调度保持暂停直至 wake_up;release 后指标 awake=0/weights_offloaded=0/discard_all=0(无专用 Prometheus 状态)。

## 关联

同分类 Frontend / Serving:#51360。相关:#42658(通用 tag 级 sleep)、#46438(分配器选择性丢弃)、verl#7373(下游用例);同批 #45635 触及 reset_prefix_cache/aux output 与 pause 语义的交互测试。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
