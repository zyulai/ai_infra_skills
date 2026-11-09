# PR #46725: Runtime Draft Weight Update for Speculative Decoding

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @vx120(co-authored by @crp0128) |
| 合并 | 2026-07-12 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, speculative-decoding, new-model, ready, v1, frontend, documentation, nvidia, rocm, intel-gpu, cpu, qwen, deepseek, llama, mistral, gpt-oss, kv-connector, multi-modality, tool-calling, ci/build, rust |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/46725 |

## 一句话结论

给 RL 训练回路补上 draft 模型的运行时权重更新通道,使 spec decode drafter 能随 verifier 一起刷新,保住 acceptance length 与吞吐收益。

## 动机

vLLM 为 RL 训练暴露三条权重更新路径(checkpoint/layerwise、direct reload、kernel-format),全部只作用于 `self.model_runner.model`(verifier),没有任何路径触碰 speculative draft 模型。在 RL 训练循环里会出三个问题:(1) trainer 产出的权重迭代器只覆盖 verifier 参数名,draft 自有参数(FC 层、Transformer 层、norm)加载一次后永不更新;(2) `finalize_layerwise_reload` 可能重建参数对象,悄悄切断 draft `embed_tokens`/`lm_head` 与 verifier 的 Python 引用共享,draft 开始读 stale embeddings;(3) `sleep(level=2)` 会丢弃 untagged pool 的全部 cumem 分配,draft 参数恰在其中,`wake_up` 只恢复 verifier 的 named buffers,draft 模块持有失效 GPU 内存。SGLang 也有同样缺口(sglang#27718,修复中 sglang#28257),vLLM 此前无任何路径的等价处理。

## 优化点

五个互补改动:(1) `Worker.get_draft_model()` 作为 draft `nn.Module` 的稳定访问器,分别处理 GPUModelRunner 的 `drafter`(自动 unwrap `CUDAGraphWrapper`/`UBatchWrapper`/`BreakableCUDAGraphWrapper`)与 speculator 形态;(2) 新 API `start_draft_weight_update()`——引擎协议、LLM、AsyncLLM、RLHF API router 全链路打通,worker 侧通过 `WeightTransferEngine.set_weight_update_target()` 把传输引擎本次 session 重定向到 draft 模型与 `speculative_config.draft_model_config`,`finish_weight_update`/异常时 `reset_weight_update_target()` 复位,`update_weights` 每个分块加载进 session 选定的目标;`SparseNCCLWeightTransferEngine` 声明 `supports_draft_weight_update = False` 明确不支持;(3) `sleep(level=2)`/`wake_up` 间保留 draft 参数——sleep 前记录 draft 是否需要 `_build_fused_kv_buffers`,wake_up 后重建该元数据 buffer;(4) `EAGLEConfig._normalize_dflash_layer_ids()` 移除 DFlash 配置隐患;(5) spec decode acceptance stats 始终以 INFO 级别输出,便于 RL 回路监控接受率。

## 关键改动

```text
vllm/v1/worker/gpu_worker.py: start_draft_weight_update + set/reset_weight_update_target,sleep/wake_up 的 draft buffer 重建
vllm/distributed/weight_transfer/base.py: WeightTransferEngine 增加 set_weight_update_target/reset 与 supports_draft_weight_update
vllm/engine/protocol.py + entrypoints/llm.py + v1/engine/async_llm.py + entrypoints/serve/dev/rlhf/api_router.py: 新增 /start_draft_weight_update 全链路
vllm/v1/worker/gpu_model_runner.py + vllm/v1/worker/gpu/model_runner.py: get_draft_model() 访问器(处理 CUDAGraph 等 wrapper unwrap)
```

关键摘录(session 级目标重定向):

```python
def start_draft_weight_update(self) -> None:
    """Like start_weight_update, but retargets the engine at the
    speculative draft model for this session."""
    self._start_weight_update(is_draft=True)
...
if is_draft:
    self._set_draft_weight_update_target()
self.weight_transfer_engine.start_weight_update()
```

## 性能声明

无直接吞吐数字。PR 正文给出 RL 实验曲线截图:Qwen3-4B + DFlash drafter(batch size 16)的 reward 与 acceptance length 曲线;Qwen3-8B + DFlash drafter(batch size 16)且训练 drafter 模型后 "acceptance length longer"(具体数值在截图内,正文无表格数字);未注明硬件与精度。

## 验证与回归风险

测试:`tests/v1/worker/test_gpu_worker_weight_transfer.py` 扩展 `_RecordingEngine` 断言 `reset_weight_update_target` 恰被调用一次(finish 与出错路径各一);`tests/entrypoints/openai/test_openai_schema.py` 把 `/start_draft_weight_update` 加入 stateless 路径白名单;docs/training/weight_transfer/base.md 更新用法说明。回归风险:权重更新 session 的目标状态机(start→update→finish/异常复位)影响所有 weight transfer 路径;`sleep(level=2)`/`wake_up` 行为变化影响带 drafter 的 RL/offload 场景;`start_draft_weight_update` 在无 draft 模型或无 draft_model_config 时抛 RuntimeError;SparseNCCL 引擎显式不支持并报错。

## 关联

同区域:spec decode drafter(Eagle/Eagle3/DFlash)权重与 KV 元数据管理;SGLang 对应问题 sglang#27718 / sglang#28257。本卡所在批次素材中未列出 vLLM 侧其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
