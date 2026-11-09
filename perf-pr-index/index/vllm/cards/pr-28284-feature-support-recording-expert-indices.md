# PR #28284: [Feature] Support recording expert indices for rollout router replay

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xhx1022 |
| 合并 | 2026-01-12 |
| 分类 | MoE (tier: major) |
| Label | performance, frontend, ready, v1, rl |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 10 个文件(config、fused_moe capturer、model runner、scheduler、输出管线) |
| 链接 | https://github.com/vllm-project/vllm/pull/28284 |

## 一句话结论

引入 Rollout Router Replay(R3)支持:推理时按 token/layer 记录 MoE 路由专家索引并随请求返回,供 RL 后训练重放路由。

## 动机

MoE LLM 的 RL 对齐研究(arXiv:2510.11370、arXiv:2507.18071)需要在 rollout 阶段拿到每个 token 在每一层的专家路由决策,以便后训练时重放(replay)路由过程;vLLM 此前不暴露这些内部量。实现参考 SGLang PR #12162(致谢 @ocss884 与 SGLang RL 团队)。

## 优化点

一条 opt-in 的端到端路由追踪管线。入口:`ModelConfig.enable_return_routed_experts`,经 `EngineArgs`/CLI(`--enable-return-routed-experts`)、`LLM` 入口与 config 日志打通。捕获:新增 `fused_moe/routed_experts_capturer.py`,`RoutedExpertsCapturer` 持有 `(max_num_batched_tokens, num_layers, num_experts_per_tok)` 的 int32 device buffer 与共享内存 buffer(`multiprocessing.shared_memory` + 文件锁,尺寸按 `max_num_kv_tokens` 预留),`FusedMoE` 层在 select_experts 返回前按 `layer_id` 调 `capture(topk_ids)`(dummy run 时 capturer 未初始化则跳过)。落盘:TP rank 0 在 `execute_model` 末尾 `save_captured_experts(slot_mapping)` 把 device buffer 按 slot 写入共享内存;`GPUModelRunner` 负责 init/clear/save 并留存 `slot_mapping`。回读:Scheduler 侧 `RoutedExpertsReader` attach 同一共享内存,请求结束时从 KV blocks 重建 token 的 slot 映射(`block_id * block_size + offset`),读出 `[seq_len, layer_num, topk]` numpy 数组,经 `EngineCoreOutput.routed_experts` → `CompletionOutput.routed_experts` 传给客户端。约束:开启时禁用 async scheduling,断言不支持 context parallelism(DCP/PCP > 1);支持 TP、Prefix Cache 与 CUDA Graph 执行。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/routed_experts_capturer.py: 新增 Capturer/Reader 单例与共享内存缓冲
vllm/model_executor/layers/fused_moe/layer.py: select_experts 中按 layer_id 捕获 gate topk_ids
vllm/v1/worker/gpu_model_runner.py: init/clear/save 捕获、slot_mapping 计算、TP rank 0 写共享内存
vllm/v1/core/sched/scheduler.py: 请求结束时从 KV blocks 重建 slot 并读出 routed_experts
vllm/outputs.py、vllm/v1/engine/*: CompletionOutput/EngineCoreOutput 增加 routed_experts 字段
vllm/config/model.py、vllm/engine/arg_utils.py: enable_return_routed_experts 开关
```

捕获钩子(节选):

```python
capturer = RoutedExpertsCapturer.get_instance()
if capturer is not None:  # in dummmy_run may be None
    capturer.capture(
        layer_id=self.layer_id,
        topk_ids=topk_ids,
    )
```

## 性能声明

PR 正文无 benchmark 数字(素材正则未检出性能声明;本 PR 为功能开关,路由记录本身有额外开销但未量化)。给出的是功能验证:Qwen3-30B-A3B、`tensor_parallel_size=8`、async 并发推理(10 请求)下 `result.outputs[0].routed_experts.shape` 为 `[seq_len, layer_num, topk]`。正文注明:输出专家数可能比 `prompt_length + response_token_count` 少 1,因为最后一个 token 是采样所得、未经 forward 层计算。

## 验证与回归风险

素材 diff 中未见新增测试文件;正文给出最小可复现脚本(TP=8 + `enable_return_routed_experts=True`)。回归风险:关闭(默认 False)时无行为变化;开启时每层 FusedMoE 多一次 device buffer 写、每个请求结束多一次共享内存读(经文件锁串行化),TP rank 0 独占写;共享内存按 `max_num_kv_tokens x layers x topk` int32 预留,大模型/长上下文下可观;不支持 DCP/PCP,开启时强制禁用 async scheduling;dummy run(prefill profiling)路径 capturer 可能为 None,依赖判空。路由重放的正确性依赖 slot_mapping 与 KV block 布局的一致性(prefix cache 命中时 token 槽位不变是其成立前提)。

## 关联

参考 SGLang PR #12162(RoutedExpertsCapturer 移植来源);同区域后续性能优化:PR #62392(routed-experts store `BlockObjectStore` 的驱逐重扫修复,服务于同一条 aux/routed experts 输出通道)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
