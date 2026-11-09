# PR #12162: [Feature] Enable return routed experts

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ocss884 |
| 合并 | 2025-12-21 |
| 分类 | MoE (tier: major) |
| Label | documentation, high priority, quant, amd, dependencies, Multi-modal, deepseek, sgl-kernel, blackwell, npu, run-ci, piecewise-cuda-graph, diffusion, format, model-gateway |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/12162 |

## 一句话结论

推理时可返回每层 MoE 路由专家 id(topk),供 RL 训练侧使用,消除 train/inference 路由分歧带来的 logprob 差。

## 动机

RL 社区需要 SGLang 在 forward 过程中返回 MoE 路由的专家(topk id)供训练阶段使用。MiMo 团队提出的 R3 方法(arXiv 2510.11370)用于稳定 MoE 强化学习,已被 MiMo-V2-Flash 与 DeepSeek-V3.2 采用;社区同时观察到该方法能有效降低训练与推理输出之间的 logprob 差异——即推理引擎侧的路由信息缺失曾是 RL 训练回路中的一致性瓶颈。这是一个能力型 PR,为 RL rolllout 数据通路补齐路由捕获,而非直接的推理加速。

## 优化点

核心是全局 `RoutedExpertsCapturer` 三层管道。捕获层:`TopK` 增加可选 `layer_id`(约 12 个 MoE 模型文件逐一传入),`select_experts` 在返回前调用 `get_global_experts_capturer().capture(layer_id, topk_ids)` 写入 device 端 int32 捕获 buffer(形状 `[max(chunked_prefill_size*dp_size, max_running_requests), num_hidden_layers, topk+shared]`),CUDA graph 兼容(graph 捕获用的 buffer 预先分配)。同步层:每次 forward 结束(`on_forward_end`)把 device buffer 按 `out_cache_loc` 索引 D2H 拷进 pinned host cache(含 DP attention 的 local 偏移与 graph padding 处理);代码自注 `FIXME: sync explicitly here, overlap scheduler breaks here`——显式 `.cpu()` 同步与 overlap 调度冲突。输出层:请求完成时按 `req_to_token` 映射收集该请求各 token 的路由专家,经 `io_struct`/detokenizer 流出;为解决传输性能问题,专家 id 以 base64(int32 `tobytes` + `pybase64`)编码返回。默认关闭:`--enable-return-routed-experts` 启用,关闭时全局 capturer 是 Noop 实现、零开销。已兼容 CUDA graph;不兼容 deepep(下一 PR)、MTP(下一 PR)、PP(无计划)。

## 关键改动

```text
python/sglang/srt/layers/moe/routed_experts_capturer.py(新):
  device/host 双 buffer 捕获器(Real/Noop)、base64 提取辅助
python/sglang/srt/layers/moe/topk.py:
  TopK 增加 layer_id;select_experts 挂 capture 钩子
python/sglang/srt/managers/{io_struct,detokenizer_manager,tokenizer_manager,
  scheduler_output_processor_mixin,schedule_batch,scheduler}.py:
  return_routed_experts 请求字段与 output_routed_experts 输出链路
python/sglang/srt/models/{deepseek_v2,qwen2_moe,qwen3_moe,grok,...}.py:
  各 MoE 模型 TopK 传入 layer_id
```

```python
# 每层路由捕获:layer_id 为 None 时 assert,防止漏传
def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
    batch, _ = topk_ids.shape
    self.buffer[:batch, layer_id, :] = topk_ids
```

## 性能声明

PR 正文无 benchmark 数字(Accuracy/Benchmarking 小节为空,checklist 未勾选)。与性能相关的声明只有两点:TODO 列表中的 "use tobytes to improve performance"(base64/tobytes 编码减少输出序列化开销)与 device 捕获 buffer 的容量公式(按 chunked_prefill_size*dp_size 与 max_running_requests 取 max);功能开启时会引入每次 forward 一次显式 D2H 同步(代码注释注明 overlap scheduler 下会 break)。定量数字 PR 未注明。

## 验证与回归风险

新增 `test/srt/rl/test_return_routed_experts.py`(Qwen3-30B-A3B,TP4/DP4 + dp-attention,基线组合为 deterministic-inference + disable-overlap-schedule + disable-cuda-graph + disable-radix-cache,含 ShareGPT 数据);TODO 注明在 DeepSeek-V3 上测试过、已启用 CUDA graph 并建立 experts capture buffer。风险面:功能与 overlap scheduler 不兼容(FIXME 注明,依赖显式 sync);device buffer 按最大 token 容量常驻显存(num_layers × topk × int32),大模型上开销可观;deepep/MTP/PP 路径不支持但启动侧未列硬门(本 PR 范围);`max_running_requests` 计算逻辑从 tp_worker 上移到 model_runner,两处存在冗余(TODO 注明)。

## 关联

同区域(MoE topk):本批次 #7058(topk 后处理编译优化,同一 `topk.py`);路由信息后续消费(R3/RL 训练)在 sglang 之外。pack 未列出前序/后续 PR 编号(deepep/MTP 支持注明为 "next pr" 但未给编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
