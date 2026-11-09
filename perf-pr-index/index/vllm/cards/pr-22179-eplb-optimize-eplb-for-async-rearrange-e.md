# PR #22179: [EPLB] Optimize EPLB for Async Rearrange Experts

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @david6666666 |
| 合并 | 2025-11-24 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 2 个文件因超长被略去 |
| 链接 | https://github.com/vllm-project/vllm/pull/22179 |

## 一句话结论

为 EPLB 专家权重重排引入非阻塞(async)模式:后台线程用独立 CUDA Stream 逐层把权重搬到预分配 buffer,主线程在 step 间以层为单位落位,把重排对 decode 时延的冲击摊平。

## 动机

同步 EPLB 在重排触发时一次性完成全部 MoE 层的专家权重 P2P 交换与拷贝,期间模型推理被阻塞,TPOT/ITL 出现尖刺(benchmark 中 main 分支 w/ EPLB 的 P99 TPOT 达 63.84 ms,而无 EPLB 仅 29.96 ms)。对应 EPLB 执行路线(issue #20805/#24116,初始 PR #18343)中的三项:一次 shuffle 一层、分多步落位以降低对 inter-token latency 的影响;预分配传输用的专家权重 buffer;用 `cuda.Stream()` 异步搬移权重到 buffer。作者还总结了异步实现的 5 个关键坑:stream 必须绑定设备、`move_to_buffer`(异步隐藏传输延迟)与 `move_to_workspace`(~300-400 µs 的设备内拷贝,同步执行)分离、逐层等待 P2P 完成、All Reduce 与 rank 同步耦合防死锁(且仅在 Rearrange 阶段执行)、每层 `move_from_buffer` 后立即更新专家权重以免用到旧权重。

## 优化点

机制上:(1) 新增 `async_worker.py` 后台线程:绑定目标 GPU 设备、创建专用 `torch.cuda.Stream` 与 asyncio loop,`rearrange_event` 触发后循环调用 `transfer_layer` 逐层执行 `move_to_buffer`(本地 copy 与 P2P `batch_isend_irecv` 均在指定 stream 上 non-blocking 执行),完成一层记录 CUDA event 并置 `ep_buffer_ready`。(2) 主线程 `step()` 中推进 `move_to_workspace`:等 `buffer_ready_event` 后从 buffer 拷回工作区、更新该层 mapping、`layer_to_transfer += 1`;跨 rank 的 buffer 就绪通过 CPU/device group 的 all_reduce 标志对齐(`_all_ranks_buffer_ready`)。(3) `rebalance_execute.shuffle_layer` 拆分为 `move_to_buffer` + `move_from_buffer` 两阶段,同步路径(`rearrange_expert_weights_inplace`)复用同一对函数(cuda_stream=None);buffer 在 `add_model` 时按层预分配(`torch.empty_like`)。(4) 配置入口 `EPLBConfig.use_async`(默认 False),`gpu_model_runner.load_model` 后启动 async loop;重排进行中会跳过新的 rearrange 触发。

## 关键改动

```text
vllm/distributed/eplb/async_worker.py: 后台异步线程(asyncio loop + 专用 CUDA Stream,逐层 transfer_layer)
vllm/distributed/eplb/eplb_state.py: EplbModelState 扩展 buffer/状态字段,step() 集成 move_to_workspace
vllm/distributed/eplb/rebalance_execute.py: shuffle_layer 拆分 move_to_buffer/move_from_buffer,新增 transfer_layer
vllm/config/parallel.py: EPLBConfig.use_async 开关
vllm/v1/worker/gpu_model_runner.py: load_model 后启动 async loop
```

关键片段(buffer 搬移在专用 stream 上执行):

```python
with torch.cuda.stream(cuda_stream):
    buffer[dst].copy_(weight[src], non_blocking=True)
...
if p2p_ops and cuda_stream is not None:
    with torch.cuda.stream(cuda_stream):
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()
```

## 性能声明

PR 正文给出 8xH800 80G、TP=8/EP=8、Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 的数字(eplb window_size=1000,step_interval=3000):

- GSM8K(1319 题,5-shot):w/o EPLB accuracy 0.901、总延迟 67.070 s(19.666 q/s);main 分支 w/ EPLB 0.902、82.434 s(16.001);本 PR w/ EPLB 0.908、80.133 s(16.460)。async EPLB + MTP(Qwen3-Next-80B-A3B-Instruct,tp4)accuracy 0.864、94.256 s(13.994)。
- `vllm bench serve`(mmlu_pro 数据集,max-concurrency 64,1000 prompts):w/o EPLB 请求吞吐 9.13 req/s、输出吞吐 2113.74 tok/s、Mean TTFT 93.20 ms、Mean TPOT 29.11 ms、P99 TPOT 29.96 ms;main w/ EPLB 7.58 / 1755.72 / 97.56 / 35.04 / P99 TPOT 63.84 ms;本 PR w/ EPLB 7.70 / 1777.61 / 100.83 / 34.47 / P99 TPOT 46.01 ms(相对 main 将 P99 TPOT 尖刺从 63.84 降至 46.01 ms)。

作者注明:受资源限制只测到 EP8,更大 EP 且非阻塞 EPLB 收益更大;profile 图显示 non-blocking EPLB 的传输已与推理重叠。

## 验证与回归风险

测试:`pytest tests/distributed/test_eplb_execute.py`(新增 async transfer_layer/move_from_buffer 的一致性验证,含冗余专家权重相同性检查)、`tests/distributed/test_eplb_spec_decode.py`。回归风险:async 模式下重排尚未完成时会推迟新的 rearrange 触发;`_all_ranks_buffer_ready` 每 rank 交互都要 all_reduce,死锁风险依赖"仅在 Rearrange 阶段执行"的约束;sync 路径重构为两阶段函数调用(行为应等价);MTP/投机解码与 async EPLB 组合依赖 event 同步正确性。

## 关联

初始 EPLB:PR #18343;路线图 issue:#20805、#24116。索引内同区域(EPLB/EP)卡片:#23745(静态 EPLB placement)、#28284(recording expert indices)、#28369(balance packing 重构)、#28164(TP-EP shared experts overlap)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
