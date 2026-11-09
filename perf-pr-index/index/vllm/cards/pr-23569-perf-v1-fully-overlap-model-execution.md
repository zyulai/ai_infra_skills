# PR #23569: [Perf][V1] Fully overlap model execution

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @benchislett |
| 合并 | 2025-09-06 |
| 分类 | Speculative Decoding (tier: major) |
| Label | performance, ready, v1 |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 集中在 `gpu_model_runner.py`、`multiproc_executor.py`、`outputs.py`、`gpu_worker.py`、`gpu_input_batch.py` |
| 链接 | https://github.com/vllm-project/vllm/pull/23569 |

## 一句话结论

V1 async scheduling 下 model runner 异步返回 GPU 采样张量,完全重叠 CPU 输入准备与 GPU 前向,预期提速 5-10%。

## 动机

async scheduling 已部分重叠 CPU/GPU,但 execute_model 仍在采样后同步做 GPU→CPU 拷贝并 python 化 token ids,prepare_inputs 等与上一轮前向串行。目标是让 model runner 在 GPU 处理上一迭代时就跑到前面处理新输入,消除输入准备与采样的 CPU 开销。作者预期比现有 async scheduling 快 5-10%。

## 优化点

核心机制:execute_model 在 async scheduling 下不再做 GPU→CPU 同步,而是返回 `AsyncGPUModelRunnerOutput` 包装——在独立 CUDA copy stream 上发起 `sampled_token_ids` 的非阻塞 host 拷贝并记录 event;专用输出线程(`async_output_busy_loop`)随后阻塞在 event 上完成 `tolist()` 再放入输出队列。下一步的关键是 `_prepare_input_ids`:上一轮采样到的 token 不再回写 `output_token_ids`/`token_ids_cpu`,而是把 GPU 张量引用缓存为 `prev_sampled_token_ids`,在下一轮 prepare_inputs 期间直接从 GPU scatter 进 `input_ids`;batch 无重排时走单 slice 拷贝的 common-case 优化。同时 ModelRunnerOutput 中的 req_ids/req_id_to_index 改用返回前拷贝的副本,避免异步窗口内被修改。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: AsyncGPUModelRunnerOutput(copy stream
  + event)、_prepare_input_ids(prev_sampled_token_ids 回填)、execute_model
  异步分支
vllm/v1/outputs.py: AsyncModelRunnerOutput 抽象基类
vllm/v1/executor/multiproc_executor.py: WorkerProc 异步输出队列与
  handle_output/enqueue_output
vllm/v1/worker/gpu_input_batch.py: prev_sampled_token_ids 等缓存字段
vllm/v1/worker/gpu_worker.py: execute_model 返回类型扩展
```

关键片段(异步拷贝包装):

```python
with torch.cuda.stream(async_output_copy_stream):
    async_output_copy_stream.wait_stream(default_stream)
    self._sampled_token_ids_cpu = self._sampled_token_ids.to(
        'cpu', non_blocking=True)
    self._async_copy_ready_event.record()
```

## 性能声明

PR 未给出量化 benchmark 数字,仅有:预期比当前 async scheduling 提速 5-10%;以及 nsys profile 对比图(decode 迭代 BS=1,Llama 3.2 1B-Instruct,1xB200,async scheduling + full cudagraph,两种配置均开启),PR 正文未粘贴具体数值。"Test Result" 一节为空,测试计划注明"coming soon"。

## 验证与回归风险

PR 未附正式测试;作者说明正确性可由 async scheduling 端到端测试验证,但难点是防止 execute_model 中混入新的同步点——当前 FlashInfer 实现就有两处同步,因此基准改用 Flash Attention。不兼容:Pipeline Parallelism、投机解码(含 n-gram,后者必须在 CPU 上拿到 token)、guided decoding。作者指出与 #23224(guided decoding 进程化)和 #22684(MLA+MTP 去 GPU→CPU 同步)的集成路径。风险面:采样结果延迟一拍才落 CPU 状态,任何依赖实时 token id 的功能都可能错。

## 关联

同分类(投机解码运行时去同步化方向):#22684(MLA+MTP 重构)、#5765、#13365、#24505。相关 guided decoding 重构:#23224。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
