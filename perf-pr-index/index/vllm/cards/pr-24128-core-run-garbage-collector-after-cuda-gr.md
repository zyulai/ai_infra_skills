# PR #24128: [Core] Run garbage collector after CUDA graph capture to fix throughput regression

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @micah-wil |
| 合并 | 2025-09-09 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready, v1 |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际仅改动 1 个文件、净增 1 行 |
| 链接 | https://github.com/vllm-project/vllm/pull/24128 |

## 一句话结论

CUDA graph capture 结束解冻 GC 后补一次 gc.collect(),修复 #21146 引入的约 5% 吞吐回归并保留 capture 提速。

## 动机

PR #21146 在 CUDA graph capture 期间冻结 GC,使 capture 提速(10.02 it/s,约 6 秒;不冻结时仅 1.68 it/s,约 39 秒),但在部分场景引入吞吐回归:8xMI300X 基准上从 16416 降到 15694 tok/s,小输入/小输出长度时最明显。设置 `VLLM_ENABLE_CUDAGRAPH_GC=1` 关闭 gc freeze 虽可恢复吞吐,却会失去 capture 提速。回归根因与 capture 期间被冻结的 GC 有关——积压的回收工作滞留到了服务运行期。

## 优化点

在 `freeze_gc()` 上下文管理器的 `finally` 分支中,`gc.unfreeze()` 之后追加一次 `gc.collect()`:capture 完成时立即把冻结期间积压的对象一次性回收掉,避免这些对象滞留到服务运行期再触发零散 GC。这样既保留了 #21146 的 capture 期间无 GC 开销收益(capture 仍约 6 秒),又消除了运行期的吞吐回归。改动本身只有一行,位于 v1 GPU model runner 的 capture 流程内。

## 关键改动

```text
vllm/v1/worker/gpu_model_runner.py: freeze_gc() 的 finally 中 gc.unfreeze() 后追加 gc.collect()
```

关键代码摘录(<=15 行):

```python
finally:
    if should_freeze:
        gc.unfreeze()
        gc.collect()
```

## 性能声明

- 基准:Llama-3.1-70B-Instruct-FP8-KV,8xMI300X,TP=8,benchmark_throughput.py,1024 prompts,input-len 128 / output-len 128,fp8 KV cache,FULL cudagraph mode,--max-num-seqs 1024,取 5 次平均:
  - main(#21146 后):15694.636 total tok/s(capture 10.02 it/s)。
  - `VLLM_ENABLE_CUDAGRAPH_GC=1`(关闭 gc freeze):16416.11 total tok/s,但 capture 降至 1.68 it/s(约 39 秒)。
  - 本 PR:16334.29 total tok/s,capture 保持约 9.99 it/s(约 6 秒)。
- 结论:以几乎不增加 capture 时间的代价恢复约 5% 的吞吐(16334.29 vs 15694.64,约 +4.1%;PR 表述为恢复 baseline)。

## 验证与回归风险

- 验证手段为 benchmark_throughput.py 的 A/B 对照,pack 中未显示新增单测;gc.collect() 在 capture 之后同步执行,会增加一次启动期 GC 的耗时(该成本体现在启动而非稳态吞吐)。
- 影响路径:所有启用 CUDA graph capture 的 v1 GPU worker;GC 冻结逻辑本身(#21146)不变,风险集中在 freeze/unfreeze 状态切换后的 collect 语义(如与进程内其他 GC 使用方的交互)。对非 CUDA graph 路径无影响。

## 关联

- 前序:PR #21146(freeze GC during CUDA graph capture,索引库已有卡片 pr-21146);本 PR 修复其吞吐回归。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
