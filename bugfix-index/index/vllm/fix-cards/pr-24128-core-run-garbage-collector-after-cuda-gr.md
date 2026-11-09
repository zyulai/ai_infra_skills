# PR #24128: [Core] Run garbage collector after CUDA graph capture to fix throughput regression

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24128](https://github.com/vllm-project/vllm/pull/24128) |
| 作者 | micah-wil |
| 合并日期 | 2025-09-09 |
| labels | ready, v1 |
| 热度 | 👍 1 · 💬 5 |
| 分类 | CUDA Graph / Compiler |
| bug 类型 | correctness |

## 一句话结论
#21146 的 capture 期 gc 冻结引入约 5% 吞吐回退;在 capture 结束 unfreeze 后补一次 gc.collect() 恢复基线。

## 问题现象
#21146 引入 CUDA graph capture 期间的 gc 冻结后,部分测试出现吞吐回退,小输入/输出长度时最明显。8xMI300X 上用 benchmark_throughput.py(Llama-3.1-70B-Instruct-FP8-KV、tp 8、fp8 KV、input/output 128、FULL cudagraph 模式)测得:默认(冻结 gc)15694.6 tok/s;设 `VLLM_ENABLE_CUDAGRAPH_GC=1` 完全禁用冻结为 16416.11 tok/s(约高 5%),但 capture 阶段从 10.02it/s 慢到 1.68it/s;两者此前的取舍是"要吞吐就得忍受数倍的 capture 时间"。

## 根因
正文披露:回退由 #21146 造成——该 PR 为加速 capture 在其间冻结了 Python gc。作者观察到"在 CUDA graph capture 完成后立刻强制一次 gc.collect() 即可恢复吞吐",即回退源自冻结状态下 capture 结束后积累的垃圾未被回收,进而拖慢运行期;至于冻结期间垃圾积累影响运行期的具体机制,正文未展开,属(由 diff 推断):修复在 `gc.unfreeze()` 之后紧接着调用 `gc.collect()`,意味着冻结窗口内跳过的回收需要在窗口结束时一次性补做。

## 修复方案
改 `vllm/v1/worker/gpu_model_runner.py` 的 `freeze_gc`,在 capture 结束、`gc.unfreeze()` 后补一次收集,仅 1 行:

```diff
             finally:
                 if should_freeze:
                     gc.unfreeze()
+                    gc.collect()
```

效果(同基准、5 次平均):16334.29 tok/s,恢复到接近禁用冻结的 16416 水平(较回退的 15694 提升),同时 capture 速度保持 9.99it/s 不变——即保留 #21146 的 capture 加速,又拿回运行期吞吐。

## 影响与回归风险
- 影响窗口:自 #21146 引入 capture 期 gc 冻结起,至本修复前;受影响最明显的是小输入/输出长度、FULL cudagraph 模式的部署。
- 回归面:按 diff 仅在 gc 冻结的 finally 分支追加一次 `gc.collect()`,只影响走"冻结过 gc"的路径(should_freeze 为真);未冻结时行为不变。代价是 capture 完成时一次全量 GC 的短暂停顿(发生在启动阶段,不影响服务期),收益为运行期吞吐恢复。若后续在冻结窗口内新增大对象分配,该一次性收集的时长会相应增长。

## 关联
- [#21146](https://github.com/vllm-project/vllm/pull/21146)(引入 gc 冻结的 PR,正文指认为回退来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
