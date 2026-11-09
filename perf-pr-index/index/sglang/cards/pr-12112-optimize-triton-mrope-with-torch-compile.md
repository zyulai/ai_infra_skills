# PR #12112: Optimize triton_mrope with torch compile

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuan-luo |
| 合并 | 2025-10-27 |
| 分类 | CUDA Graph / Compiler |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/12112 |

## 一句话结论

给 triton mrope 前后处理加 torch.compile,消除 #11722 引入的 op 开销,VLM TTFT 较 main 快 28%。

## 动机

#11722 的 Triton mrope 内核本身提速 30-40%,但引入的若干 Torch ops 开销导致端到端性能回退,甚至低于 legacy 的 native mrope 路径。需要在不放弃 kernel 收益的前提下消掉框架开销,让 triton mrope 的 E2E 性能重新超过 legacy 版本。

## 优化点

在 `_forward_triton` 上加 `@torch.compile(dynamic=True, backend=get_compiler_backend())`,并在 `triton_mrope` 调用前后各插一个 `torch._dynamo.graph_break()`:Dynamo 只编译 kernel 调用前后的 Torch ops(reshape 等前置/后置处理),triton kernel 本身保持黑盒不被追踪。改动仅 rotary_embedding.py 数行,不改 kernel。

## 关键改动

```text
python/sglang/srt/layers/rotary_embedding.py: _forward_triton 加 torch.compile 装饰器;triton_mrope 调用两侧插入 graph_break
```

```python
@torch.compile(dynamic=True, backend=get_compiler_backend())
def _forward_triton(self, ...):
    ...
    torch._dynamo.graph_break()
    q, k = triton_mrope(query, key, ...)
    torch._dynamo.graph_break()
    return q.reshape(query_shape), k.reshape(key_shape)
```

## 性能声明

VLM online latency(test_bench_serving.TestBenchServing.test_vlm_online_latency,250 请求,rate 1):对比 #11722 之前的 native mrope:TTFT 156.52→131.41 ms(16% speedup),E2E 8434.92→8272 ms(2% speedup);对比 main(triton mrope kernel):TTFT 182.89→131.41 ms(28% speedup),E2E 8903.92→8272 ms(7% speedup)。精度:MMMU 900 样本 overall accuracy 0.513,作者称 accuracy no drop。PR 未注明测试硬件型号。

## 验证与回归风险

MMMU benchmark(900 样本,overall 0.513)验证多模态精度。影响所有走 `_forward_triton` 的 mrope 路径;依赖 torch.compile/dynamo 的 graph break 行为与 `get_compiler_backend()`;checklist 各项(格式化、单测、文档、benchmark)未勾选,素材未附 CI 结果。

## 关联

#11722(前序 Triton mrope kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
