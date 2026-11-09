# PR #37376:  fused qknorm+rope kernel optimization for SM9.0

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @EricccYang |
| 合并 | 2026-04-13 |
| 分类 | Attention / Kernel \| tier: major |
| Label | performance, ready, ci/build, qwen, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未标注 dropped from diff 说明（diff 完整呈现） |
| 链接 | https://github.com/vllm-project/vllm/pull/37376 |

## 一句话结论

H100 上 fused qknorm+rope kernel 按 token 数动态调 warp 负载，P99 TTFT 199.65→180.09ms。

## 动机

PR 正文 Purpose 指出，qknorm+rope fused kernel 在 H100 上反而比 unfused Triton kernel 更慢，主要原因是 "1-head per warp" 模式在大 token batch 下表现不佳。本 PR 针对该瓶颈优化，正文标注 Related issue：#34391，分类为 SM9.0 的 kernel 优化。

## 优化点

正文说明的改动是：根据 batch 中的 token 数动态调整每个 warp 的工作量（workload per warp），阈值（threshold）通过 offline benchmarking 计算得到，从而缓解 1-head-per-warp 在大 batch 下的低效。结合 diff 可见，改动引入 `forced_token_heads_per_warp` 参数并经 fusion pass 与 functionalization 透传；新增 `csrc/async_util.cuh`，封装 SM80+ 的 `cp.async.cg/ca.shared.global` 异步全局到共享内存拷贝（ROCm 与低架构回退为直接 int4 拷贝），配合 `fused_qknorm_rope_kernel.cu` 的内核重写。PR 正文未给出更细的算法描述。

## 关键改动

```text
csrc/fused_qknorm_rope_kernel.cu: 重写 fused qknorm+rope kernel，按 token 数动态分配每 warp 工作量
csrc/async_util.cuh: 新增 cp_async_shared_global_16_cg / cp_async_shared_global_ca 等 cp.async 异步拷贝封装，SM80+ 走 PTX、ROCm/其他回退 int4 拷贝
vllm/compilation/passes/fusion/qk_norm_rope_fusion.py、vllm/compilation/passes/utility/fix_functionalization.py: fusion pass 与 functionalization 透传 forced_token_heads_per_warp 参数
csrc/ops.h、csrc/torch_bindings.cpp、vllm/_custom_ops.py: 把新参数接入 op 绑定与 Python 侧接口
```

```text
                     "is_neox",
                     "position_ids",
+                    "forced_token_heads_per_warp",
                 )
```

（摘自 `vllm/compilation/passes/utility/fix_functionalization.py`：新增参数进入 mutated_args。）

## 性能声明

正文 Performance 表（H100，`vllm bench serve` 在相同 request rate 下对比 unfused / fused baseline(main) / fused new）：Request throughput (req/s) 7.84 / 7.81 / 7.82；Output token throughput (tok/s) 1003.09 / 999.42 / 1000.73；Total token throughput (tok/s) 9027.77 / 8994.75 / 9006.55；Mean TTFT (ms) 113.55 / 116.39 / 112.31，Median TTFT 106.07 / 108.02 / 103.11，P99 TTFT 179.22 / 199.65 / 180.09；Mean TPOT 23.10 / 23.68 / 23.15，P99 TPOT 28.19 / 28.84 / 28.37；Mean ITL 23.10 / 23.68 / 23.15，P99 ITL 77.66 / 88.56 / 79.47。测试条件：H100（SM9.0）、`vllm bench serve` 相同 request rate 三方对比；具体模型、精度与 workload PR 未注明。

## 验证与回归风险

PR 引入/依赖的测试：Test Plan 为 `vllm bench serve` 对 unfused Triton / baseline fused / improved fused 在相同 request rate 下的对比；Accuracy 部分给出 lm_eval，正文结论 "lm_eval tests show no accuracy reduction"，gsm8k flexible-extract 0.8726→0.8757、strict-match 0.8893→0.8923，hellaswag acc 0.5190→0.5197、acc_norm 0.6527→0.6537（unfused → fused）。PR 未注明回归风险；kernel 面向 SM9.0/H100，阈值由 offline benchmarking 标定，其他 GPU 架构、更小 batch 与不同精度的行为正文未说明。

## 关联

正文标注 Related issue：https://github.com/vllm-project/vllm/issues/34391（issue #34391）；未给出其他 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
