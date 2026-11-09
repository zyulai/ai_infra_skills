# PR #23827: [NPU] Support shared expert dual stream optimization

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @iridiumine |
| 合并 | 2026-05-11 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 +N/-N 统计),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/23827 |

## 一句话结论
NPU 上把 MoE shared expert 卸载到独立 NPU stream 与 routed expert 并行,Qwen3.6-35B-A3B 吞吐 +11%。

## 动机
NPU 上 MoE 模型的 shared expert 与 routed expert 在同一 stream 上顺序执行,算力资源利用不足。本 PR 引入 dual stream 优化,把 shared expert 计算放到独立 NPU stream(专用 core 分配,8 cores)与 routed expert 重叠执行。

## 优化点
`cmo.py` 新增 `shared_expert_on_independent_stream`:懒创建并缓存全局 share stream,`stream.wait_stream(current)` 建立依赖后在 `torch.npu.stream(stream)` 上下文中执行传入的 forward 函数;配套 `get/set/wait_share_stream` 辅助函数。`Qwen2MoeModel._forward_deepep` 在三个条件同时满足时启用(`is_npu()`、`SGLANG_NPU_USE_MULTI_STREAM` 开启、当前处于 CUDA graph 模式):shared expert 以 `hidden_states.clone()` 在独立 stream 上发起,主 stream 继续做 gate/topk/routed expert,MoE 输出合并前 `wait_share_stream()` 同步,最后 `final_hidden_states.add_(shared_output)`。

## 关键改动

```text
python/sglang/srt/hardware_backend/npu/cmo.py: shared_expert_on_independent_stream + share_stream 生命周期
python/sglang/srt/models/qwen2_moe.py: _forward_deepep 按 env/graph-mode 分派双流,合并前 wait_share_stream
```

## 性能声明
Qwen3.6-35B-A3B、NPU、128 并发请求、TP4 + dp2 + DeepEP + NEXTN 投机(`--speculative-num-steps 3 --speculative-num-draft-tokens 4`):throughput 1.72 → 1.91 req/s(+11.0%);Output token throughput 2580.73 → 2868.83 tok/s(+11.2%);Total token throughput 8602.42 → 9562.76 tok/s(+11.2%);Mean E2E Latency 57695 → 52973 ms(-8.2%);Mean TTFT 14646 → 13708 ms(-6.4%);Mean TPOT 28.72 → 26.19 ms(-8.8%)。精度:GSM8K(256 题)0.9766、ceval(1346 题)0.893。

## 验证与回归风险
PR checklist 显示未添加单元测试(该项未勾选);验证为端到端吞吐 benchmark 与 GSM8K/ceval 评测(开启 `SGLANG_NPU_USE_MULTI_STREAM=1` 的完整 serving 配置)。风险:双流路径仅在 CUDA graph 模式且 env 开启时生效;`hidden_states.clone()` 增加一次显存拷贝;共享 stream 为进程级全局单例,跨层复用时正确性依赖 `wait_stream`/`wait_share_stream` 的时序约定。

## 关联
无(同 NPU MoE 方向另见 #39881 的融合 gmm1 kernel,素材中未标注直接关系)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
