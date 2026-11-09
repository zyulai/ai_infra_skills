# PR #38938: Bug/test eagle dp v0

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Monishver11 |
| 合并 | 2026-04-13 |
| 分类 | Speculative Decoding |
| Label | bug, documentation, performance, rocm, frontend, intel-gpu, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, cpu, gpt-oss, kv-connector, nvidia |
| Diff | pack 未提供 diffstat(+/- 与 files 数缺失,详见 PR 页面) |
| 链接 | https://github.com/vllm-project/vllm/pull/38938 |

## 一句话结论

修复 EAGLE DP 测试 flaky,补齐 batch invariance 在 lm_head 投影与 SM<90 设备上的覆盖。

## 动机

EAGLE DP 的 batch invariance 测试在 CI 中长期 flaky(issue #31913),非 ROCm 平台上一直挂着 xfail。排查发现两个缺口:其一,`UnquantizedLinearMethod.apply` 会检查 `VLLM_BATCH_INVARIANT` 并路由到确定性 Triton persistent kernel,但 lm_head 使用的 `UnquantizedEmbeddingMethod.apply` 缺少该检查,始终走 cuBLAS 的 `dispatch_unquantized_gemm()`,在 L4 上产生随 batch size 变化的 logits;其二,SM<90 上 torch.compile 与 CUDA graph 组合下 batch invariance 不可靠,而 EAGLE DP 是唯一没有按 PR #30018 模式设置 `enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90` 的 batch invariance 测试。作者还用 `cudagraph_mode=NONE` 做了隔离实验:仅关 CUDA graph 仍在 token 80 处发散,说明两者都有贡献。

## 优化点

本 PR 不改热路径性能,而是修复 batch-invariant 推理路径的覆盖完整性:当 `VLLM_BATCH_INVARIANT` 开启且为 CUDA 类设备时,lm_head 投影改走 `linear_batch_invariant` 确定性 kernel,与线性层行为对齐;EAGLE DP 测试在 SM<90(如 L4/SM89)上同时禁用 torch.compile 与 CUDA graph,H200(SM90)保持原行为;同时移除非 ROCm 平台的 xfail 标记,并把该测试加入 Buildkite 的 Distributed Tests (2 GPUs)(H100) 组,使其进入常规 CI 覆盖。

## 关键改动

```text
vllm/model_executor/layers/vocab_parallel_embedding.py: UnquantizedEmbeddingMethod.apply 增加 batch invariant 分支
tests/v1/distributed/test_eagle_dp.py: enforce_eager 按设备 SM 版本设置,移除非 ROCm xfail
.buildkite/test_areas/distributed.yaml: EAGLE DP 测试纳入 H100 2-GPU 分布式测试组
```

```python
if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
    return linear_batch_invariant(x, layer.weight, bias)
return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

## 性能声明

PR 未提供 benchmark 数字,正文也无性能声明;本 PR 为正确性/测试基础设施修复,服务于 batch-invariant 推理(确定性输出)路径的可用性。

## 验证与回归风险

依赖 `tests/v1/determinism` 的 batch invariance 测试模式(PR #30018 引入);新增 H100 2-GPU CI 步骤运行 `test_eagle_dp.py`。`vocab_parallel_embedding.py` 的改动影响所有使用 `ParallelLMHead` 的模型在 batch-invariant 模式下的 logits 路径;作者注明 utils.py 中"SM<90 仅 CUDA graph 不支持"的注释不完整,torch.compile 同样 implicated, broader tracking 在 issue #39096。

## 关联

前序:PR #30018(确立 SM<90 batch invariance 测试 enforce_eager 模式);issue #31913、#39096。同分类无直接性能优化 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
