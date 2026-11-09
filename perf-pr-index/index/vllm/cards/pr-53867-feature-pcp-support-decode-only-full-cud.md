# PR #53867: [Feature][PCP] Support decode-only FULL CUDA graphs

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @pisceskkk |
| 合并 | 2026-09-15 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | ready, nvidia, mrv2 |
| Diff | 素材未提供 diffstat(pack 记录为 +None/-None across None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/53867 |

## 一句话结论

让 PCP(prefill context parallelism)支持 decode-only FULL CUDA graph(`FULL_DECODE_ONLY`/`FULL_AND_PIECEWISE`),PCP 下 decode 走整图捕获回放以省 launch 开销。

## 动机

PCP 开启时 MRV2 worker 此前只允许 PIECEWISE CUDA graph(`validate_config` 对 `has_full_cudagraphs()` 直接抛 "MRV2 PCP supports PIECEWISE CUDA graphs only"),decode 无法使用 FULL 整图回放,每步付出 kernel launch 与 eager 调度开销。限制的根源是 PCP 会把全局 batch 分区为 rank-local batch,捕获与回放需要持久 rank-local 输入 buffer 与按模式推导的 padding 容量,此前未打通。

## 优化点

机制上是一组围绕 PCPManager 的捕获/回放基建:(1) 配置校验改为精确拒绝——仅当 `has_full_cudagraphs()` 且非 `separate_routine()`(即纯 `FULL`)时抛错,提示改用 FULL_DECODE_ONLY/FULL_AND_PIECEWISE/PIECEWISE/NONE;纯 FULL 仍被限制,因为 PCP prefill 走单独分派的执行路径。(2) `maybe_partition_pcp_batch` 改收 `BatchExecutionDescriptor`,在单一分支里按 graph 模式推导 padding:NONE 不 padding;PIECEWISE/FULL_DECODE_ONLY 只 pad token;FULL 再 pad request,分区后保留图捕获的 request/token 容量(`_resolve_num_reqs_after_padding` 校验 padded 容量必须覆盖 rank-local batch,有 prefill 时直接报 "PCP FULL graphs require a decode-only batch")。(3) `partition_batch` 为 padded 请求补齐零值 seq_lens/is_prefilling/dcp upper bound 等 host 侧数组,真实请求的语义不变。(4) 捕获路径统一:`prepare_inputs_to_capture` 把 dummy batch 拷入持久 PCP input buffers(data_ptr 与 manager buffer 一致),`get_dummy_block_tables` 返回 PCP-local block tables;`cudagraph_utils.prepare_inputs_to_capture` 与 `model_runner` 的 dummy 路径统一从 `pcp_manager or block_tables` 取 provider,`set_dummy_context` 支持传入 PCP-local 的 input_block_tables。

## 关键改动

```text
vllm/v1/worker/gpu/pcp_manager.py: validate_config 放行 decode-only FULL;分区按 BatchExecutionDescriptor 推导 token/request padding;新增 prepare_inputs_to_capture/get_dummy_block_tables
vllm/v1/worker/gpu/cudagraph_utils.py: 捕获输入统一经 pcp_manager 准备
vllm/v1/worker/gpu/model_runner.py: dummy attn/context 准备走 PCP-local block tables
vllm/v1/worker/gpu/input_batch.py: set_dummy_context 支持外部 input_block_tables
tests/v1/worker/test_gpu_pcp_manager.py: 配置校验、padding 推导、持久 buffer、PCP-local block tables 用例
```

关键摘录(模式→padding 的单点推导):

```python
if batch_desc.cg_mode != CUDAGraphMode.NONE:
    padded_num_tokens = batch_desc.num_tokens
    if batch_desc.cg_mode == CUDAGraphMode.FULL:
        padded_num_reqs = batch_desc.num_reqs
return manager.partition_batch(
    input_batch, padded_num_tokens=padded_num_tokens,
    padded_num_reqs=padded_num_reqs)
```

## 性能声明

无吞吐/延迟数字;PR 正文只给正确性对照。E2E 条件:NVIDIA H100 80GB HBM3 x2,GLM-4.7-Flash BF16,TP=1 / PCP=2 / EP=2,max model len 8192,max seqs 32,max batched tokens 32768;GSM8K 全量 1319 题、5-shot、并发 32、temperature 0、seed 42、max output 4096。结果:Eager 1077/1319 = 81.6528%(baseline);PIECEWISE 82.0318%(+0.3791 pp);FULL_DECODE_ONLY 81.5011%(-0.1516 pp);FULL_AND_PIECEWISE 82.2593%(+0.6065 pp);四种模式零 invalid completion、零请求错误,graph-mode delta 在 eager ±0.61 pp 内。即本 PR 的收益指向 decode 路径的 launch 开销,但正文未提供吞吐对比数据。

## 验证与回归风险

单元测试新增:`test_validate_config_accepts_decode_only_full_graphs`(两种模式)、`test_validate_config_rejects_full_graph_for_prefills`(FULL 抛 NotImplementedError)、`test_partition_padding_is_derived_from_batch_descriptor`、`test_capture_uses_pcp_persistent_inputs`(断言 data_ptr 指向 PCP buffer)、`test_dummy_context_updates_pcp_local_block_tables`;既有 DCP partition 测试更新为 padded 语义(seq_lens 补零、dcp upper bound 补零)。分支验证:mypy(Python 3.10 target)、ruff、GSM8K E2E YAML 断言、pre-commit workflow 通过。回归风险:PCP + 各种 CUDA graph 模式组合;`partition_batch` 的 padded request 语义变化影响 DCP 元数据派生;`prepare_dummy_attn`/dummy context 路径在有 pcp_manager 时改走 PCP-local block tables;纯 FULL 模式对 PCP 仍显式不可用。

## 关联

同区域:PCP/DCP 并行与 CUDA graph 模式(PR #52188 同批涉及 Kimi-K3 DCP;PR #42785 涉及 MM encoder CUDA graph);MRV2 稀疏 MLA PCP 仍不支持 CUDA graph(validate_config 中既有提示)。本卡所在批次素材中未列出其他同区域 PR 编号,暂无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
