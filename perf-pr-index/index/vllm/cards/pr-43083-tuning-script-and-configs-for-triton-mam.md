# PR #43083: Tuning script and configs for Triton Mamba SSU kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @danisereb |
| 合并 | 2026-05-24 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 可用但被截断(9 个文件未收录) |
| 链接 | https://github.com/vllm-project/vllm/pull/43083 |

## 一句话结论

为 Mamba selective_state_update(SSU)Triton 内核引入按 effective_batch 调优的 JSON 配置体系与调优脚本,B200 上对比启发式默认最高约 1.5× 内核提速。

## 动机

SSU 内核此前只用按 dstate 硬编码的启发式 launch 配置(BLOCK_SIZE_M/num_warps),对所有 (batch×nheads) 组合一刀切,在不同 effective_batch 下偏离最优(如 B200 float32 cache 下 eb=8 时 3.14 µs vs 最优 1.98 µs)。PR #41398 引入 Triton SSU 后的 code review 要求:使用 effective batch 维度、记录 triton_version、为 Nemotron Nano/Super 提供调优配置、支持 mamba cache dtype (float32/float16) 维度( closes issue #33034)。

## 优化点

镜像 fused MoE 调优工作流:新增 `benchmarks/kernels/benchmark_selective_state_update.py`,在 effective_batch = batch × nheads 网格上扫 (BLOCK_SIZE_M, num_warps),用 CUDA graph capture-and-replay 计时(隔离 Python dispatch 开销,20 warmup + 100 iters),BLOCK_SIZE_M 候选按 next_pow2(headdim) 封顶,并过滤 CUDA grid dim 65535 与 int32 溢出上限 effective_batch 262144(覆盖 Nemotron Super TP1 BS=2048);最优配置按 (headdim, dstate, cache_dtype, device) 命名存为 JSON(含 triton_version 信息字段),运行时由 `selective_state_update` 的配置加载器按最近 effective_batch 插值选取,无配置文件时回退启发式;支持 `--mamba-ssm-cache-dtype float32/float16`(bf16 与 fp16 共用配置),`VLLM_TUNED_CONFIG_FOLDER` 可覆盖配置目录;调优结果经 `--validate` 与 fp32 上转的参考实现比对,失败配置不落盘。

## 关键改动

```text
benchmarks/kernels/benchmark_selective_state_update.py: 新增调优/对比/校验脚本(CUDA graph 计时)
vllm/model_executor/layers/mamba/ops/mamba_ssm.py: JSON 配置加载、cache_dtype 维度与最近 eb 插值(diff 未直接展示)
tests/kernels/mamba/test_mamba_ssm_configs.py: 配置加载器单测(文件名/env 覆盖/回退/插值/坏 JSON)
tests/kernels/mamba/utils.py: 抽出 selective_state_update_ref 共享参考实现
```

计时方法核心:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(graph_batch_size):
        _call_kernel()
# divide by (num_iters * graph_batch_size), ms -> us
```

## 性能声明

调优环境:NVIDIA B200 / GB200(sm_100)与 H100 80GB(sm_90),bfloat16,Triton 3.6.0,headdim=64、ngroups=8、dstate=128。

- B200、ssm_cache float32:全部 effective_batch 提速 1.16–1.58×(eb=8:3.14→1.98 µs,1.58×;eb=4096:69.76→46.80 µs,1.49×;eb=262144:1.35×),21/21 配置校验通过(atol=0.01)。
- B200、float16 cache:1.01–1.39×,大 eb 区间约 1.3×。
- GB200、float32 cache:1.18–1.56×(大 eb 稳定 ~1.51×);float16 cache:1.05–1.45×。
- H100(sm_90)收益较小:float32 cache 1.01–1.41×(中大 eb 约 1.07–1.09×),说明 Blackwell 上启发式偏差更大。
- PR 正文未给出 E2E serving 数字(声称用生成配置做了 Evals 与 E2E benchmarks,但未列数据)。

## 验证与回归风险

测试:新增 `test_mamba_ssm_configs.py` 覆盖配置文件名格式、env 覆盖、无配置回退启发式、最近 effective_batch 插值、非 dict JSON 与空配置边界;`test_mamba_ssm.py` 的参考实现移入共享 `tests/kernels/mamba/utils.py`。风险面:配置按 (device, headdim, dstate, cache_dtype) 键查找,未覆盖组合回退启发式、无行为变化;`override_ssm_config` 临时覆盖与缓存 (`get_ssm_configs.cache_clear`) 的正确性由单测保障;int32 溢出上限 262144 之外的超大 effective_batch 超出调优范围。

## 关联

前序 PR #41398(Triton SSU 内核);closes issue #33034;同区域可参考索引中 Triton 内核调优卡(如 pr-16488 fused MoE tuned configs)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
