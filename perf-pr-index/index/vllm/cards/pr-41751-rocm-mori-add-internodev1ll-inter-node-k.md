# PR #41751: [ROCm] mori: add InterNodeV1LL inter-node kernel selection via VLLM_MORI_INTERNODE_KERNEL

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jatseng-ai |
| 合并 | 2026-05-27 |
| 分类 | Attention / Kernel (tier: major) |
| Label | rocm, ready, nvidia |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);diff 本身可用,含 7 个文件(all2all、communicator、config 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/41751 |

## 一句话结论

让 MoRI EP all2all 在多节点下可选 `InterNodeV1LL` kernel,MI300X RoCE v2 上 dispatch 带宽约为原 `InterNodeV1` 的 2x。

## 动机

MoRI 通过 `EpDispatchCombineKernelType` 暴露 5 种 EP dispatch/combine kernel(IntraNode / InterNode / InterNodeV1 / InterNodeV1LL / AsyncLL),其中 `InterNodeV1LL` 是官方推荐的多节点 kernel,MoRI README 带有其 benchmark 数字;但 vLLM 在 `MoriAll2AllManager._make_all2all_kwargs` 中硬编码 `InterNodeV1`,`InterNodeV1LL` 从 vLLM 侧完全不可达,性能差距因此长期未被注意。正文还指出 `InterNodeV1LL` 虽名为 low latency,在 MI300X 多节点(RoCE v2)集群的 dispatch 吞吐同样显著优于 `InterNodeV1`。

## 优化点

选择机制的改动(以 diff 为准):`mori` all2all backend 拆分为 `mori_high_throughput` 与 `mori_low_latency` 两个值(镜像 deepep_high_throughput/deepep_low_latency 的划分),注册进 `ParallelConfig` 的合法 backend 列表;`MoriAll2AllManager.__init__` 接收 backend 名并断言取值,`_make_all2all_kwargs` 多节点分支按 backend 选择 kernel——`mori_low_latency` 映射 `InterNodeV1LL`,`mori_high_throughput` 保持 `InterNodeV1`;`cuda_communicator.py` 构造时把 backend 传入 manager;`fused_moe/config.py` 的 `use_mori_kernels`、`use_sequence_parallel_moe` 等判定同步扩展到两个新值;测试侧 `BACKENDS` 列表与 `BACKEND_SUPPORTED_QUANTS`、`BACKEND_EP_DP_TP_SUPPORT` 表按两个 backend 展开(量化支持、DP/TP/SP 矩阵相同)。PR 标题与正文描述的是经 `VLLM_MORI_INTERNODE_KERNEL` 环境变量选择(默认 `"V1"`),最终 diff 落地为 all2all-backend 拆分,默认行为(即 high_throughput → InterNodeV1)不变。

## 关键改动

```text
vllm/distributed/device_communicators/all2all.py: MoriAll2AllManager 接收 backend,多节点按 backend 选 InterNodeV1LL / InterNodeV1
vllm/config/parallel.py: backend 列表与文档替换 mori -> mori_high_throughput / mori_low_latency
vllm/distributed/device_communicators/cuda_communicator.py: 构造 MoriAll2AllManager 时传入 backend
vllm/model_executor/layers/fused_moe/config.py: use_mori_kernels 覆盖两个 backend
tests/kernels/moe/(test_moe_layer.py 等): BACKENDS 与支持矩阵按新 backend 展开
```

kernel 选择(节选):

```python
# Multi-node: kernel follows --all2all-backend (mirrors deepep_* split).
# mori_low_latency → InterNodeV1LL; mori_high_throughput → V1.
if self._all2all_backend == "mori_low_latency":
    kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
else:
    kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
```

## 性能声明

PR 正文声明:MI300X 多节点集群(RoCE v2)上 `InterNodeV1LL` 对比 `InterNodeV1` 的 dispatch 带宽约 2x higher,与 MoRI README 的 benchmark 数字一致;据此认为 `InterNodeV1LL` 应作为一般多节点 EP 的首选 kernel,而非仅延迟敏感场景。测试条件:PR 未注明节点数、模型、token 规模与测量工具,dispatch 带宽为 kernel 级对照(引用上游 README)。

## 验证与回归风险

正文 Testing 清单:单节点 EP 路径不受影响(IntraNode kernel 不变);多节点默认行为不变(默认仍走 V1,即 mori_high_throughput 语义);`V1LL` 在 MI300X 多节点(RoCE v2)验证,dispatch 带宽与 MoRI README 一致、约为 V1 的 2x。回归风险:`mori` 旧 backend 字符串被移除,现有部署若显式配置 `--all2all-backend mori` 需改用新名称;`MoriAll2AllManager` 新增断言限制取值;两个 backend 的量化支持与 EP/DP/TP/SP 矩阵在测试中相同,行为差异集中在 kernel 选择。

## 关联

同区域多节点 EP all2all 优化:deepep_high_throughput / deepep_low_latency(本 PR 明确镜像其划分),以及 nixl_ep、flashinfer_nvlink_* 等同列表 backend。具体关联 PR 编号素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
