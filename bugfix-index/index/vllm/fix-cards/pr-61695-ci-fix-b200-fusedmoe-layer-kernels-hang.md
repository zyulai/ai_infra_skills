# PR #61695: [CI] Fix B200 FusedMoE Layer Kernels hang on NVSHMEM IB setup

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#61695](https://github.com/vllm-project/vllm/pull/61695) |
| 作者 | mgoin |
| 合并日期 | 2026-10-16 |
| labels | ci/build |
| 热度 | 👍 0 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | hang |

## 一句话结论
单节点 DeepEP 测试依赖 NVSHMEM IB 初始化,偶发失败后 worker 不退出导致 CI 挂满 90 分钟;改走 NVLink P2P 并加超时快速失败。

## 问题现象
`(B200) FusedMoE Layer Kernels` CI 任务近两周 59/247 次 main run 失败(约 24%),几乎全部是挂起直到 90 分钟 job 超时。挂起点在 `test_moe_layer[False-deepep_low_latency-2-1-True]`:NVSHMEM 的 InfiniBand transport 初始化失败,spawn 出的 worker 永不退出。8 次超时的日志签名相同:

```
ibrc.cpp:427:  non-zero status: 22 ibv_modify_qp failed
ibrc.cpp:1526: non-zero status: 7 ep_connect failed
init.cu:1047:  non-zero status: 7 nvshmem setup connections failed
```

之后无任何输出直到 job 被取消。

## 根因
正文 Details 披露:这是 job 中**第一次** NVSHMEM init 就失败(非重初始化)——每个 subtest 新建 DeepEP low-latency buffer,通过的 run 会 init NVSHMEM 48 次;8 次 hang 全部停在恰好第 32 条 `cudaHostRegister` fallback 警告后,即第一个 subtest 结束前。CI pod 里 IBGDA 不可用,NVSHMEM 回退到 IBRC,IBRC 的 queue-pair connect 间歇性失败,worker group 随即挂住而不是退出。表现上与主机/争用相关:超时集中在 dgxB200-09/12/14(32 次中 8 次),而 01/11/16 为 0/23;8 次中有 5 次在另一同 host 任务开始后 2–5 分钟内启动;pod 能看到节点全部 12 块 NIC,并发 pod 共享。正文注明这是 "suggestive, not proven"。

## 修复方案
三处改动(均在测试/CI 配置,不碰生产代码):
- `tests/kernels/moe/test_moe_layer.py`:为 CUDA DeepEP 后端注入 `DEEPEP_SINGLE_NODE_NVSHMEM_ENV`,经现有 `env_dict` hook 传给 `parallel_launch_with_config`;ROCm/MoRI 路径不动:

```python
DEEPEP_SINGLE_NODE_NVSHMEM_ENV = {
    "NVSHMEM_REMOTE_TRANSPORT": "none",
    # DeepEP forces IBGDA on, so also hide the NICs from it.
    "NVSHMEM_HCA_LIST": "none",
}
```

单节点对端经 NVLink 走 NVSHMEM P2P transport(`nvshmemi_get_p2p_ptr` 对单节点 peer 均有指针,low-latency kernel 取 NVLink store 路径)。
- `tests/kernels/moe/modular_kernel_tools/parallel_utils.py`:`parallel_launch_with_config` 改 `join=False`,以 900 秒 deadline 轮询 `context.join`;超时则 kill 全部 worker 并抛 `TimeoutError`。正文验证:睡眠 worker 31 秒后报错、无残留子进程;通过 run 中最长单次 launch 约 195 秒,远低于 900 秒。
- `.buildkite/test_areas/kernels.yaml`:B200 job 超时 90 → 45 分钟(通过 run 15–18 分钟,两周内最慢 36 分钟)。

本地 2x B300 复现:low_latency 用例由 NVSHMEM init 失败(`cq creation failed`)变为 48/48 子用例通过。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围;CI 故障统计覆盖近两周。
- 回归面:改动只影响 `test_moe_layer` 的 DeepEP 测试与 Buildkite kernels.yaml。`test_moe_layer` 不再覆盖 IB transport 路径——正文承认这是代价,但认为单节点测试的 IB 覆盖本无意义(IB 仅靠 fallback 才能用);多节点 DeepEP 覆盖应放多节点测试。900 秒超时对 `parallel_launch_with_config` 的所有调用方生效,正常 launch 不受影响。若未来 runner 的 GPU 对无 peer 访问,low-latency 用例会清晰失败而非挂起。

## 关联
- vllm-project/vllm#53332(stalled draft,采用重试而非回避 IB 的替代方案)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
