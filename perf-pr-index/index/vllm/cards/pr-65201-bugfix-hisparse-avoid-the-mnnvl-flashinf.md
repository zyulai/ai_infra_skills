# PR #65201: [Bugfix][HiSparse] Avoid the MNNVL FlashInfer all-reduce with HiSparse

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @MatthewBonanni |
| 合并 | 2026-11-02 |
| 分类 | Attention / Kernel(tier: major) |
| Label | bug, ready, nvidia, hisparse |
| Diff | diffstat 缺失(pack 未提供加/删行数);diff 显示改动 2 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/65201 |

## 一句话结论

HiSparse 开启时把 FlashInfer allreduce 后端从 MNNVL 换成 trtllm(单机),规避会导致引擎崩溃的 cluster-barrier race。

## 动机

FlashInfer 的 MNNVL fused all-reduce + residual + RMSNorm 存在 cluster-barrier 竞态(flashinfer-ai/flashinfer#7782)。自 #61989 起 vLLM 单机默认用 MNNVL 后端;而 HiSparse 会在 copy stream 上做 host-memory gather,与该竞态叠加时引擎直接死于 `CUDA error: unspecified launch failure`(Xid 13 `CTA Not Present`)。B200 上 8 卡 GLM-5.3-NVFP4 TP8 + HiSparse 用 MNNVL 时 7/7 次运行在第一波就崩溃,属可用性阻塞问题(fixes #64327, unblocks #63801)。

## 优化点

这是稳定性修复而非提速:在 `flashinfer_all_reduce.py` 的 `_resolve_fi_ar_backend` 中,当 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto` 且检测到 `attention_config.hisparse_config` 非空时,走专门的分支——单节点返回 `trtllm` 后端并打 `info_once` 日志;多节点直接禁用 FlashInfer allreduce(返回 `None`,因为 trtllm 不支持多节点进程组)。用户显式设置的 `VLLM_FLASHINFER_ALLREDUCE_BACKEND` 仍然生效。代码注释标明这是临时方案,待 FlashInfer 发布包含 #7782 修复的版本后应删除。

## 关键改动

```text
vllm/distributed/device_communicators/flashinfer_all_reduce.py: _resolve_fi_ar_backend 增加 HiSparse 分支(auto 模式下单机用 trtllm、多机禁用)
tests/distributed/test_comm_ops.py: test_flashinfer_auto_backend_uses_group_topology 参数化新增 hisparse 维度与 2 个用例
```

```python
if config is not None and config.attention_config.hisparse_config is not None:
    if node_count == 1:
        return "trtllm", False
    logger.warning_once("FlashInfer allreduce is disabled: ...")
    return None, False
```

## 性能声明

对比基准(4×H100,GLM-5.3 dummy weights 截到 5 层 + HiSparse + MTP spec decode,random 52000/1024、32 prompts、6 waves 均值):`mnnvl` 499 output tok/s,`auto`(本 PR,实际选 trtllm)552 output tok/s;两者均 6/6 waves 无失败请求(此缩减版服务在 H100 上未触发崩溃)。8×B200 GLM-5.3-NVFP4 TP8 + HiSparse(引自 #64327/flashinfer#7782):MNNVL 7/7 崩溃,TRTLLM 4/4 waves 稳定,decode 吞吐约低 3%。

## 验证与回归风险

`pytest tests/distributed/test_comm_ops.py -k "flashinfer_auto_backend or workspace_failure or standalone_workspace"`,10 passed;新增的 2 个 HiSparse 用例在改动前失败、改动后通过。风险点:仅影响 auto 模式下的后端选择,显式 env 配置不受影响;多节点 + HiSparse 组合会整体禁用 FlashInfer allreduce,回退到默认通信路径,性能特性改变;该绕行逻辑与 FlashInfer 版本耦合,升级含修复的 FlashInfer 后需移除 TODO 标记的检查。

## 关联

同区域:#61989(将单机默认 allreduce 后端切到 MNNVL,即本 PR 临时绕开的默认路径)、#64327(崩溃主 issue)、#63801(被本 PR unblock 的 HiSparse PR)、#20691(FlashInfer fused allreduce 集成)。
