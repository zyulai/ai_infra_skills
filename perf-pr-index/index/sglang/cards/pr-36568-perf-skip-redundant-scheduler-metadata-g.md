# PR #36568: perf: skip redundant scheduler metadata gather for DP1

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @YAMY1234 |
| 合并 | 2026-08-28 |
| 分类 | Speculative Decoding(classify 给出;改动实为 scheduler 元数据同步) |
| Label | run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及 2 个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/36568 |

## 一句话结论

DP1 时自动跳过冗余的 scheduler metadata all-gather,GB300 集群上纯 decode TPS/user +11.45%、decode step latency -10.35%。

## 动机

attention DP size 为 1 时,scheduler metadata 已经是本地且 rank-invariant:TP schedulers 消费同一 broadcast 请求流,再 gather 一次相同的 batch mode、graph eligibility 与 token counts 只增加一次 device collective 加 host 同步,却不调和任何跨 DP 状态。SGLang 此前已有 `SGLANG_SCHEDULER_SKIP_ALL_GATHER` 手动 override,本 PR 把 skip 变成 DP1 的默认行为,同时保留大 DP 尺寸下的显式 opt-in。

## 优化点

新增 `should_skip_scheduler_all_gather(dp_size)`:`dp_size == 1` 或 env override 为真即跳过。DP1 分支调用新增的 `finalize_local()`,从唯一的 attention-DP rank 本地填充原本由 gather 派生的元数据(`tp0_info_cpu`、`global_num_tokens`、`global_num_tokens_for_logprob`、可选 DP metrics),使 TBO split metadata、global/recv forward mode、token counts、graph eligibility 无需 collective 依然可用。`_update_gather_batch` 的参数从 `skip_all_gather` 重构为 `skip_global_metadata=not metadata_ready`,以 `tp0_info_cpu` 是否就绪为准;多 DP 且无 override 的路径保持不变。收益是每 decode step 都省掉一次 collective 与 host 同步,因此对延迟敏感的低并发 decode 场景最明显。

## 关键改动

```text
python/sglang/srt/managers/scheduler_components/dp_attn.py: DP1 本地 finalize 与 skip 策略函数
test/registered/unit/managers/scheduler_components/test_dp_attn.py: skip 策略与 DP1 本地元数据/TBO 路径单测
```

```python
def finalize_local(self):
    """Populate gather-derived metadata from the sole attention-DP rank."""
    self.tp0_info_cpu = self._get_local_tensor(device="cpu").view(1, -1)
    self.global_num_tokens = [self.num_tokens]
    self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob]

def should_skip_scheduler_all_gather(dp_size: int) -> bool:
    return dp_size == 1 or envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
```

(代码摘录略去 `_ENABLE_METRICS_DP_ATTENTION` 的可选 metrics 分支。)

## 性能声明

直接 A/B:baseline 为 main commit `8739d56a`,候选为 PR head patch `ae16c5cc`(未设 env,走本 PR 的自动 DP1 路径)。条件:2 节点 × 4 GB300 GPU,TP8/DP1/EP1/PP1,DP attention disabled,实际 batch size 1,CUDA Graph bucket 8,radix cache disabled,ISL/OSL 89,472/1,024,每臂 2 warmup + 8 测量请求(作者注明为单 job 配对测量,非重复统计基准):Pure decode TPS/user 351.46→391.70(**+11.45%**),TPS/GPU 同 +11.45%,mean decode step latency 14.336→12.852 ms(**-10.35%**),full-request TPS +5.63%,mean TTFT -0.64%(基本不变)。另注:该手动 override 在冻结 benchmark 栈上的历史测试显示同方向 18.01% TPS/user 改善;正文以 current-main 直接 A/B 为主要结果。

## 验证与回归风险

不改模型数学与输出,只改 DP1 的 scheduler 元数据同步。新增注册 CPU 单测:skip 策略(DP1 必 skip、DP2 默认不 skip、override 生效)与 DP1 本地路径(all_gather 未被调用,TBO split/forward mode/recv skipper 正确派生)。作者本地环境依赖冲突,完整 CPU CI 留给注册的 `base-a-test-cpu` shard。剩余验证范围(作者自列):DP1 extend/mixed batches、overlap scheduler 开关、PP 配置、DP1 + DP attention、多 DP 保持原 gather 路径。pack 中 CI:Base 进行中,Extra 与 AMD ROCm 7.2 为失败标记。

## 关联

前序特性:既有 env override `SGLANG_SCHEDULER_SKIP_ALL_GATHER`(正文提及,未给 PR 号)。同批索引 #36527 亦涉及 two-batch overlap 门控,区域相关。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
