# PR #38990: [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @voipmonitor |
| 合并 | 2026-04-05 |
| 分类 | MoE(major) |
| Label | bug, ready, ready-run-all-tests |
| Diff | diffstat 缺失(diff 全文已提供,未附行数统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/38990 |

## 一句话结论

修复 #35153 引入的 TP-only MoE 模型 6-8% decode 吞吐回退:恢复 aux CUDA stream 上的 shared expert 与 routed expert 并行重叠。

## 动机

#35153 改动后,`SharedExperts._determine_shared_experts_order()` 里 `_has_external_experts` 检查先于 aux-stream 重叠判定执行。TP-only 配置(无 EP/EPLB)下该属性恒为 True(因为"非 EPLB 且非 flashinfer two-sided kernel"即视为 external 可用),于是无条件选中 `EXTERNAL`——shared expert 在主 stream 上先于 routed experts 串行执行;`MULTI_STREAM_OVERLAPPED` 路径(0.18.x 行为,shared expert 在 aux stream 上并行)永远走不到。影响所有多卡 TP 无 EP 的 MoE 模型(DeepSeek-V3、GLM-5 等)。PR 注明该回退未波及任何已发布版本。

## 优化点

把属性反转为 `_use_external_experts`:语义改为"是否强制 EXTERNAL"——仅当 EPLB 启用且 all2all 后端非 `allgather_reducescatter`(正确性考虑)或使用 flashinfer NVL two-sided kernels(DP 下无重叠收益)时返回 True;`_use_dp_chunking` 时返回 False。`_determine_shared_experts_order` 中据此只有被强制时才返回 `EXTERNAL`,否则继续按原有逻辑尝试 aux stream 的 `MULTI_STREAM_OVERLAPPED` 重叠,使 TP-only 部署重新获得 shared/routed 专家并行执行。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/runner/shared_experts.py: _has_external_experts 反转为 _use_external_experts 并调整判定顺序
```

判定逻辑节选:

```python
@property
def _use_external_experts(self) -> bool:
    if self._use_dp_chunking:
        return False
    backend = self._moe_config.moe_parallel_config.all2all_backend
    return (
        self._moe_config.moe_parallel_config.enable_eplb
        and backend != "allgather_reducescatter"
    ) or self._moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
```

## 性能声明

GLM-5 MoE,8×RTX PRO 6000 Blackwell,TP=8,C=1,无 MTP,decode 吞吐(Before=EXTERNAL → After=MULTI_STREAM):

- ctx 0:36.8 → 41.8 tok/s(+14%)
- ctx 16k:34.8 → 37.8 tok/s(+9%)
- ctx 32k:33.8 → 36.8 tok/s(+9%)

修复后 vLLM 0.19.1(41.8/37.8/36.8)优于 0.18.1(39.8/37.7/35.8)并与 SGLang 持平(同 cutlass MoE backend、无 custom allreduce:42.6/37.7/36.3)。

## 验证与回归风险

PR checklist 中"运行既有 MoE 测试验证无正确性回退"未勾选;作者已实测 GLM-5 TP=8 前后对比,并确认 FLASHINFER_MLA + FLASHINFER_CUTLASS MoE 后端行为不变。风险面:改动的是所有 FusedMoE 共享的 shared-expert 调度判定,EP/EPLB 与 flashinfer two-sided 路径保持 EXTERNAL,DP chunking 明确禁用重叠;正确性依赖 aux stream 重叠路径既有实现。相关 issue:#37113。

## 关联

回归来源:#35153(shared experts 调度重构);关联 issue:#37113。其他同分类编号素材未列出。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
