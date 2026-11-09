# PR #14676: Revert "[Bug] fix not desired disable fused share experts caused by r…

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#14676](https://github.com/sgl-project/sglang/pull/14676) |
| 作者 | zhyncs |
| 合并日期 | 2025-12-09 |
| labels | deepseek, high priority, run-ci |
| 热度 | 👍 0 · 💬 3 |
| 分类 | MoE |
| bug 类型 | regression |

## 一句话结论
回滚 #14432:其 ROCm 逻辑改动使 NVIDIA+EP 下 shared experts fusion 未被禁用,CUTLASS MoE 精度归零(GSM8K 0.000→0.950)。

## 问题现象
`nvidia/DeepSeek-V3-0324-NVFP4` + `--tp 4 --ep 4 --attention-backend trtllm_mla --moe-runner-backend flashinfer_cutlass --quantization modelopt_fp4` 下 GSM8K(8-shot,1319q,parallel 1319)精度为 0.000(Invalid 0.000,Latency 51.032s,吞吐 13233.487 token/s);回滚后恢复 Accuracy 0.950、Latency 24.641s。正文并指出近一周 OSS 稳定性明显下滑的背景:B200 等许多 CI 任务未跑完即合并,#14432 引入 CUTLASS MoE accuracy 0,#13761 曾引入 RMSNorm failure(经 #14135 修复),#14213 引入 `ValueError: too many values to unpack (expected 2)`。

## 根因
正文披露:#14432 引入了 CUTLASS MoE accuracy 0(本 PR 即其修复)。(由 diff 推断)机制:#14432 把 `deepseek_v2.py` `determine_num_fused_shared_experts` 中「EP>1 时禁用 shared experts fusion」的判断从 `EP>1 and (not _is_hip or capability < (9,4))` 改成了 `EP>1 and _is_hip and capability < (9,4)`,在 NVIDIA 平台开启 EP 时不再进入禁用分支,fused shared experts 意外保持启用,与 flashinfer_cutlass MoE 路径组合后输出错误、精度归零。

## 修复方案
原样回滚 commit `2ecee7571cdfe10a124ae313fe020166ad5c56b8`(#14432),改动在 `python/sglang/srt/models/deepseek_v2.py` 的 `determine_num_fused_shared_experts`,恢复禁用条件:
```diff
-        elif (
-            get_moe_expert_parallel_world_size() > 1
-            and _is_hip
-            and torch.cuda.get_device_capability("cuda") < (9, 4)
+        elif get_moe_expert_parallel_world_size() > 1 and (
+            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
         ):
```
即 NVIDIA 平台(非 HIP)开启 EP 时同样禁用 shared experts fusion,AMD 仅在 capability < gfx942(MI30x)时禁用。

## 影响与回归风险
- 影响窗口:正文未说明;按引入关系,#14432 合并之后、本 PR 之前。
- 回归面:`deepseek_v2.py` 的 shared experts fusion 启用决策,影响 DeepSeek V3/R1 系模型在 EP 下的 MoE 路径;#14432 原本想在 AMD ROCm 下避免「不必要的禁用」,回滚后该诉求回到旧状态(AMD EP 场景恢复旧行为);NVIDIA + EP + flashinfer_cutlass/modelopt_fp4 路径恢复正确精度(正文 benchmark 验证)。

## 关联
- https://github.com/sgl-project/sglang/pull/14432 (被回滚的 PR)
- https://github.com/sgl-project/sglang/pull/13761 、https://github.com/sgl-project/sglang/pull/14135 、https://github.com/sgl-project/sglang/pull/14213 (正文列举的同期稳定性问题)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
