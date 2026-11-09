# PR #34662: fix: restore VLM nightly regression coverage

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#34662](https://github.com/sgl-project/sglang/pull/34662) |
| 作者 | mickqian |
| 合并日期 | 2026-08-13 |
| labels | Multi-modal |
| 热度 | 👍 0 · 💬 11 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
1024-token CoT 预算使 VLM 夜间旧延迟阈值失效,单节点 CUDA IPC 默认挤占 KV cache 致真实回归;恢复 CPU 默认并重校准阈值。

## 问题现象
VLM 夜间 MMMU 评测的阈值失去回归检测能力:PR #27327 把 MMMU CoT 生成预算从 30 提到 1024 tokens 后,夜间延迟立即落在新的稳定平台,旧阈值测的仍是 30-token 工作负载,不再反映引擎回归。另有真实回归:GLM-4.5V-FP8 在单节点 CUDA IPC 成为默认后,固定 1 GiB feature pool 使 KV cache 从 25,998 降到 14,088 tokens,输出吞吐约 687 → 549 tokens/s,延迟约 107 → 131 秒,最近一次达 148 秒。Pixtral-12b 的分数自该变更后在固定 100 样本上于 0.33–0.37 波动,0.36 的准确率下限会因一两题的正常波动随机失败。

## 根因
正文披露:(1) PR #27327 提高 CoT 预算至 1024 tokens,但 `test_vlms_mmmu_eval.py` 的延迟阈值仍对应旧的 30-token 工作负载,夜间结果进入新平台后旧阈值形同虚设;(2) 单节点 CUDA IPC 被设为默认后,其固定 GPU feature pool 恒占 HBM(空闲时也占用),挤掉了 GLM 夜间的 KV cache 空间,造成吞吐/延迟回归。

## 修复方案
- `server_args.py::_handle_multimodal_feature_transport`:单节点多模态默认传输从 `cuda_ipc` 恢复为 `cpu`,CUDA IPC 改为需显式 `--mm-feature-transport=cuda_ipc`;多节点 GB200/GB300 MNNVL 仍自动选 `cuda_vmm`。同步更新参数帮助文案。
- `test_vlms_mmmu_eval.py`:延迟基线按 1024-token CoT 预算刷新(pixtral 准确率下限 0.360 → 0.320,GLM-4.5V-FP8 延迟阈值 34.0 → 140.0,gemma/NVILA/Mistral-Small 等多数模型延迟阈值放宽);`ModelEvalMetrics` 改为直接用 `(accuracy, latency)` 元组并删除该类。
- `test_server_args.py`:默认传输断言由 cuda_ipc 改为 cpu。

关键 diff(server_args.py):

```diff
                 if self.nnodes == 1:
-                    requested_transport = "cuda_ipc"
-                    logger.info(
-                        "Multimodal feature transport auto-resolved to cuda_ipc "
-                        "(single-node CUDA). Pass --mm-feature-transport=cpu to "
-                        "opt out."
-                    )
+                    requested_transport = "cpu"
```

## 影响与回归风险
- 影响窗口:正文未给版本号(回归始于单节点 CUDA IPC 成为默认的变更之后、本 PR 之前)。
- 回归面:所有单节点 CUDA 多模态部署的 `--mm-feature-transport` 默认值从 cuda_ipc 变回 cpu,多模态特征走 CPU 内存传输(GPU 传输需显式指定);依赖固定池占用的显存预期会变化;VLM 夜间评测通过/失败判定随阈值刷新变化;`test_utils.ModelEvalMetrics` 被删除,外部引用该类会受影响。

## 关联
- https://github.com/sgl-project/sglang/pull/27327

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
