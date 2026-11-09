# PR #4002: [Kernel] Fused MoE Config for Mixtral 8x22

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ywang96 |
| 合并 | 2024-04-11 |
| 分类 | MoE（tier: major） |
| Label | (none) |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾未提供「N file(s) dropped from diff」说明 |
| 链接 | https://github.com/vllm-project/vllm/pull/4002 |

## 一句话结论

为 Mixtral 8x22 新增 A100/H100、TP4/8 的 fused MoE config，高 batch size 下 latency 约降 10%。

## 动机

本 PR 为 Mixtral 8x22 在 A100-80G 与 H100、TP4 与 TP8 组合新增 fused MoE configs。PR 标注高 batch size 下约 10% latency speedup，并附上 A100-80G 上运行 `benchmark_latency.py` 的 mean / median 结果作为 sanity check。

## 优化点

新增 4 份 tuned fused MoE Triton config JSON（`E=8`，`N=2048` 与 `N=4096`，设备 `NVIDIA_A100-SXM4-80GB` 与 `NVIDIA_H100_80GB_HBM3`），按 batch size 键（1、2、4、8、16、24、32、48、64、96、128、256、512、1024、1536 等）给出 `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`。改动是配置数据补充，无 kernel 代码变更。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_A100-SXM4-80GB.json: 新增 A100 / N=2048 的 tuned fused MoE 配置
vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H100_80GB_HBM3.json: 新增 H100 / N=2048 的 tuned 配置
vllm/model_executor/layers/fused_moe/configs/E=8,N=4096,device_name=NVIDIA_A100-SXM4-80GB.json: 新增 A100 / N=4096 的 tuned 配置
vllm/model_executor/layers/fused_moe/configs/E=8,N=4096,device_name=NVIDIA_H100_80GB_HBM3.json: 新增 H100 / N=4096 的 tuned 配置
```

## 性能声明

测试条件：Mixtral 8x22，A100-80G 与 H100，TP4 与 TP8，`benchmark_latency.py`，batch size 8 / 16 / 32；PR 标注高 batch size 下约 10% latency speedup，附的 mean / median 数据来自 A100-80G。正文给出的 A100-80G 结果（Avg latency / 50% percentile latency，单位秒，无 config → with config）：TP4 bs8 5.787593524421876 / 5.785140061285347 → 5.4268179705676935 / 5.4153253806289285；bs16 6.900521802188208 / 6.910593919456005 → 6.107941208345195 / 6.10570535203442；bs32 7.437153746870657 / 7.425246893661097 → 6.893939550562451 / 6.877172949025407。TP8 bs8 3.83942656442523 / 3.8410149838309735 → 3.7083894722008455 / 3.7043474409729242；bs16 4.428924084718649 / 4.421724954620004 → 4.047313968433688 / 4.0400408857967705；bs32 4.767531445249915 / 4.752460908377543 → 4.390694409639885 / 4.380223051644862。正文未给出 H100 的具体延迟数字。

## 验证与回归风险

PR 正文未列出专门单测；验证依据为 `benchmark_latency.py` 在 A100-80G 上的 mean / median 结果（作为 sanity check）。正文说明配置覆盖 A100-80G 与 H100、TP4 与 TP8，但所附的 mean / median 数据仅来自 A100-80G，未附 H100 的延迟数字。回归风险：改动仅为新增 config JSON 数据（按 `E` / `N` / `device_name` 命中的 tuned 参数），影响 Mixtral 8x22 在 A100-80G / H100、TP4 / TP8 下的 fused MoE kernel 选型；正文未列出其他受影响模型、精度或已知回归。

## 关联

无（PR 正文未给出其他 PR 编号线索）。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
