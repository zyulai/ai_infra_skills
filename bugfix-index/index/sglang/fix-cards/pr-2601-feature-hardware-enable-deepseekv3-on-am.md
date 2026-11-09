# PR #2601: [Feature, Hardware] Enable DeepseekV3 on AMD GPUs

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#2601](https://github.com/sgl-project/sglang/pull/2601) |
| 作者 | BruceXcluding |
| 合并日期 | 2025-01-03 |
| labels | bug, high priority, amd |
| 热度 | 👍 12 · 💬 23 |
| 分类 | Model-specific Fast Path |
| bug 类型 | general |

## 一句话结论
在 AMD MI300X 上启用 DeepSeekV3 FP8:修复 AMD FP8 e4m3fnuz、MLA 中绕过 FlashInfer bmm_fp8、调整 triton stages,并 workaround decode attention 共享内存超限。

## 问题现象
这是一个 enable 类 PR,正文未描述统一的故障现象,而是在 Issues 节列出运行中会遇到的三类错误:1) triton 报 `raise OutOfResources(self.metadata.shared, max_shared, "shared memory")`(与 issue #2384 相同),已通过 `decode_attention.py` 的改动解决;2) `ImportError: cannot import name 'build_regex_from_schema' from 'outlines.fsm.json_schema'`(与 issue #2530 相同),解法是 downgrade vllm;3) `RuntimeError: [enforce fail at .../gloo/transport/tcp/device.cc:83] ifa != nullptr. Unable to find address for: eth0`,解法是检查网卡并 `export GLOO_SOCKET_IFNAME=your eth`。修复后在 MI300X 上验证:GSM8K(2000 questions, 8-shot)Accuracy 0.950、Invalid 0.000;bench_one_batch decode 中位吞吐约 433-435 token/s。

## 根因
正文未披露统一的根因分析,只给出动机与对应解法:对 OutOfResources,正文指出是 MI3xx 共享内存上限问题,解法即 diff 中 `is_hip_ and Lk >= 576` 时把 BLOCK 降到 16 的 workaround;对 FP8 部分,Modifications 说明需要为 AMD FP8 `e4m3fnuz` 格式做适配才能支持 DeepseekV3 FP8 模型,并在 MLA 中绕过 FlashInfer 的 `bmm_fp8`(把 FP8 cast 到 BF16)。

## 修复方案
改动三个文件:

1. `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`:`_decode_grouped_att_m_fwd` 中增加 MI3xx 共享内存限制 workaround:

```python
# [TODO] work around shmem limit on MI3xx
if is_hip_ and Lk >= 576:
    BLOCK = 16
```

2. `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`:`invoke_fused_moe_kernel` 中 `padded_size` 的赋值移入 `block_shape is None` 分支;`fused_experts_impl` 中改为 `if not use_fp8_w8a8 or block_shape is not None: padded_size = 0`;3) 同文件 `get_default_config` 中为 HIP 增加 `triton stages` 配置——`num_stages` 在 `is_hip_flag` 时取 2(CUDA 保持 4 或 3 不变)。

## 影响与回归风险
- 影响窗口:正文未说明,不写。
- 回归面:HIP 的 triton decode attention(Lk >= 576 时 BLOCK 变小,可能影响性能);fused MoE FP8 w8a8 路径的 padded_size 逻辑在 CUDA 与 HIP 上都随 `block_shape` 变化;HIP 上 MoE 默认 config 的 num_stages 降为 2。

## 关联
- https://github.com/sgl-project/sglang/issues/2384
- https://github.com/sgl-project/sglang/issues/2530
- https://github.com/sgl-project/sglang/pull/2692
- https://github.com/sgl-project/sglang/pull/3287
- https://github.com/sgl-project/sglang/pull/4075

(另含 TODO 项中的 PR #3146,因上限 5 个未列出)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
