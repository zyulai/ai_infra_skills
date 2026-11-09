# PR #24635: Fix stuck when enabling MTP on DSA models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#24635](https://github.com/sgl-project/sglang/pull/24635) |
| 作者 | Fridge003 |
| 合并日期 | 2026-05-08 |
| labels | high priority, deepseek |
| 热度 | 👍 0 · 💬 7 |
| 分类 | Speculative Decoding |
| bug 类型 | hang |

## 一句话结论
DSA 模型开 MTP 后卡死;修 NSA 后端 draft-extend-v2 判定与 2D seqlens 归一化,并在 CUDA Graph replay 时刷新 DeepGEMM 调度元数据。

## 问题现象
在 DSA(DeepSeek Sparse Attention)模型上启用 MTP(multi-token prediction)后服务卡死(stuck/hang)。正文仅写 "close #24571",未展开现象细节;按 diff 恢复的三个测试(`test_dsa_models_mtp.py`、`test_deepseek_v32_cp_single_node.py`、`test_deepseek_v32_fp4_mtp_4gpu.py`)此前均标注 "Disabled due to #24268. Should be fixed soon.",说明相关 8-gpu/cp/quant 测试因该问题被禁用。

## 根因
正文未披露根因(仅 close #24571)。

机制(由 diff 推断,依据 diff 内注释与改动):
1. diff 注释明确指出:"Refresh DeepGEMM paged MQA schedule metadata for the actual seqlens of this replay (the captured graph holds stale data otherwise, which can deadlock the kernel when the runtime work decomposition diverges from the captured one)"——CUDA Graph replay 时 `paged_mqa_schedule_metadata` 沿用捕获时的旧数据,运行期 work 分解与捕获时不一致时 `deep_gemm.fp8_paged_mqa_logits` 内核死锁。
2. `_to_2d_context_lens` 的旧实现允许返回 (bs, next_n) 布局,新增注释 "Always normalize to (N_total, 1) layout, to avoid deadlock at deep_gemm.fp8_paged_mqa_logits"。
3. 多处 `is_draft_extend()` 判定漏掉 v2 模式,改为 `is_draft_extend(include_v2=True)`,即 MTP 使用的 draft-extend-v2 路径此前未按该分支初始化 metadata。

## 修复方案
改动集中在 `python/sglang/srt/layers/attention/nsa_backend.py`:

1. `init_forward_metadata_replay_cuda_graph_from_precomputed` 新增一段:按本次 replay 的实际 seqlens 重建 `deep_gemm.get_paged_mqa_logits_metadata(...)` 并 `copy_` 进 `metadata.paged_mqa_schedule_metadata`;设置处改用 `object.__setattr__`(metadata 属性不可直接赋值,replay 中的 `metadata.paged_mqa_schedule_metadata = ...` 原写法同样替换)。
2. `_to_2d_context_lens` 统一归一化为 (N_total, 1):输入为 2D 且第二维非 1 时先 `reshape(-1)` 再 `view(-1, 1)`。
3. `init_forward_metadata` / `init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph` 四处 `is_draft_extend()` 全部改为 `is_draft_extend(include_v2=True)`。
4. 三个测试文件移除 `disabled="Disabled due to #24268..."` 重新启用。

## 影响与回归风险
- 影响窗口:正文未说明具体受影响版本/提交范围。
- 回归面:NSA 后端的 CUDA Graph capture/replay 与 draft-extend 路径,均属 DSA + MTP(deepseek V3.2 系列)场景;非 NSA 后端与其他模型不受影响。replay 时按实际 seqlens 重建调度元数据会带来少量额外计算;`_to_2d_context_lens` 对 (bs, next_n) 输入改为重排,若下游还有依赖该布局的调用需核对(此前已存在 assert 检查)。

## 关联
- [#24571](https://github.com/sgl-project/sglang/issues/24571)(正文提到,本 PR close)
- (#24268 仅出现在 diff 的测试 disabled 标注中,正文未提)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
