# PR #27202: [NPU][Bugfix] fix MTP accuracy regression on Qwen3 hybrid models

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#27202](https://github.com/sgl-project/sglang/pull/27202) |
| 作者 | AndyLi429 |
| 合并日期 | 2026-06-09 |
| labels | npu, run-ci |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Speculative Decoding |
| bug 类型 | correctness |

## 一句话结论
NPU 上 Qwen3 hybrid 模型开 MTP 精度回退且思考模式循环输出;三个独立 bug(SSM 索引混叠、mamba 状态跟踪缺失、top-k 快速路径)一并修复。

## 问题现象
在 Ascend NPU 上运行 Qwen3.6-35B-A3(w4a8,启用 MTP)并用 `--mamba-scheduler-strategy extra_buffer` 时,ceval 准确率低于预期基线;开启 thinking mode 后更糟——精度进一步下降且模型产生重复/循环的推理内容。正文定位出 NPU hybrid-linear attention graph-mode verify 路径、verify 后 mamba 状态跟踪、EAGLE 投机 top-k 选择三处 bug。

## 根因
正文逐条披露:
1. **SSM 状态索引混叠**:graph-mode target verify 时 `ssm_state_indices` 只按 `(bs - num_padding) * draft_token_num` 构造、padding 尾部填 0,导致每个 padded 行的 verify 写入都混叠(alias)到真实请求 0 的中间 scratch 槽;verify 后 `move_intermediate_cache` 状态恢复读到损坏数据,产生乱码输出或 token 重复。
2. **MTP verify 后 mamba 状态跟踪缺失**:NPU 路径未实现 CUDA 路径已有的 `mamba_track_indices` / `mamba_steps_to_track` scatter;mamba prefix reuse(长 thinking 序列会触发)时恢复到**陈旧**的 SSM/conv 状态,正是重复推理内容与 thinking 模式精度下降的来源。
3. **NPU argmax 快速路径偏差**:`topk == 1` 的 argmax 快速路径此前仅用 `not _is_hip` 排除 ROCm(#26358 引入);NPU 的 argmax tie-break 在某些 logit 分布下与完整 `softmax + topk` 路径不一致,破坏 MTP draft 选择。

## 修复方案
- `ascend_hybrid_linear_attn_backend.py` `_replay_metadata`:索引改为按整个 graph bucket 构造并整体 `copy_`,每个 padded 行拥有独立 scratch 槽:
```diff
-                len(mamba_indices[: bs - num_padding]) * spec_info.draft_token_num,
+                bs * spec_info.draft_token_num,
```
- 同文件 `update_mamba_state_after_mtp_verify`:补上 `mamba_track_indices` 分支——经 `move_intermediate_cache` 把 verify 时 SSM 中间状态 scatter 进持久 track 槽;conv 状态因 NPU 无 per-step `intermediate_conv_window` 缓存,先在工作槽回滚前拷贝到 track 槽,再对工作槽与 track 槽各自独立 `conv_state_rollback`。
- eagle 文件的 `not _is_npu` 守卫(正文描述三处调用点;该部分改动未包含在本 pack 的 diff 摘录中,NPU draft-extend graph runner 继承 CUDA runner 的 `run_once` 故守卫生效)。
- diff 另含 `memory_pool_npu.py` 新增 `_init_kv_copy_and_warmup` 将 `_kv_copy_config` 置 None(正文未解释;注释称其实现依赖 NPU paged buffer 布局从未构建的 `data_strides`/`data_ptrs`)。
- 验证:ceval 1346 样本(正文表格标注 Qwen3.5-397B-A17B-w4a8-mtp,启动命令模型路径为 Qwen3.6-35B-A3B):thinking off 87.5%→89.75%,thinking on ~86%→90.71%,重复推理回归消失。

## 影响与回归风险
- 影响窗口:正文未说明。
- 回归面:仅 NPU 后端(ascend hybrid-linear attention 后端、NPU memory pool)与 EAGLE top-k 守卫;CUDA 路径不受影响(其 mamba 跟踪为既有实现)。风险在 NPU conv 状态重建逻辑依赖 track 槽与工作槽不相交及 `conv_state_rollback` 的按步窗口还原。

## 关联
- [#26358](https://github.com/sgl-project/sglang/pull/26358)(ROCm argmax 守卫的来源)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
