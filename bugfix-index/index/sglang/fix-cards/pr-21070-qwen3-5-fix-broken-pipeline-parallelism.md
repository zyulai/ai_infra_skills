# PR #21070: [Qwen3.5] Fix broken pipeline parallelism layer splitting

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#21070](https://github.com/sgl-project/sglang/pull/21070) |
| 作者 | alisonshao |
| 合并日期 | 2026-03-21 |
| labels | high priority, run-ci, bypass-maintenance |
| 热度 | 👍 0 · 💬 26 |
| 分类 | Parallelism / Communication |
| bug 类型 | general |

## 一句话结论
Qwen3.5 `make_layers()` 未传 pp_rank/pp_size,所有 PP 阶段都加载全模型致 H100 OOM;传入 PP 参数并跳过本 rank 之外层的权重加载。

## 问题现象
Qwen3.5 开 pipeline parallelism 时,每个 PP stage 都实例化每一层并加载完整模型(每 GPU 约 66GB 而非约 33GB),PP 显存节省为零;在 H200(141GB)上被掩盖,H100(80GB)上 OOM。服务端日志显示两个 PP stage 加载了相同的权重:

```
PP0: Load weight end. avail mem=13.12 GB, mem usage=64.54 GB
PP1: Load weight end. avail mem=13.12 GB, mem usage=64.54 GB
RuntimeError: Not enough memory.
```

修好 `make_layers` 后,权重加载又命中缺失层报 `KeyError: 'model.layers.13.mlp.experts.w2_weight'`。此前 CI 通过只是因为分到了 H200 runner;H100 上失败。

## 根因
正文披露:#19670 为 Qwen3.5 添加 PP 支持时,调用 `make_layers()` 没有传 `pp_rank`/`pp_size`(从非 PP 模型复制粘贴而来),导致所有 PP stage 实例化全部层并加载完整模型;并明确 #20838 只调整了测试阈值,不是本问题的原因。

## 修复方案
- `python/sglang/srt/models/qwen3_5.py`:向 `make_layers()` 传入 `pp_rank=self.pp_group.rank_in_group`、`pp_size=self.pp_group.world_size`(对齐 `qwen2_moe.py` 的可用写法),直接接收返回的 `self._start_layer`/`self._end_layer`,删除原先冗余的 `get_pp_indices` 回退计算;
- `load_fused_expert_weights`(两处)加 guard:`if name not in params_dict: return False`,跳过 PP 范围之外缺失层的参数,避免 KeyError;
- 测试 `test_pp_single_node.py`:baseline 改用 tp=2(完整模型 ~66GB BF16 单张 H100 放不下),`pp=2` 一致性容差由 2% 放宽到 5%。

## 影响与回归风险
- 影响窗口:自 #19670 引入 Qwen3.5 PP 支持起。
- 回归面:Qwen3.5 的层划分与 fused expert 权重加载;`load_fused_expert_weights` 遇到 `params_dict` 中不存在的参数名时由报错改为静默跳过;PP 一致性测试的 baseline 配置与容差同步变化。

## 关联
#19670, #20838
失败示例:
- https://github.com/sgl-project/sglang/actions/runs/23367673750/job/67984876757
- https://github.com/sgl-project/sglang/actions/runs/23372577470/job/67999100214

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
