# PR #5707: [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#5707](https://github.com/sgl-project/sglang/pull/5707) |
| 作者 | guoyuhong |
| 合并日期 | 2025-04-24 |
| labels | (none) |
| 热度 | 👍 5 · 💬 5 |
| 分类 | Speculative Decoding |
| bug 类型 | general |

## 一句话结论
MTP 与 --n-share-experts-fusion 同开时权重加载报错或结果全错;为 NextN 层补共享专家克隆并统一 fusion 状态判定。

## 问题现象
正文披露两个 bug,均在 MTP(DeepSeek-R1 NextN 投机解码)与 `--n-share-experts-fusion` 组合使用时出现:
1. 使用 `--n-share-experts-fusion ${TP_SIZE}` 加 MTP 时,出现权重加载错误(正文附截图)。
2. 使用 `--n-share-experts-fusion 0` 加 MTP 时,按不同配置结果可能错误:有时日志显示 `n-share-experts-fusion` 被自动打开,但 `deepseek_nextn.py` 里的状态可能与它不一致,最终推理结果完全错误(正文附日志与结果截图)。

## 根因
正文披露:问题出在 NextN(MTP)模型层与共享专家融合优化状态的配合上——
- 场景 1:基座模型开启 fusion 后 MoE 多出若干"克隆专家"槽位,而 `DeepseekV3ForCausalLMNextN.load_weights` 缺少把共享专家权重克隆到这些槽位的逻辑、专家数也未把 fusion 数量计入,导致权重加载失败;
- 场景 2:fusion 会被自动打开,但 NextN 侧的 `n_share_experts_fusion` 状态与它不一致(正文原话 "the state may not be always constant with it"),两侧专家配置不匹配导致结果错误。

## 修复方案
- `python/sglang/srt/models/deepseek_nextn.py`:`DeepseekV3ForCausalLMNextN.__init__` 中读取 TP size 并调用新函数 `self.determine_n_share_experts_fusion("DeepseekV3ForCausalLMNextN")`;`load_weights` 中当 `n_share_experts_fusion > 0` 时,按量化类型(int8 走 `weight_scale`、否则 `weight_scale_inv`)把 `model.layers.0.mlp.shared_experts` 的 down/gate/up 权重克隆为 `mlp.experts.{n_routed_experts + num_repeat}` 槽位,并从权重流中移除共享专家原条目;MoE 加载的 `num_experts` 改为 `n_routed_experts + self.n_share_experts_fusion`。
- `python/sglang/srt/models/deepseek_v2.py`:把 `n_share_experts_fusion` 的判定逻辑抽取为 `determine_n_share_experts_fusion(architecture="DeepseekV3ForCausalLM")`,架构匹配从写死的 `DeepseekV3ForCausalLM` 改为按传入 architecture 参数比较,供基座与 NextN 两侧共用。

关键 diff(deepseek_nextn.py):
```diff
+            num_experts=self.config.n_routed_experts + self.n_share_experts_fusion,
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交区间。
- 回归面:DeepSeek V3/R1 系模型的权重加载与 MTP 投机解码路径——NextN 层专家数与共享专家克隆逻辑变更影响 `--n-share-experts-fusion` 与 MTP 组合的所有配置;`determine_n_share_experts_fusion` 抽取后,基座 DeepseekV2/V3 的 fusion 自动开启判定行为保持(参数默认值不变),但架构匹配语义改为可传参。克隆逻辑按量化类型分支(int8/fp8),其他量化方式的 suffix 组合需留意。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
