# PR #7410: [Bugfix] Fix weight loading for Chameleon when TP>1

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#7410](https://github.com/vllm-project/vllm/pull/7410) |
| 作者 | DarkLight1337 |
| 合并日期 | 2024-08-13 |
| labels | ready |
| 热度 | 👍 5 · 💬 4 |
| 分类 | Other |
| bug 类型 | startup |

## 一句话结论
Chameleon 在 TP>1 时因 layer norm 权重加载与 None logits 处理不当无法运行,新增 row_parallel_weight_loader 等修复。

## 问题现象
Chameleon(7B 与 30B)在张量并行(TP>1)下无法运行(FIX #7388),表现为 TP 权重加载失败。

## 根因
正文披露的修复点:Chameleon 模型 TP 下的 layer normalization 权重加载有错;此外 TP logits processor 返回的 logits 可能为 None,但类型标注与下游(含 Medusa)未处理。由 diff 推断:Chameleon 自定义 LayerNorm 的权重需按 TP rank 沿分片维 narrow 后加载,此前用的 default_weight_loader 直接整份拷贝导致尺寸不匹配。

## 修复方案
- `weight_utils.py` 新增 `row_parallel_weight_loader`(从 Command-R 的实现抽出,按 `tp_rank * shard_size` narrow 后再走 default loader),Chameleon LayerNorm 的 weight/bias 挂上该 loader;Command-R 改用公共实现;
- `logits_processor.py`、`tensor_model_parallel_gather` 及约 30 个模型文件的 `compute_logits` 标注改为 `Optional[torch.Tensor]`(rank>0 时 gather 返回 None),Chameleon 在 logits 非 None 时才屏蔽 image token,Medusa `compute_logits` 跳过 `_logits=None`;
- 修 `CompletionOutput.token_ids` 类型标注(tuple→数组)、HfRunner 增加 `postprocess_inputs`;新增 Chameleon TP 分布式测试与 vLLM/HF 一致性测试。

注:pack 中 diff 在 40000 字符处截断,以上摘自可见部分,未包含截断部分的改动。

## 影响与回归风险
- 影响窗口:正文未写。
- 回归面:全部模型的 `compute_logits` 返回值语义(rank>0 可为 None)与类型标注;Medusa 投机采样 logits 收集;Command-R layer norm 换用公共 loader;Chameleon VQVAE forward 增加 dtype 转换。

## 关联
[#7388](https://github.com/vllm-project/vllm/issues/7388)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
