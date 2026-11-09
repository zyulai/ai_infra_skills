# PR #52301: [Perf][Nemotron] Skip redundant latent-MoE all-reduce at TP>1 (~13% decode win)

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ECMGit |
| 合并 | 2026-09-15 |
| 分类 | MoE (tier: major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/52301 |

## 一句话结论

Nemotron-3 latent-MoE 在 TP>1 下每层每 token 发两次 all-reduce,本 PR 通过可交换变换标记省掉一次,恢复 v0.26.0 的 decode ITL(约 13-14% 回归)。

## 动机

Nemotron-3 decode 在 v0.26.0 → v0.27.x 间回归约 14%:v0.27.0rc1 引入的 `_maybe_reduce_routed_output_before_transform()` 在 latent 空间先对 routed output 做 all-reduce,而合并输出在 late path 又被 all-reduce 一次,于是 TP>1/EP>1 时每个 latent-MoE 层每 token 发两次 TP all-reduce 而非一次。`NemotronH` 因把 `fc2_latent_proj` 作为 `routed_output_transform` 而中招。

## 优化点

机制:给 routed output transform 增加可选 `reduce_commutative` 属性,默认 `False`——未声明的模型逐字节保持原行为。当变换与 TP 求和可交换(`sum_r T(x_r) == T(sum_r x_r)`,即无 bias 的线性且未量化)时,对本地 partial 输出先做变换、由既有 late all-reduce 汇总,等价且少一次 collective;`_maybe_reduce_routed_output_before_transform` 据此跳过早 reduce。量化判定基于构造好的 layer(`isinstance(quant_method, UnquantizedLinearMethod)`)而非 model-wide `quant_config`:ModelOpt 量化 checkpoint 排除了 latent 投影,层级判定才能让量化 checkpoint 也吃到该优化(回归最大的场景)。Kimi K3 不受影响——其 `KimiRoutedOutputTransform` 包含 RMSNorm(非线性,不可交换),保持两次 collective。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/runner/moe_runner.py: 早 reduce 前检查 reduce_commutative 属性
vllm/model_executor/models/nemotron_h.py: fc2_latent_proj 按无 bias + UnquantizedLinearMethod 设置标记
```

```python
self.fc2_latent_proj.reduce_commutative = (
    not config.mlp_bias
    and isinstance(self.fc2_latent_proj.quant_method, UnquantizedLinearMethod)
)
```

## 性能声明

4x GB300 (sm_103),tp4,aiperf(输入/输出 1000 token),v0.26.0 vs v0.27.1 vs +PR:
- Super 120B-A12B BF16,c=1 ITL:4.07 → 4.64 ms(+13.8%)→ **4.08 ms**(+0.15% vs v0.26.0);tok/s per user 245.56 → 215.73 → 245.21。
- 并发扫描 c=1/8/32/64 全部恢复到 v0.26.0 水平(如 c=8:6.23/6.63/6.11 ms);收益随 batch 增大而缩小。
- Ultra 550B-A55B NVFP4,c=1:6.86 → 7.21 → **6.79 ms**;同机同容器 v0.27.1→+PR 为 **−5.8% ITL / +6.1% tok/s**;v0.26.0 行为跨节点仅供参考(节点间差异约 8%)。
- 对照:tp1 负对照 −0.3%(无差异);GSM8K 5-shot 全集 1319 题,四组 delta 均在一个标准误内。

## 验证与回归风险

验证依赖上述 aiperf/lm_eval 手工对比,素材 diff 中未新增自动化测试;`reduce_commutative` 契约写在 guard 的 docstring 中。风险面:默认 False,只有显式设置的 `NemotronH` 行为改变;与 `skip_final_all_reduce`/`_fused_output_is_reduced` 的交互已在 PR 中核查;移动 all-reduce 越过线性变换改变浮点求和顺序,bitwise 不恒等(accuracy 数据在误差内)。

## 关联

同区域:#50678(移动 LatentMoERunner 至 kimi_k3 目录,PR 曾考虑复用它);v0.27.0rc1 引入早 reduce 的原改动;Kimi K3 latent-MoE 路径保持不变。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
