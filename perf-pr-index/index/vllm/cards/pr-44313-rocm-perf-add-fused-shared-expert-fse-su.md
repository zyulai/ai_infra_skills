# PR #44313: [ROCm][Perf] Add Fused Shared Expert (FSE) support for GLM-4.5/6/7

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @omirosh |
| 合并 | 2026-06-28 |
| 分类 | Speculative Decoding(major,classify 结果,实为 MoE shared-expert 融合) |
| Label | rocm, ready |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 glm4_moe.py 与 glm4_moe_mtp.py 两个文件改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/44313 |

## 一句话结论

ROCm AITER 上把 GLM-4.5/6/7 的 shared expert 折叠进 FusedMoE kernel,低并发下输出吞吐 +20–22%、TPOT −18–19%,gsm8k 无回归。

## 动机

AITER 的 Fused Shared Expert 路径此前已覆盖 DeepSeek-V2/V3(#28540)与 Qwen3-Next(#39280),但 GLM-4 MoE 家族仍保留独立的 shared-expert MLP forward:每个 MoE 层要在 routed expert kernel 之外单独跑一遍 shared expert,低/中并发下这次独立前向构成可观的延迟开销。GLM 系列还特有一个坑:checkpoint 把所有 shared expert 打包成单个 fat MLP,且 GLM-4.7 的 `routed_scaling_factor=2.5` 在 FSE 路径下若二次缩放会引发结构性精度错误(首版 PR 曾出现 gsm8k 0.4685 对 0.9469 的 ~48pp 回归)。

## 优化点

`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 时:FSE 逻辑上不新建独立 `shared_experts` MLP,而是把 shared expert 作为 `n_shared_experts` 个额外 expert slot 并入 AITER `FusedMoE` kernel,一次调用完成 routed + shared 计算。权重加载上,把 checkpoint 中打包的 `mlp.shared_experts.{gate,up,down}_proj` 沿中间维度切成 `n_shared_experts` 份(Column-Parallel 走 dim 0,RowParallel 的 down_proj 走 dim 1),以 `mlp.experts.{n_routed_experts + j}` 的专家名路由进 FusedMoE 的 expert-aware loader;`get_expert_mapping` 同步扩容。MTP draft 模块(`glm4_moe_mtp.py`)有独立的 weight loader,做同样的 FSE-aware 切分。同时将 `apply_routed_scale_to_output` 改为 `not is_rocm_aiter_moe_enabled`,让 AITER 仅对 routed slot 内部施加 scaling,避免连带放大按单位权重插入的 shared-expert slot。

## 关键改动

```text
vllm/model_executor/models/glm4_moe.py: FSE 开关下跳过独立 shared_experts 构建、向 FusedMoE 传 n_shared_experts、修正 routed scaling、权重切分加载
vllm/model_executor/models/glm4_moe_mtp.py: MTP 层同样扩容 expert mapping 并切分加载 shared-expert 权重
```

关键片段(glm4_moe.py):

```python
if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
    self.shared_experts = None
else:
    ...
self.experts = FusedMoE(
    ...
    apply_routed_scale_to_output=not self.is_rocm_aiter_moe_enabled,
    n_shared_experts=(config.n_shared_experts
        if self.is_fusion_moe_shared_experts_enabled else None),
)
```

## 性能声明

测试条件:zai-org/GLM-4.7-FP8,1× MI355X 节点 4 GPU,TP=4 + EP,AITER ≥ v0.1.13.post1,fp8 KV cache,(ISL/OSL) ∈ {1000/100, 5000/500, 10000/1000} × MC ∈ {4,16,64} 随机数据集 sweep。
- 无 MTP:MC=4 时 output tok/s +22.1%(199.4→243.6)、TPOT mean −19.2%(17.76→14.36 ms);MC=16 温和收益(+4.3%~+6.7%);MC=64 基本持平(−2% 左右)。
- 开 MTP(num_speculative_tokens=2):收益反转到高并发 —— MC=64 时 +8%~+17% 输出吞吐(MC=64 @10000/1000 达 +17.0%),MC=4 时 −5%~−11%;draft acceptance rate 27.1%→32.9%,mean acceptance length 1.54→1.66。
- gsm8k(5-shot、256 样本):FSE=0/1 分别 0.9469/0.9439(flex),在 1σ 内无回归;FSE-on MTP 0.9530,与非 MTP 基线相当。

## 验证与回归风险

验证含 gsm8k 精度对照、9 格吞吐 sweep(开/不开 MTP 各一轮)、MTP acceptance 统计。风险面:仅 ROCm AITER 路径,默认环境变量下行为不变;`apply_routed_scale_to_output` 的语义依赖 AITER 内部按 routed slot 施加 scaling 的实现,换 kernel/版本需复核;权重切分要求 checkpoint 维度可被 `n_shared_experts` 整除,否则显式报错;EP 下依赖 `return_success=True` 避免把远端 expert 副本误记为已加载。

## 关联

前序:#28540(DeepSeek-V2/V3 FSE,模式蓝本)、#39280(Qwen3-Next FSE 与 AITER op plumbing,本 PR 无 kernel 改动)。同区域:#61457(per-layer fused shared expert 量化扩展)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
