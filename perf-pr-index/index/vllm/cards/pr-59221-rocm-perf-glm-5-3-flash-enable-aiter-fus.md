# PR #59221: [ROCm][Perf][GLM-5.3-Flash] Enable AITER fused shared experts on gfx950

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jin-amd |
| 合并 | 2026-10-07 |
| 分类 | Speculative Decoding (tier: major) |
| Label | rocm, ready, glm |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);diff 可见 3 个文件(glm5next model、mtp 与测试) |
| 链接 | https://github.com/vllm-project/vllm/pull/59221 |

## 一句话结论

为 GLM-5.3-Flash 在 gfx950 接通 AITER 融合共享专家,TPOT 在 c2-4 降 6-9%、c6-16 最多 3%。

## 动机

PR 正文 Purpose:`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` 此前对 GLM-5.3-Flash(`glm5_next`)被静默忽略——`Glm5NextMoE` 用 `n_shared_experts=None` 构建 `FusedMoE`,42 个 MoE 层仍把 shared expert 当作独立 MLP(两个 FP8 block-scale GEMM、一个 clamped SiLU、一次 quant)再与 routed 输出相加,带来额外 kernel 与显存流量。

## 优化点

按 `DeepseekV2MoE` 的方式接通该 flag:当 `resolve_layer_fused_shared_expert()` 允许时,`Glm5NextMoE` 去掉独立 MLP,把 `n_shared_experts` / `fuse_shared_experts` 传给 `FusedMoE`,让 shared expert 作为 AITER fused MoE 里第 289 个专家(E=289, topk=9)参与;routed 权重保留 `routed_scaling_factor`、shared 槽给 1.0,输出仍是原 shared + routed 之和。加载侧在 target 与 MTP draft 模型中把 checkpoint 的 `mlp.shared_experts.*`(含 FP8 scales)载入专家槽 `n_routed_experts`,`is_model_fused_shared_expert_compatible()` 保证融合 all-or-nothing。门控 `_fused_shared_experts_tuned()` 只在 gfx950 的 TP4/TP8、且无 DP/PCP/EP 时允许融合,其他情况打印一次性 warning 并保持独立 MLP;flag 默认关闭,非 ROCm 平台直接返回 False。

## 关键改动

```text
vllm/models/glm5next/common/model.py: Glm5NextMoE 根据 resolve_layer_fused_shared_expert() 与 _fused_shared_experts_tuned() 决定是否把 shared expert 并入 FusedMoE(E=289, topk=9)
vllm/models/glm5next/common/mtp.py: MTP draft 模型同样把 shared_experts 载入 fused 专家槽(否则 MTP+fusion 加载报 KeyError)
tests/model_executor/layers/test_fused_shared_expert.py: 覆盖 glm5_next 初始化、gate、loader 槽位的 95 个单测
```

关键机制代码:

```python
        self.is_fused_shared_expert_enabled = False
        if config.n_shared_experts is not None:
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config, prefix
            ) and _fused_shared_experts_tuned(parallel_config)
        if config.n_shared_experts is None or self.is_fused_shared_expert_enabled:
            self.shared_experts = None
...
            n_shared_experts=config.n_shared_experts
            if self.is_fused_shared_expert_enabled
            else None,
            fuse_shared_experts=self.is_fused_shared_expert_enabled,
```

## 性能声明

PR 正文 Test Result(gfx950 / MI350X,`zai-org/GLM-5.3-Flash` FP8,TP4,ROCm nightly + AITER v0.1.23,三次背靠背 sweep on/off/on):flag on 时 TPOT 在 concurrency 2-4 降 6-9%、6-16 最多 3%。Perf(TP4)表:c2 output tok/s 101.8 → 106.5(+4.6%)、TPOT 13.06 → 12.28 ms(−6.0%);c4 133.2 → 142.3(+6.8%)、19.15 → 17.43(−9.0%);c8 160.8 → 165.8(+3.1%)、31.32 → 30.71(−2.0%);c16 180.8 → 185.7(+2.7%)、49.30 → 47.80(−3.0%)。TTFT 在 3.5% 噪声内不变。gsm8k strict/flexible:off 0.9757/0.9750,on 两次为 0.9712/0.9704 与 0.9735/0.9719,±0.0045 stderr 内视为不变;MTP(k=5)draft acceptance 63.66% → 63.56%、mean acceptance length 4.18 → 4.18。gfx942(4× MI325X)上 flag 仅告警、性能与精度不变。

## 验证与回归风险

Test Plan:`vllm serve ... VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`(TP4/MTP)、各并发 `vllm bench serve` sweep、lm_eval gsm8k、`pytest tests/model_executor/layers/test_fused_shared_expert.py`;正文报告 95 个单测全过,并用 `AITER_LOG_TUNED_CONFIG=1` 确认 on 时 fused-MoE lookup 为 (E=289, topk=9)、off 为 (E=288, topk=8)。影响路径:仅 gfx950 TP4/TP8 且关闭 DP/PCP/EP 时生效;依赖 ROCm/aiter#5735 的 tuned configs(AITER v0.1.23,由 #58867 引入镜像),gfx942 因缺 tuned rows 暂不开放(正文称 untuned 时一次 16K-token MoE 调用 4289 µs vs 3033 µs)。

## 关联

外部依赖 ROCm/aiter#5735(gfx950 TP4/TP8 的 tuned configs,由 #58867 引入镜像);后续 gfx942 支持待 ROCm/aiter#5953;正文以 DeepSeek-V4 的 shared-expert gate 为同类门控参考。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
