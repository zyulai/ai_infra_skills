# PR #15777: [Sampler] Adapt to FlashInfer 0.2.3 sampler API

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @abmfy |
| 合并 | 2025-05-16 |
| 分类 | Attention / Kernel \| tier: major |
| Label | ready, ci/build, v1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files) |
| 链接 | https://github.com/vllm-project/vllm/pull/15777 |

## 一句话结论

适配 FlashInfer 0.2.3 sampler API 破坏性变更，简化 top-k/top-p 采样调用并移除手写 uniform_samples 路径。

## 动机

PR 正文说明 FlashInfer 0.2.3 对其 sampler API 引入了破坏性变更（breaking changes），本 PR 更新 vLLM 中的调用点以适配该更新，并标注 FIX #14815、FIX #15666。正文另指向 FlashInfer 0.2.3 的 release 说明，并说明适配范围仅涉及 vLLM 内的 sampler 调用点；除此之外未给出更多背景、改动前瓶颈或性能动机描述。

## 优化点

从 diff 看：docker/Dockerfile 更新 FlashInfer 版本；vllm/model_executor/layers/sampler.py 移除 `import flashinfer.sampling` 与 V0 路径里手写的 max_top_k_round=32、uniform_samples 生成逻辑，把 flashinfer_top_k_top_p_sampling 改为只接收 (probs, top_ks, top_ps) 并去掉 success/失败回退（top_k_renorm_prob、top_p_renorm_prob、sampling_from_probs）分支；因 0.2.3+ 不支持 per-request generators，V0 采样在 flashinfer 分支打 warning 后统一走 PyTorch-native 的 _multinomial。vllm/v1/sample/ops/topk_topp_sampler.py、vllm/model_executor/layers/rejection_sampler.py 及 tests/samplers、tests/v1/sample 下的测试同步适配新 API。

## 关键改动

```text
vllm/model_executor/layers/sampler.py: 简化 top-k/top-p 调用、移除 uniform_samples 与 rejection fallback
vllm/v1/sample/ops/topk_topp_sampler.py: V1 采样适配新 API
vllm/model_executor/layers/rejection_sampler.py: 拒绝采样适配新 API
docker/Dockerfile: 更新 FlashInfer 版本
tests/samplers/test_sampler.py、test_rejection_sampler.py、tests/v1/sample/test_topk_topp_sampler.py: 测试适配
```

```python
            if flashinfer_top_k_top_p_sampling is not None:
                logger.warning("FlashInfer 0.2.3+ does not support "
                               "per-request generators. Falling back to "
                               "PyTorch-native implementation.")

            multinomial_samples[sampling_type] = _multinomial(
                probs[long_sample_indices],
                max_n_in_batch,
                seq_groups=seq_groups_arg)
```

## 性能声明

PR 正文未给出定量性能数据（pack 内 perf claims 标注 none detected by regex）；本 PR 为 FlashInfer sampler API 兼容性适配，正文未声明性能收益。PR 未注明测试条件。

## 验证与回归风险

PR 正文未列出具体测试命令或结果；diff 修改了 tests/samplers/test_sampler.py、test_rejection_sampler.py 与 tests/v1/sample/test_topk_topp_sampler.py 等测试以适配新 API。回归风险：FlashInfer 0.2.3+ 不支持 per-request generators，V0 sampler 路径回退到 PyTorch-native _multinomial 并打 warning；移除了 rejection sampling 失败时的 fallback 逻辑；涉及 V0/V1 sampler、rejection sampler 与 Docker 镜像中的 FlashInfer 版本；采样正确性依赖新 API 的返回语义。

## 关联

正文标注 FIX #14815、FIX #15666，并指向 FlashInfer 0.2.3 release 说明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
