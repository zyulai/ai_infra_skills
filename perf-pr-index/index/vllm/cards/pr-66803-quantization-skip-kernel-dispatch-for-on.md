# PR #66803: [Quantization] Skip kernel dispatch for online requantization sources

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @fxmarty-amd |
| 合并 | 2026-11-13 |
| 分类 | Quantization(tier: major) |
| Label | ready, quantization |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/66803 |

## 一句话结论

Online requantization 源层不再要求原始 quant method 后端可用:所有 MoE oracle 与 linear kernel selector 对 requantization source 直接返回 placeholder(`NONE` / `Fake*LinearKernel`),跳过 compute backend dispatch,为 #61620 铺路。

## 动机

Online re-quantization 会把 checkpoint 量化层按 checkpoint quantization method 加载、dequantize、再用 online method 重量化,因此原始后端本无必要;但此前该 skip 是遗留 TODO(仅 #64378 为 NVFP4 MoE 专门加过 workaround:让 NVFP4 oracle 对 source 返回 `EMULATION` backend)。本 PR 把该逻辑统一到 linear/MoE oracle,是 #61620 所需的标准化。

## 优化点

1. 所有基于 `MoEKernelOracle` 的 MoE selector(fp8、mxfp8、mxfp4 含 DeepSeek-V4、nvfp4、int8、wna16、w4a8_int8、unquantized)对 requantization source 返回 `(XxBackend.NONE, RequantizationSourceExperts)`,不做 compute backend probing。
2. 所有 `init_*_linear_kernel` selector(fp8、int8、mxfp8、mxfp4、mxfp6、nvfp4、wfp8_a16)对 source 返回 typed `Fake*LinearKernel`(在 weight processing 或 compute 时 raise),selector 返回类型不变。
3. 范围同时覆盖 linear 层与 MoE 以及 source 的 `create_weights`,在 scheme `__init__` 或 `create_weights` 里选的 kernel 也会被跳过。
4. 删除 NVFP4-only 的 `EMULATION` 特例;非 requantization source 的层选 backend 行为与之前完全一致。
5. Source 不再因假设真实 backend 的检查而失败:ModelOpt per-token NVFP4 activation、compressed-tensors MXFP4 MoE with `--moe-backend b12x|deep_gemm|flashinfer EP`。

## 关键改动

```text
vllm/model_executor/kernels/linear/requantization_source.py(新增): _FakeLinearKernel + Fake{Fp8,Int8,MxFp8,MxFp4,MxFp6,NvFp4}LinearKernel placeholder
vllm/model_executor/kernels/linear/base.py: LinearKernelOracle.maybe_skip_for_requantization_source()
vllm/model_executor/kernels/linear/__init__.py: 7 个 init_*_linear_kernel 接入 skip
vllm/model_executor/layers/fused_moe/oracle/base.py: RequantizationSourceExperts + MoEKernelOracle.placeholder_backend / maybe_skip_for_requantization_source
vllm/model_executor/layers/fused_moe/oracle/{fp8,int8,int_wna16,mxfp4,mxfp8,nvfp4,unquantized,w4a8_int8}.py: 各 select_* 开头 skip;nvfp4 删除 EMULATION 特例;各 backend enum 补 NONE
tests/quantization/test_modelopt.py / test_online.py / test_online_shared_expert_fusion.py: 参数化 skip-dispatch 测试
```

(注:pack 的 diff 截断,有 3 个文件未展示,以上仅覆盖已展示部分。)

```python
    @classmethod
    def maybe_skip_for_requantization_source(
        cls,
    ) -> tuple[BackendT, type[mk.FusedMoEExperts]] | None:
        """Skip backend selection for online requantization sources."""
        from vllm.model_executor.layers.quantization.online.requantization import (
            is_requantization_source,
        )

        if not is_requantization_source():
            return None
        return cls.placeholder_backend, RequantizationSourceExperts
```

## 性能声明

- Pack 的 perf claim 正则未检测到任何量化加速数字,PR 正文与 diff 也未提供 benchmark/吞吐数据。
- 正文 Claims 属结构性收益:requantization source 不再 probe compute backend(NVFP4 MoE 之前还要走 `EMULATION` 仿真路径),kernels 在 scheme `__init__`/`create_weights` 阶段即被跳过。

## 验证与回归风险

- 测试环境:AMD Instinct MI350X,HIP 7.15.26333,AMD clang 23.0.0git。
- 定向:`HIP_VISIBLE_DEVICES=0,1,2,3 python -m pytest tests/quantization/test_online.py tests/quantization/test_online_shared_expert_fusion.py -k "skip_dispatch_for_requantization_source or ..."` → 24 passed, 248 deselected, 14 warnings in 16.71s。
- 相关既有测试全量(fused 结果):1 failed, 252 passed, 32 skipped, 1 deselected;唯一失败 `test_online_lut_b_moe` 是 ROCm 不支持该量化(#62759 引入)的 `ModelConfig` 校验,与本 PR 无关。
- 正文注明 quantization 测试自 10/11/2026 起不在 CI 运行(#63608),必须本地跑。
- 回归风险(正文自列):`resolve_quant_method` 的扩大作用域——unquantized checkpoint + online target 时,unquantized MoE method 现在也 skip dispatch;遗留 TODO 清单:`oracle/w4a8.py` 仍 dispatch、`choose_mp_linear_kernel`(GPTQ/AWQ/WNA16/W4A8 linear)仍 dispatch、skip 调用待迁移到 oracle 基类、`--moe-backend`/`--linear-backend` 无 per-layer API、缺真实 checkpoint(原始后端不可用)的 end-to-end 测试。

## 关联

- #55684、#64378(此前 NVFP4 MoE 特例)、#61620(本 PR 为其标准化前置)。
- #62759(`test_online_lut_b_moe` 来源)、#63608(CI 停跑 quantization tests)、#65978(linear oracle 去重 TODO)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
