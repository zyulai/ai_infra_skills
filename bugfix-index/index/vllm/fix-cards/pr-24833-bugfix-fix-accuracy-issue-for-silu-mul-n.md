# PR #24833: [Bugfix] Fix accuracy issue for silu_mul + nvfp4 quant fusion kernel

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#24833](https://github.com/vllm-project/vllm/pull/24833) |
| 作者 | elvischenv |
| 合并日期 | 2025-09-17 |
| labels | bug, ci/build, ready |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Quantization |
| bug 类型 | correctness |

## 一句话结论
silu_mul+nvfp4 量化融合 kernel 用 tanh 近似算 sigmoid 造成精度下降;改为标准 float32 silu 实现,在线 GSM8K strict-match 恢复至不融合水平。

## 问题现象
开启 silu_mul + nvfp4 quant fusion 时精度轻微下降:离线 lm_eval 未察觉(见 #23671),但在线 lm_eval 有下降。模型 `nvidia/Llama-3.3-70B-Instruct-FP4`,GSM8K(5-shot)在线评测:main 不融合 strict-match 0.6209±0.0134(flexible-extract 0.9257),main 融合 strict-match 降到 0.6035±0.0135(flexible 0.9287)。影响面为启用该 fusion 的编译路径(compute capability ≥ 10 的 nvfp4 设备)。

## 根因
PR 正文披露:融合 kernel 来自 PR #23671,原作者用近似方式计算 silu——sigmoid 用 `tanh(x/2) * 0.5 + 0.5` 计算,而标准定义应为 `1 / (1 + e^(-x))`。近似实现导致融合路径相对不融合路径出现可测量的精度损失。

## 修复方案
重写 `csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu` 中的 silu 计算,改为标准实现(float32 计算,`__fdividef`/`__expf`):

```diff
+// silu in float32
+__device__ __forceinline__ float silu(float x) {
+  return __fdividef(x, (1.f + __expf(-x)));
+}
```

原来在 half/bf16 半精度域内用 `h2tanh` 计算的 `compute_silu` 与专用的 `silu_and_cvt_warp_fp16_to_fp4` 被移除:新的 `compute_silu_mul` 先把输入转 float32 做 silu、再与门控分支相乘并转回半精度,量化部分直接复用 `nvfp4_utils` 已有的 `cvt_warp_fp16_to_fp4`;kernel 改名为 `silu_mul_cvt_fp16_to_fp4`(host 入口仍为 `silu_and_mul_nvfp4_quant_sm1xxa`)。

测试改动:删除旧的 `test_silu_nvfp4_quant_fusion.py`(原阈值允许 2% 失配),新建 `tests/kernels/quantization/test_silu_mul_nvfp4_quant.py` 直接对比 fused op 与 `scaled_fp4_quant` 参考的去量化输出(atol/rtol 3e-1);`tests/compile/test_silu_mul_quant_fusion.py` 参数矩阵从 num_tokens=64/hidden_size=128/fp16 扩到 32/64 x 128/256 x bf16+fp16。结果:24 passed, 8 skipped;kernel 测试 8 passed。

## 影响与回归风险
- 影响窗口:PR #23671 引入 fusion 起到本 PR 之前的 main 分支(在线评测精度下降窗口);正文未给出更多版本细节。
- 回归面:改动限于 nvfp4 silu_mul 量化融合 kernel 及其测试;计算从半精度域改为 float32 域,kernel 耗时 6.527μs → 6.624μs(仍优于不融合的 4.288+4.544μs 两步);未融合路径与其它量化路径不受影响。在线 GSM8K 修复后 flexible 0.9325 / strict 0.6255,均不低于不融合基线。

## 关联
- https://github.com/vllm-project/vllm/pull/23671(引入近似 silu 的原 PR)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
