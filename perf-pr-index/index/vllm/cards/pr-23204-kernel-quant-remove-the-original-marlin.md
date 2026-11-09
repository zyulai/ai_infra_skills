# PR #23204: [Kernel/Quant] Remove the original marlin format and qqq

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-08-20 |
| 分类 | Quantization |
| Label | performance, ready, ci/build |
| Diff | 素材 diffstat 缺失(pack 记录 +None/-None across None files),diff 以删除为主,20 个文件因过长未展示 |
| 链接 | https://github.com/vllm-project/vllm/pull/23204 |

## 一句话结论

删除已无人使用的原始 marlin 格式 kernel 与 QQQ(w4a8)量化支持,精简内核代码与编译产物。

## 动机

最初的 marlin 集成绑定了 AutoGPTQ 中可选的 "marlin" checkpoint 格式,但绝大多数用户继续产出 gptq 格式 checkpoint;社区因此把 gptq_marlin 路线大力发展并扩展出 awq_marlin、fp8、nvfp4、mxfp4 等变体,原始 marlin kernel/格式已实际弃用。QQQ 是 w4a8 格式,HF 上仅 11 个模型,同样缺乏采用。作者还发现 Machete 一直在为 w4a8 编译 kernel 但没有任何调用方。

## 优化点

这是清理型 PR,不新增优化机制,收益在维护与构建层面:整体删除 `csrc/quantization/marlin/dense/`(含 `marlin_cuda_kernel.cu`、`common/base.h`、`common/mem.h`、LICENSE 等约千行 CUDA 代码)以及原始 marlin 格式的 Python 侧支持;删除 QQQ 量化方法(`marlin_qqq_gemm` 相关)与 Machete 中从未被使用的 w4a8 kernel 生成配置(`generate.py` 里 QQQ_kernel_types/tile heuristic 整段注释掉,标注 "TODO: Support W4A8 when ready");`benchmark_machete.py` 的 QQQ 分支改为 `raise NotImplementedError("QQQ is not supported anymore")`;CI 的 lm-eval-harness 配置移除 `Meta-Llama-3-8B-QQQ.yaml`。保留的 gptq_marlin/awq_marlin 等活跃路径不受影响。

## 关键改动

```text
csrc/quantization/marlin/dense/: 整目录删除(原始 marlin dense kernel 与第三方许可文件)
csrc/quantization/machete/generate.py: 注释掉未使用的 w4a8(QQQ)kernel 生成配置
benchmarks/kernels/benchmark_machete.py: QQQ 基准分支改为 NotImplementedError
.quantization/qqq 等 Python 侧 QQQ/marlin 格式支持移除(20 个文件细节见 PR)
```

## 性能声明

PR 正文与 Test Result 未给出任何性能或精度数字(Test Plan/Test Result 留空)。收益为删除死代码、减少编译与维护负担,不构成运行时性能声明。PR 未注明测试条件。

## 验证与回归风险

- diff 中未见新增测试;依赖现有量化 CI(移除了 QQQ 的 lm-eval 配置项)。
- 风险:对仍在使用原始 "marlin" checkpoint 格式或 QQQ 量化模型的用户是破坏性移除(加载时会报不支持);Machete w4a8 仅是注释而非删除,后续可能恢复。对主流 gptq_marlin/awq_marlin/fp8/nvfp4/mxfp4 路径无行为改动。

## 关联

- 同类清理/演进:#14138(gptq marlin 优化)、#14447(moe wna16 marlin kernel)、#16850(dense marlin 优化)、#20268(Machete zp 支持)。
- 保留路线即 marlin 家族的现代变体(gptq_marlin、awq_marlin、nvfp4/mxfp4 marlin)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
