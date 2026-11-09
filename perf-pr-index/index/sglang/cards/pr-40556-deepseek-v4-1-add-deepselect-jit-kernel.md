# PR #40556: [DeepSeek V4.1] Add DeepSelect JIT kernel.

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @yuyu5333 |
| 合并 | 2026-09-26 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | documentation, high priority, sgl-kernel, mthreads, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计);diff 可用但超长截断(21 个文件未展示) |
| 链接 | https://github.com/sgl-project/sglang/pull/40556 |

## 一句话结论

把 DeepSeek 的 DeepSelect top-k 内核从 AOT 预编译迁入 SGLang JIT 框架,只为本地 CUDA 目标编译所需签名,候选路径整体较 PyTorch 快 1.91x–3.77x。

## 动机

对应 #39421。此前的 DeepSelect 以 AOT 方式集成,需要 86 个显式实例化 `.cu` 文件把所有 (dtype, topk, vocab, batch, arch) 配置编译进同一个共享库,构建负担重且无法按需裁剪。本 PR 把它迁到 SGLang 现有 JIT 基础设施,只编译本地目标实际需要的调用签名。

## 优化点

CUDA 算法本身不变:来自 deepseek-ai/DeepSelect(commit `382d62a` 加上游 PR #14 的 SM90 cluster 变体)的三个 kernel 头文件(bf16 单 CTA/行、bf16 cluster/行——SM90 用 8 CTA、SM100/SM103 用 16 CTA、fp32)仅删除对已删 AOT 头文件的 include。机制变化在于分发层:原来由运行时 `BOOL_SWITCH`/`INTEGER_TYPE_SWITCH` 宏展开的全部布尔与索引类型轴(value dtype、index dtype、`sorted_value`、`sorted_index`、`return_value`、topk bucket、cluster size)改为 Python module factory 挑选的模板参数,C++ 端只剩"该 bucket 的两种 tuning 选哪个"一个决策。JIT 入口新增 C++ 参数/存储校验(TensorMatcher 检查 shape/stride/dtype/对齐)、对齐输出 staging 与空 batch 处理;直接使用 JIT 框架的本地目标,不加 `ScopedCudaDevice`、运行时 SM 检查或 `override_jit_cuda_arch`。kernel 需要 `--expt-extended-lambda` 与 `--ftz=false`(radix pass 用 fp32 加法模拟整数计数器,需要 subnormals)。

## 关键改动

```text
python/sglang/kernels/jit/csrc/deepselect/vendor/: vendored DeepSelect 内核(去 AOT 实例化,MIT)
python/sglang/kernels/jit/csrc/deepselect/entry.cuh: JIT host 入口,make_plan 校验 + TopkNormal/TopkCluster 派发
python/sglang/kernels/ops/deep_select.py: 编译并调用 JIT 入口的 Python wrapper
```

```cpp
+    if constexpr (std::is_same_v<typename Config::ValueT, float>) {
+      topk_select_fp32::run_topk_select_kernel<Config>(args);
+    } else {
+      topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
+    }
```

## 性能声明

NVIDIA H20,FP32 candidate logits,`topk_blocks=2048`,20 次预热 + 100 次测量取中位数,每次测量前 flush L2,两次完整运行报告中位数均值。测量区域含候选块分数归约 → 候选 Top-K → mask 发布 → 下游 paged Top-K 全链路:

- Final Top-K 512:1 行 128K 153.264→44.112 us(3.47x);6 行 128K 166.032→44.064 us(3.77x,最佳);6 行 1M 277.936→104.672 us(2.66x)。
- Final Top-K 2048:1 行 1M 187.600→98.096 us(1.91x,最差);其余 2.39x–3.42x。
- 16 个 shape 汇总:speedup 1.91x–3.77x,几何平均 2.79x。
- 与外部 DeepSelect 实现在 width 131072、Top-K 2048 直接对比 AOT kernel 延迟差约 0.3%–0.4%,迁入 sgl-kernel 无 kernel 级性能回归。

注意:表中对比对象为 PyTorch 参考实现与融合 kernel,PyTorch 路径含完整候选链路;PR 未注明测试用模型。

## 验证与回归风险

- PR 正文"Accuracy Tests"一节留空,未提供精度测试结果;checklist 各项未勾选。
- 回归风险:算法不变,风险主要在 JIT 构建环境(CUTLASS 依赖、`--ftz=false` 必须跟在 `--use_fast_math` 后)与 SM90/SM100/SM103 cluster 路径的覆盖;cluster size 选择(8 vs 16 CTA)由 Python 端决定。CI 三条(Base/Extra/AMD ROCm 10)显示失败。

## 关联

- 前序/动机:#39421。
- 同区域:DeepSeek V4.x attention 稀疏选择相关 kernel 系列;#46263(attention 后端选择)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
