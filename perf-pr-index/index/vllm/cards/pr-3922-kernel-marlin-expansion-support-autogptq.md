# PR #3922: [Kernel] Marlin Expansion: Support AutoGPTQ Models with Marlin

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat（kernel 工作由 @alexm-nm 完成） |
| 合并 | 2024-04-29 |
| 分类 | Quantization（tier: major） |
| Label | （无） |
| Diff | diffstat 缺失（素材记录为 +None/-None）；diff 展示 3 个文件（ops.h/pybind.cpp/新增 gptq_marlin.cu 约 1520 行），12 个文件因超长被剔除 |
| 链接 | https://github.com/vllm-project/vllm/pull/3922 |

## 一句话结论

扩展 Marlin 4-bit GEMM kernel 支持任意 group size、`desc_act=True`（激活重排序）与 AutoGPTQ 格式自动转换，使 HF hub 上绝大多数 GPTQ 模型免重排直接走 Marlin 快速路径。

## 动机

原 Marlin kernel 只支持序列化在 Marlin 格式下的特定权重布局与少量 group size，HF hub 上的 AutoGPTQ 模型（group size 多样、部分带 `desc_act` 激活重排序）无法直接受益于 Marlin 的高吞吐 fp16×int4 tensor core GEMM，只能退回较慢的旧 GPTQ kernel，或手动序列化成 Marlin 格式。

## 优化点

在独立命名空间 `gptq_marlin` 下新增 GPTQ 变体 kernel（`csrc/quantization/gptq_marlin/gptq_marlin.cu`，基于 IST-DASLab/marlin 改写）：(1) 通过 `g_idx`（每 k 的 group 索引）与 `perm`（列置换）参数支持任意 group size 与 act-order——`has_act_order` 模板分支下在共享内存中取 `g_idx` tile、按 tensor-core fragment 布局逐 K 取对应 scale（`scale4`），`is_same_group` 快路径在 group 内复用 scale；(2) 新增 `permute_cols_kernel` 在 gemm 前对激活 A 的 K 列按 `perm` 重排（act-order 场景），以行块为单位、每线程步进遍历 K；(3) 新增 `gptq_marlin_repack` host 端 op，把 AutoGPTQ checkpoint 权重在线重排成 Marlin 布局，免去离线序列化；(4) 低架构（<sm_80）编译为 stub 并在调用时报 `TORCH_CHECK_NOT_IMPLEMENTED`。新 linear method `GPTQMarlinLinearMethod` 处理 AutoGPTQ 模型自动转换；原 `MarlinLinearMethod` 保留给 Marlin 序列化格式，计划后续 PR 移除（对用户透明）。UX：默认加载 GPTQ 模型即用 marlin kernel，`quantization="gptq"` 可显式回退旧 kernel。

## 关键改动

```text
csrc/quantization/gptq_marlin/gptq_marlin.cu: 新增 ~1520 行，gptq_marlin::Marlin kernel（g_idx/perm 支持、act-order scale4、stripe 分块 + 双缓冲 cp.async 流水）
csrc/quantization/gptq_marlin/（repack 实现）: gptq_marlin_repack（AutoGPTQ→Marlin 布局在线转换）（12 个剔除文件之一，按正文与 ops.h 声明推断存在）
csrc/ops.h / csrc/pybind.cpp: 注册 gptq_marlin_gemm 与 gptq_marlin_repack
vllm/model_executor/layers/quantization/gptq_marlin.py: GPTQMarlinLinearMethod（12 个剔除文件之一，正文指明）
```

act-order 下的逐 K scale 选取：

```cpp
int group_id = sh_g_idx_int_ptr[actual_k];
int rel_group_id = group_id - sh_first_group_id;
*(reinterpret_cast<int4 *>(&(act_frag_s[k % 2][i][0]))) =
    sh_s[rel_group_id * s_sh_stride + s_col_shift];
```

## 性能声明

PR 正文未给出 benchmark 数字（正则未检出性能声明；正文仅声明功能扩展与 UX 变化）。Marlin 相对旧 GPTQ kernel 的收益在本卡中无第一手数据，PR 未注明测试条件。

## 验证与回归风险

diff 中 12 个文件被剔除（含实现与可能的测试文件），素材内不可见测试内容，正文也未列 Test Plan 结果。风险面：(1) `GPTQMarlinLinearMethod` 与旧 `MarlinLinearMethod` 并存，两条 quantization 路径的分派逻辑（默认 marlin / 显式 gptq 回退）若判断失误会影响所有 GPTQ 模型加载；(2) repack 在线转换增加模型加载时间（PR 未注明量化代价）；(3) kernel 要求 sm_80+，pre-Ampere 上显式报 not implemented；(4) act-order 分支共享内存布局复杂（g_idx/scale tile），group 边界与 `sh_max_num_groups=32` 截断逻辑需依赖上游 marlin 的正确性。

## 关联

后续：作者注明将在 follow-up PR 中移除旧 `MarlinLinearMethod`。同分类后续 Marlin 系列优化可见本索引其他卡：#16850（dense marlin 优化）、#12978（block-quantized Marlin StreamK）、#47268（marlin int4-fp8 preprocess 访存修复，后者为 2026 年的衍生路径）。素材未列直接关联编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
