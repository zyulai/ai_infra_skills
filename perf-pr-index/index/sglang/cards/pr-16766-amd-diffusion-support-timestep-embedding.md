# PR #16766: [AMD][Diffusion] support timestep embedding kernel for AMD GPUs

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hubertlu-tw |
| 合并 | 2026-01-13 |
| 分类 | Attention / Kernel(major) |
| Label | documentation, amd, sgl-kernel, run-ci, diffusion |
| Diff | diffstat 数据缺失(pack 未提供 +X/-Y 统计);diff 覆盖 8 个文件(ROCm 扩展注册、kernel 源、HIP 头、setup、模型接线、CI/Docker/文档) |
| 链接 | https://github.com/sgl-project/sglang/pull/16766 |

## 一句话结论

把 sgl-kernel 的 `timestep_embedding` kernel 移植到 AMD GPU 供 FLUX 等扩散模型使用,MI355 上 kernel 平均提速 4.89x,FLUX.1-dev 生成端到端约 +0.87%。

## 动机

sgl-kernel 已有 CUDA 版 `timestep_embedding` kernel(替换 diffusers 的 PyTorch 实现),但 AMD GPU 上不可用:kernel 源里直接用 `__ldg`(HIP 下不可移植)、`hip_math_def.h` 中 `cast` 模板只有声明没有通用定义,ROCm 构建也未把该源文件编入。AMD 上的扩散模型推理只能走慢的 PyTorch 路径。

## 优化点

kernel 本体已存在,本 PR 做的是 HIP 适配与接线:(1) `timestep_embedding.cu` 中 `__ldg` 换成可移植的 `SGLANG_LDG`(CUDA 上映射 `__ldg`,ROCm/HIP 上为普通 load);(2) `hip_math_def.h` 给 `cast` 模板补通用定义(`static_cast` fallback),fp16/bf16 保留显式特化,并把错误的双参写法改为标准 `cast<Src, Dst>` 特化语法;(3) `setup_rocm.py` 把 `csrc/sgl_diffusion/elementwise/timestep_embedding.cu` 加入 ROCm 构建源列表;(4) `common_extension_rocm.cc` 注册 op;(5) `flux.py` 改从 sglang 自己的 `visual_embedding` 导入 `CombinedTimestep*Embeddings`,使快速路径(`sgl_kernel.elementwise.timestep_embedding`)在 AMD 上被实际调用,`sgl_kernel` 不可用时 fallback 到 diffusers 实现;(6) 清理 rocm.Dockerfile 多余 `NO_DEPS_FLAG`,文档补充 `[all_hip,diffusion]` 安装说明,AMD CI 加入对应 pytest。

## 关键改动

```text
sgl-kernel/csrc/sgl_diffusion/elementwise/timestep_embedding.cu: __ldg → SGLANG_LDG 可移植加载
sgl-kernel/include/hip/hip_math_def.h: cast 模板补通用定义与正确特化语法
sgl-kernel/setup_rocm.py + csrc/common_extension_rocm.cc: ROCm 构建与 op 注册
python/sglang/multimodal_gen/runtime/models/dits/flux.py: 切到含 kernel 快速路径的 embeddings
python/sglang/multimodal_gen/runtime/layers/visual_embedding.py: kernel 缺失时 fallback diffusers
```

## 性能声明

- kernel 级微基准(batch ∈ {1,2,8,63,256,512,613,1024,1536} × dim ∈ {32..4096},CUDA event 计时,4 warmup + 20 repeats,sgl_kernel kernel vs diffusers PyTorch 实现):MI355 上平均 speedup 4.89x(详细表以截图形式给出,pack 未含具体逐项数字)。
- E2E:`sglang generate` FLUX.1-dev(black-forest-labs/FLUX.1-dev,单 prompt),4244.044ms → 4206.992ms,约 +0.87%。PR 自述改善幅度小是因为瓶颈在 DenoisingStage(attention/GEMM)而非 timestep embedding。
- 硬件写为 "MI355";PR 未注明 E2E 测试的具体 GPU 型号是否同一块卡。

## 验证与回归风险

`sgl-kernel/tests/sgl_diffusion/test_timestep_embedding.py`:1073 passed(6.50s);AMD CI workflow 加入该测试。回归风险主要在数值等价性:PyTorch/diffusers 路径与 kernel 的 float 计算顺序差异影响极小(仅 embedding 表);`visual_embedding.py` 的 fallback 保证 sgl_kernel 缺失时仍可运行;Dockerfile/安装文档改动影响 AMD 构建流程。CUDA 路径未改动(kernel 源改动对 CUDA 同样生效,SGLANG_LDG 映射回 __ldg)。

## 关联

同区域:sgl-kernel diffusion elementwise kernel 系列(CUDA 版 timestep_embedding 前序);本 PR 为 AMD 移植。同分类其他 diffusion kernel PR 本卡未收录,写"无"。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
