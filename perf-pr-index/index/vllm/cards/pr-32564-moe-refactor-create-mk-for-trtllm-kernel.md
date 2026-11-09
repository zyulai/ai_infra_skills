# PR #32564: [MoE Refactor] Create MK for TRTLLM Kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @robertgshaw2-redhat |
| 合并 | 2026-03-03 |
| 分类 | MoE(tier: major) |
| Label | documentation, performance, rocm, intel-gpu, ci/build, llama, gpt-oss, nvidia, ready-run-all-tests |
| Diff | diffstat 缺失(素材未提供 additions/deletions 与文件数);diff 另有 72 个文件因超长被略去(核心 TRTLLM MK 实现在其中,素材仅可见 docs/benchmarks/CI 部分) |
| 链接 | https://github.com/vllm-project/vllm/pull/32564 |

## 一句话结论

把 TRTLLM MoE kernel 接入 modular kernel(MK)框架,并引入"monolithic kernel"概念(单 kernel 同时承担 prepare/finalize 与 experts),移除 nvfp4 quant pre-AG 的 HACK,统一 MoE kernel 组合方式。

## 动机

vLLM 的 fused MoE 已有 modular kernel 框架:把 MoE 操作拆成 `FusedMoEPrepareAndFinalize`(量化 + All2All Dispatch/Combine)与 `FusedMoEPermuteExpertsUnpermute`(核心专家计算)两个可组合组件。TRTLLM MoE kernel 此前游离在该框架之外,无法与其他 prepare/finalize 后端(DeepEP HT/LL 等)组合复用,也享受不到 MK 的组合测试与 profiling 基础设施;同时 nvfp4 量化的 pre-AG(All-Gather 前)路径存在特判 HACK。本 PR 属 MoE refactor 系列的结构性统一工作。

## 优化点

机制上:(1) 把 TRTLLM kernels 改写为 MK 组件,使其可与框架内其他 prepare/finalize、experts 实现组合;(2) 向 MK 框架引入 monolithic kernel 概念——某些后端(如 TRTLLM)以单一整体 kernel 完成 prepare/finalize + experts 的全部工作,不适合拆成两阶段接口,`maybe_make_prepare_finalize` 增加 `use_monolithic` 选项(如 benchmark 中显式传 `use_monolithic=False` 走标准拆分,monolithic 则整体替换);(3) 框架层重命名:`FusedMoEPrepareAndFinalize` → `FusedMoEPrepareAndFinalizeModular`、`FusedMoEPermuteExpertsUnpermute` → `FusedMoEExpertsModular`,设计文档 `fused_moe_modular_kernel.md` / `moe_kernel_features.md` 全文同步更新(接口语义不变:prepare 做量化与 All2All Dispatch、`prepare_no_receive` 支持与通信重叠计算、finalize 做 Combine 与可选的 TopK weight/reduce);(4) 移除 nvfp4 quant pre-AG 的 HACK;(5) kernel benchmark 脚本统一改用 `mk.FusedMoEKernel` + `maybe_make_prepare_finalize` 构造。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/(TRTLLM MK 适配文件在 diff 中被略去): TRTLLM kernels 接入 modular kernel 框架
docs/design/fused_moe_modular_kernel.md: 组件重命名与 monolithic kernel 概念说明
benchmarks/kernels/benchmark_moe.py 等: 统一改用 FusedMoEKernel/maybe_make_prepare_finalize(use_monolithic 参数)
.buildkite/test_areas/kernels.yaml: test_modular_oai_triton_moe.py 拆为独立 CI 分片
```

关键片段(benchmark 构造方式变更):

```python
fn = mk.FusedMoEKernel(
    maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        allow_new_interface=True,
        use_monolithic=False,
    ),
    CutlassExpertsFp8(moe_config=moe_config, quant_config=quant_config),
)
```

## 性能声明

PR 正文未提供 benchmark 数字(结构重构类 PR,perf claims 检测为 none)。性能意义在于 TRTLLM kernel 可经 MK 框架与 DBO/shared expert overlap 等 async prepare/finalize 后端组合,间接获得端到端收益路径。

## 验证与回归风险

MK 组合单元测试 `tests/kernels/moe/test_modular_kernel_combinations.py` 迭代所有 prepare/finalize × experts 组合做正确性验证(新增实现需登记到 `MK_ALL_PREPARE_FINALIZE_TYPES`/`MK_FUSED_EXPERT_TYPES`);CI 把 `test_modular_oai_triton_moe.py` 从 kernels/moe 分片拆出单独跑。回归风险:类重命名波及所有 FusedMoEMethodBase 派生类的 `apply`/`select_gemm_impl` 实现(72 个文件);monolithic kernel 分支绕过两阶段接口,与 chunking/expert map 的兼容性依赖实现自查;nvfp4 pre-AG HACK 移除后相应路径行为可能变化。

## 关联

MoE refactor 系列的其他 PR(编号素材未注明);设计文档引用 #20228(TopK weight application 在 experts 内实现的示例)。索引内同区域(MoE)卡片:#28449(DeepEP low-latency + async)、#24330(Qwen3 thinking fused MoE 优化)、#24597(Kimi K2 fused MoE 优化)、#28200(fused MoE triton config tuning)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
