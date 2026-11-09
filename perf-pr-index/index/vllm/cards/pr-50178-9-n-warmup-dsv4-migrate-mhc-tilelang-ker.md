# PR #50178: [9/N][warmup][DSv4] Migrate MHC TileLang kernels

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @LopezCastroRoberto |
| 合并 | 2026-09-13 |
| 分类 | Attention / Kernel（tier: major） |
| Label | ready, v1, deepseek, DSv4, inkling |
| Diff | 素材未提供 diffstat（pack 标注 +None/-None, None files）；pack 末尾标注 6 file(s) dropped from diff for size |
| 链接 | https://github.com/vllm-project/vllm/pull/50178 |

## 一句话结论

把 DSv4 MHC 的 TileLang kernels 迁到共享 JIT warmup 契约，specialization heuristics 下沉到各 kernel 的 dispatch()。

## 动机

正文 Description 只有一句：把 DSv4 MHC TileLang path 迁移到 shared JIT warmup contract。标题 `[9/N][warmup][DSv4]` 表明这是系列拆分中的一环，正文并注明 Depends on #49315、父 draft PR #49627 与跟踪列表 issue #49349。正文未描述改动前的具体性能瓶颈或退化，也未给出改动动机之外的背景；这部分属于「PR 未注明」。从 What Changed 可见目标是把 warmup 搜索空间与 specialization 决策统一收拢到 kernel 自身，为共享 JIT warmup 契约铺路。

## 优化点

What Changed 列出：迁移 pre-fusion、post-fusion、fused MHC、head-fusion、prenorm GEMM 五个 TileLang kernels；把 TileLang specialization heuristics 移入 kernel 自有的 `dispatch(...)` 方法；通过 AST ranges、coupled inputs 与 predicates 定义 warmup 搜索空间；新增一个使用 fake tensor descriptors 的共享 TileLang compile-only helper；保持 TileLang 与 DeepGEMM 的 dispatch 关注点分离；保留既有 TileLang kernel bodies 与 runtime selection 行为；新增测试对比迁移后的 dispatch keys 与原 heuristics。diff 中 `tilelang_kernels.py` 引入 `vllm.model_executor.warmup.jit_warmup` 的 `WarmupIntRange`、`zip_inputs` 与 `jit_warmup_tilelang_helper`（`TileLangLaunchSpec`、`VllmTileLangJitKernel`、`kernel_launcher`、`make_tilelang_warmup_tensor`），新增 `HcPrenormGemmTileLangKernel` 及其 `CompileKey` dataclass（hidden_size/hc_mult/n_out/n_thr/tile_n/n_splits/use_block_m/block_m）与 `dispatch()`——依据 `num_tokens >= 1024` 决定 `use_block_m`、小 tile 场景改用 `n_thr=1024/tile_n=4`；另加 `_mhc_fused_n_splits(num_tokens, hidden_size)`。`tilelang.py` 把原 `_tilelang_hc_prenorm_gemm` 改为 `_hc_prenorm_gemm_outputs(...)`，按 `is_deep_gemm_supported() or not use_tilelang_fallback` 决定走 `tf32_hc_prenorm_gemm`（DeepGEMM）还是 TileLang，并用 `compute_num_split` 计算 n_splits。

## 关键改动

```text
vllm/model_executor/kernels/mhc/tilelang_kernels.py: 引入 warmup helper，新增 HcPrenormGemmTileLangKernel/CompileKey/dispatch() 与 warmup 搜索空间
vllm/model_executor/kernels/mhc/tilelang.py: 改为 outputs 工厂，分离 DeepGEMM 与 TileLang 的 dispatch 关注点
tests/kernels/test_mhc_jit_warmup.py: 新文件，对比迁移后 dispatch keys 与原 heuristics
tests/kernels/test_mhc_kernels.py: MHC kernel 测试

# 机制摘录 (tilelang_kernels.py: HcPrenormGemmTileLangKernel.dispatch)
    def dispatch(  # type: ignore[override]
        self,
        *,
        num_tokens: int,
        hc_hidden_size: int,
        n_thr: int = 512,
        tile_n: int = 12,
        n_splits: int = 1,
        **compile_key_fields: int,
    ) -> CompileKey:
        use_default_config = n_splits == 1 and tile_n == 12 and n_thr == 512
        use_block_m = use_default_config and num_tokens >= 1024
```

## 性能声明

PR 正文未给出定量性能数据（Description 与 What Changed 均为重构描述，无 benchmark 数字），也未注明测试条件。正文没有吞吐/延迟/精度类定量结果，故无可照抄的性能数字与模型/硬件/精度/workload 条件；PR 未注明测试条件。

## 验证与回归风险

PR 正文未给出 Test Plan 与 Test Result。What Changed 提到「Added tests comparing migrated dispatch keys with the original heuristics」，diff 中新增 `tests/kernels/test_mhc_jit_warmup.py` 并修改 `tests/kernels/test_mhc_kernels.py`，可视为随 PR 引入的验证手段：用迁移后的 dispatch keys 对照原 heuristics 保证行为一致。回归范围方面，正文明确「Preserved the existing TileLang kernel bodies and runtime selection behavior」以及「Kept TileLang and DeepGEMM dispatch concerns separate」；`tilelang.py` 中 DeepGEMM/TileLang 的分流（`use_deep_gemm` 判定）与新增的 `dispatch()` 配置决策是潜在影响点。pack 末尾另有 6 个文件被从 diff 中 dropped。正文未给出精度/性能回归结论，也未注明测试硬件与模型。

## 关联

PR 正文明确：Depends on #49315，父（draft）PR #49627，跟踪列表 issue #49349；标题 `[9/N]` 表明为系列拆分之一，其余同分类 PR 编号未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
