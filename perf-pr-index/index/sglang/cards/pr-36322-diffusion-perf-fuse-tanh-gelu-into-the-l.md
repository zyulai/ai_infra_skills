# PR #36322: [diffusion] perf: fuse tanh-GELU into the LongCat-Image DiT FFN up-proj

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @ITerydh |
| 合并 | 2026-08-26 |
| 分类 | Model-specific Fast Path | tier: major |
| Label | run-ci, diffusion, run-ci-extra |
| Diff | diffstat 缺失(pack 未提供 +N/-N);diff 仅 1 个文件:`longcat_image.py` |
| 链接 | https://github.com/sgl-project/sglang/pull/36322 |

## 一句话结论

把 LongCat-Image DiT 的 FFN up-proj 接入已有 cublasLt linear_gelu_tanh epilogue,GELU 区域提速 6.2%、端到端 denoise 快 1.50%。

## 动机

LongCat-Image DiT 在 FFN up-projection GEMM 之后把 tanh-GELU 作为独立 op 运行,涉及两处:dual-stream FFN(`ff` / `ff_context`)与 single-stream block 的 `proj_mlp`。SGLang 已内置 AOT cublasLt `linear_gelu_tanh` epilogue(把 bias-add 与 tanh-GELU 折叠进 GEMM),FLUX.1、Qwen-Image、GLM-Image、LTX-2、HunyuanVideo 均已使用,但 LongCat 从未标记自己的调用点,融合机制无法生效。在 LongCat-Image-Edit 上这些调用点每次生成执行 4000 次(50 步 x 80 个 site),独立 activation 成为 denoise 时间中可测的一部分——这是一个纯接线缺失,而非缺 kernel。

## 优化点

单文件改动,把 LongCat-Image 接入既有 site 机制:`_LongCatFFN.__init__` 调用 `mark_fused_gelu_site(self.net[0], "proj")` 覆盖每个 dual-stream block 的 `ff`/`ff_context`(被标记模块是持有 `proj` 的 ModuleDict,`getattr` 可为融合 helper 解析);`_LongCatFFN.forward` 在 `fused_gelu_active()` 与 `can_use_linear_gelu()` 同时成立时改走 `fused_linear_gelu_tanh(hidden_states, proj.weight, proj.bias)`,否则回退原 `proj` + `nn.GELU(approximate="tanh")` 链且逐字不变;`_SingleTransformerBlock` 的 `proj_mlp`/`act_mlp` 做同样处理。kernel、gate、mount 代码零改动:`mount_fused_linear_gelu`/`unmount_fused_linear_gelu` 已在 denoising stage 于 batch 边界自动发现标记 site、按 transformer 全有或全无挂载,`quality` 参数已参与 dynamic-batch signature。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py: 标记 ff/ff_context 与 proj_mlp 两个 GELU site,forward 分支走 fused_linear_gelu_tanh
```

```python
if fused_gelu_active(self.net[0]) and can_use_linear_gelu(proj, hidden_states):
    hidden_states = fused_linear_gelu_tanh(
        hidden_states, proj.weight, proj.bias
    )
else:
    hidden_states, _ = proj(hidden_states)
    hidden_states = self.act(hidden_states)
```

## 性能声明

条件:LongCat-Image-Edit,A800,50 steps,seed 42;GELU 区域用临时插桩(包住 "up-proj GEMM + tanh-GELU")测量,提交前移除;每配置 2 次重复、方差 <0.2%。

- GELU 区域(50 步合计):`lossless`(未融合)9638 ms → `high`(融合)9044 ms,**-594 ms / -6.2%**;单次调用 2.410 ms → 2.261 ms(省 0.148 ms/次);每次生成 4000 次调用。
- 端到端 denoise(每配置 5 次):`lossless` 45.496 s → `high` 44.812 s,**-0.68 s / -1.50%**。
- 其他分辨率/模型变体:PR 未注明。

## 验证与回归风险

测试:无新 kernel,site 机制已由 `test/registered/kernels/ops/diffusion/test_sites.py` 覆盖;作者 checklist 全部勾选(pre-commit、单测、文档、accuracy/speed)。回归风险:融合仅在 `quality=high` 时由 denoising stage 按 batch 挂载,默认(`lossless`)关闭且逐位不变(diff 注释:off bit-exact by default);`can_use_linear_gelu` 条件不满足(形状/dtype/TP 约束)时回退原路径;数值上融合走 cublasLt epilogue 的累加顺序,PR 未给出 fused 路径与原路径的输出差异量化(仅默认路径声明 bit-exact);文档已含机制与 `--quality` 开关说明,LongCat-Image 行待评审决定是否补充。

## 关联

同机制既有接入方:FLUX.1、Qwen-Image、GLM-Image、LTX-2、HunyuanVideo。无正文列出的具体 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
