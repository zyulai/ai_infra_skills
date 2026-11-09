# PR #30719: [Perf] [Intel] diffusion: Adding AMX optimizations on CPU platform

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @jianan-gu |
| 合并 | 2026-11-05 |
| 分类 | Other (tier: major) |
| Label | lora, Multi-modal, sgl-kernel, intel, cpu, run-ci, diffusion |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);1 file dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/30719 |

## 一句话结论

CPU 平台 diffusion(multimodal_gen)接入 Intel AMX:未量化 linear 权重 VNNI 打包走 AMX GEMM,新增 AMX attention 后端替代 Torch SDPA。

## 动机

本 PR re-land #28527(原 PR 因重复调用 `process_weights_after_loading` 破坏 CI 被 #30716 revert),并附上去重修复。此前 CPU 上 diffusion 的未量化 linear 只走普通 torch GEMM,attention 只有 Torch SDPA,AMX 矩阵引擎未被利用。

## 优化点

(1) linear 侧:`UnquantizedLinearMethod` 增加 `process_weights_after_loading`,在具备 AMX 支持的 CPU 上把 weight 转为 VNNI packed 布局;`apply` 时经 `use_intel_amx_backend` 判断走 `torch.ops.sgl_kernel.weight_packed_linear(is_vnni=True)`。Qwen-Image 的 merged linear(added qkv)对 packed 权重增加专门分支:单次 packed linear 后再 split,避免按行切 VNNI 权重出错。(2) attention 侧:新增 `AMXAttentionBackend`/`AMXATTNImpl`(自 FastVideo 适配),调用 sgl-kernel 的 CPU `flash_attn_varlen_func`;支持 head 32–256、bf16/fp16、偶数 head size。CPU 平台 backend 选择器重写:AMX 可用时默认选 AMX attn,layer 声明的 supported set 缺 `amx_attn` 时回落 SDPA,显式选 SDPA 被尊重,fp32 层回落(该 kernel 只 dispatch 浮点缩减类型)。(3) `flash_attn_varlen_func` 新增 optional `sm_scale` 参数(此前固定 1/sqrt(head_size)),vision attention 可传自定义 softmax_scale。(4) LoRA:AMX packed 权重不可 in-place merge(`can_merge_base_weight=False`),强制 dynamic LoRA,避免把 dense delta 写进 packed storage。

## 关键改动

```text
python/sglang/multimodal_gen/runtime/layers/linear.py: UnquantizedLinearMethod 接 AMX packed 权重与 weight_packed_linear
python/sglang/multimodal_gen/runtime/layers/attention/backends/amx_attn.py: 新文件,AMX attention 后端(自 FastVideo 适配)
python/sglang/multimodal_gen/runtime/platforms/cpu.py: CPU attention 选择器(AMX 优先、按 supported set 回落)
python/sglang/kernels/aot/csrc/cpu/flash_attn.cpp: flash_attn_varlen_func 增加 optional sm_scale
python/sglang/multimodal_gen/runtime/layers/lora/linear.py: AMX packed 权重禁用 LoRA in-place merge
```

## 性能声明

PR 正文未提供 benchmark 数字,PR 未注明测试条件(被 revert 的原 PR #28527 正文亦未随本 pack 附带数据)。

## 验证与回归风险

新增 `test/registered/cpu/test_diffusion_amx.py`:backend 选择 4 例(supported set 缺 amx 回落 SDPA、显式 SDPA 尊重、fp32 回落、bf16 得 AMX)、AMXAttentionBatch 逐 batch 校验(修复此前只 attend query[0]、batch=2 返回 batch=1 的 bug)、Qwen-Image packed added-QKV split 数值校验;`test_lora_pipeline.py` 新增 AMX LoRA 生命周期测试(packed base 不被 dense delta 覆盖、merge/unmerge 精确、显式 merge 报 ValueError);`test_flash_attn.py` 增加 softmax_scale 维度。风险:AMX 打包后 weight 形状不变但为 VNNI 布局、bias 变 fp32,任何直接切片权重的路径都需专门分支;LoRA merge 模式受限;CI Base 未执行(no_entry_sign)、Extra 失败、AMD ROCm 排队。

## 关联

原始 PR #28527、revert #30716(正文明确,卡片库中 #28527 亦有卡);同区域(卡片库中):#29117(CPU GDN prefill)、#29382(CPU faster exp in SiLU)、#44394(CPU prefill torch compile)。
