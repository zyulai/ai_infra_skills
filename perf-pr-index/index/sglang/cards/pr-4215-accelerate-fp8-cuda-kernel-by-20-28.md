# PR #4215: Accelerate FP8 CUDA Kernel by 20-28%

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hebiao064 |
| 合并 | 2025-03-12 |
| 分类 | Quantization(major) |
| Label | high priority |
| Diff | pack 未提供 diffstat(diff 正文可用,5 个文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/4215 |

## 一句话结论

给 per-token FP8 动态量化 kernel 加向量化访存并新增统一 `scaled_fp8_quant` 入口,对 vLLM 实现最多快 28.72%、对旧实现快 20-25%。

## 动机

属于 sglang #2965"CUDA kernel 优化"专项。当时 SGLang 的 fp8 per-token/per-tensor 量化直接调 vllm 的 `_custom_ops.scaled_fp8_quant`,而 sgl_kernel 自带的 `sgl_per_token_quant_fp8` kernel 没有使用向量化访存(max 归约逐元素标量 load,量化阶段虽有 vec_t 但 max 阶段先做了一遍标量读)。作者的目标是优化自有 kernel 并把量化入口收敛到 SGLang 自己的 API 上。

## 优化点

两件事。(1) `sgl-per-token-quant-fp8` kernel 向量化:max 归约阶段改用 `flashinfer::vec_t<T, 16/sizeof(T)>` 的 `cast_load`(8 元素/线程的向量化读),量化阶段同样向量化,并删除量化后的逐元素 remaining 尾部循环;`block_size` 从 128 提到 256。关键约束:`cast_load` 要求地址按 8 元素(或 float32 时 4 元素)对齐——per_tensor/per_token_group 的加载模式天然对齐,而 per_token 按 `token_idx * hidden_dim` 取行首,`hidden_dim % 8 != 0` 时第二行起始地址不对齐会触发 misaligned address 错误。由于观察到的所有模型 hidden_dim 都是 8 的倍数,PR 选择直接加 `TORCH_CHECK(hidden_dim % 8 == 0)` 硬检查而非处理尾部元素。(2) 新增 `sglang.srt.custom_op.scaled_fp8_quant` 仿 vLLM 签名的统一入口:2D 输入,动态 scale 时按 `use_per_token_if_dynamic` 分派到 `sgl_per_token_quant_fp8`(per-token)或 `sgl_per_tensor_quant_fp8`(per-tensor,动态),静态 scale 走 per-tensor 静态;HIP 下输出 `float8_e4m3fnuz`。`fp8.py`、`ep_moe/layer.py`、`fused_moe.py` 的调用点在 CUDA 上迁到新入口,非 CUDA 保留 vllm ops(各处 `ops.` 改名 `vllm_ops.` 以显式区分)。

## 关键改动

```text
sgl-kernel/csrc/gemm/per_token_quant_fp8.cu: max/quant 双阶段 vec_t 向量化 + hidden_dim%8 检查 + block 256
python/sglang/srt/custom_op.py: 新增 scaled_fp8_quant 统一 API
python/sglang/srt/layers/moe/fused_moe.py: invoke_fused_moe_kernel 的激活量化改走 sgl 入口
python/sglang/srt/layers/moe/ep_moe/layer.py: expert 权重量化改走 sgl 入口
python/sglang/test/test_custom_ops.py: 新增 per-tensor/per-token 量化对照测试
```

## 性能声明

per-token-dynamic-quant-fp8 性能对照表(vLLM vs SGL Kernel,单位未注明,按上下文为微秒;PR 未注明 GPU 型号与精度):batch 16 / seq_len 4096 时 1154.43 → 831.42;batch 32 / 4096 时 2289.30 → 1639.98;batch 128 / 4096 时 9107.18 → 6489.15。表头注明 "✅ All implementations match"。声明"up to 28.72%"时间削减,大输入(如 128 tokens、4096 hidden_dim)最明显;新 kernel 相对旧实现(未向量化)快 20-25%。旧基准见 #4089。E2E 模型级数字与精度评测未提供。

## 验证与回归风险

新增 `python/sglang/test/test_custom_ops.py`:per-tensor 动态/静态量化与 per-token 动态量化对纯 torch 参照实现做 `assert_close`,并刻意用 11x11(`% 8 != 0`)与 11x16(`% 8 == 0`)两种形状覆盖向量化的对齐边界。风险面:kernel 加了 `hidden_dim % 8 == 0` 硬检查,不满足的 hidden_dim 直接报错而非回退标量路径(作者依据是所有已观察模型都满足);调用点迁移只发生在 `_is_cuda` 分支,HIP/其他后端继续用 vllm ops;`ep_moe/layer.py` 中专家权重 per-tensor 量化结果需与 vllm 实现逐位一致(测试覆盖了量化参照)。

## 关联

Part of #2965(kernel 优化专项);旧基准 PR #4089。同区域:fp8 量化 kernel 后续演化线——#30924(trait-driven per_token_group_quant 统一家族)。pack 未列出其他具体 PR 编号。
