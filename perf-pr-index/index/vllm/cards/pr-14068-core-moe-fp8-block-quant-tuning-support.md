# PR #14068: [core] moe fp8 block quant tuning support

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @divakar-amd |
| 合并 | 2025-03-04 |
| 分类 | MoE(tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材完整可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/14068 |

## 一句话结论

让 fused MoE Triton 调参脚本感知 FP8 block quant,为 DeepSeek-V3 类模型产出正确的调参配置。

## 动机

现有 fp8 tuning 实现不考虑 block quantization:对 DeepSeek-V3 这类 weight_block_size=[128,128] 的模型,tuning 时不计入 weight scales(用 per-tensor 随机 scale 建模),选出的配置不能代表真实 kernel 行为,调参结果失真。

## 优化点

benchmark_moe.py 全链路引入 block_quant_shape:从模型 HF config 的 quantization_config.weight_block_size 读取;benchmark 时按 block 形状生成 (E, n_tiles, k_tiles) 的 w1/w2 scale(乘 1e-2 factor)并传 block_shape 给 kernel;搜索空间剪枝剔除 BLOCK_SIZE_K 不整除 block_k、BLOCK_SIZE_N 不整除 block_n 的配置;保存的 config 文件名带上 block_shape,与运行时查表键一致。顺带修正 ROCm prune 中漏乘 topk 的问题(num_tokens*2 → num_tokens*topk)。

## 关键改动

```text
benchmarks/kernels/benchmark_moe.py: block_quant_shape 参数、block scale 生成、配置剪枝、文件名带 block_shape
vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json: MI300X DSv3 形状配置更新
```

```python
if block_quant_shape is not None and not use_fp16:
    block_n, block_k = block_quant_shape[0], block_quant_shape[1]
    for config in configs[:]:
        if config["BLOCK_SIZE_K"] % block_k != 0 or config["BLOCK_SIZE_N"] % block_n != 0:
            configs.remove(config)
```

MI300X 配置随此次 tuning 重写:小 batch 的 BLOCK_SIZE_N 多从 32/64 改为 128,num_warps/num_stages 相应调整,并新增 1536/2048/3072/4096 batch 条目。

## 性能声明

PR 正文未给出 benchmark 数字(regex 亦未检出)。性能收益通过重新 tuning 后的 MI300X 配置文件体现,但 PR 未附调参前后的 kernel 计时对比。

## 验证与回归风险

验证即 tuning 流程本身(benchmark_moe.py --tune)与产出的配置文件,PR 未列单元测试。风险面:影响所有依赖 benchmark_moe.py 调参的用户工作流;config 文件名新增 block_shape 字段,旧的不带 block_shape 的 fp8 block-quant 配置文件名将不再由该脚本生成;非 block-quant fp8(w8a8 per-tensor)路径保持原行为(else 分支)。

## 关联

同分类 MoE 卡:#19205(H20-3e block_shape=[128,128] 配置文件即本 PR 文件名约定的产物)、#23748(block-FP8 GEMM H100/H200 调参)、#39904(gpt-oss H100 fused_moe 配置)。无前序/后续 PR 声明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
