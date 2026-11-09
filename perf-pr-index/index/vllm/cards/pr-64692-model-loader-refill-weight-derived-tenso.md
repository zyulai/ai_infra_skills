# PR #64692: [Model Loader] Refill weight-derived tensors in place on every post-load rerun

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hao-aaron |
| 合并 | 2026-11-02 |
| 分类 | Speculative Decoding (tier: major) |
| Label | speculative-decoding, ready, ci/build, llama, qwen, deepseek, nvidia, quantization, DSv4, dflash, kimi, k3, DSv4.1, beam |
| Diff | pack diffstat 缺失(+None/-None);可见 diff 含 17 个文件,另有 53 个文件因超长未展示 |
| 链接 | https://github.com/vllm-project/vllm/pull/64692 |

## 一句话结论

weight reload 重跑 process_weights_after_loading 时原地回填权重派生张量,让已捕获 CUDA graph 读到新值;没有它 17 个模型的 GSM8K 在 reload 后有 8 个崩到近 0。

## 动机

direct reload(#64336)与 layerwise reload 都会在每次权重更新后重跑 process_weights_after_loading(#64357)。许多量化方法和模型在这一步用新分配的 tensor 派生 kernel 常量(SwiGLU alpha/beta/clamp、inverse scale、decode 路径拷贝)或整体重建 experts 对象。已捕获的 CUDA graph 持有的是启动时的指针,reload 后这些派生张量地址变化,graph 就在已释放内存上读数——权重本身都正确,所以比对权重值无法发现,gpt-oss-20b GSM8K 从 0.39 崩到 0.01。

## 优化点

三层机制:新增 `refill_derived(obj, name, tensor)`——shape/dtype/device 匹配时把新值原位写进旧 tensor(地址不变),否则才替换,且绝不创建 parameter;MoE experts 对象声明 `kept_constants` 元组列出 kernel 读的张量名,`keep_constants(layer)` 把它们留在 layer 上并让新对象指向旧张量,`FallbackExperts` 转发给两个被包裹的 experts;各 linear kernel 的派生量(CUTLASS/Triton INT8 azp_adj、CUTLASS W4A8 weight_chan_scale、Marlin W4A8 input_global_scale)从 register_parameter 新建改为 replace_parameter 原位回填,Marlin FP8 权重 scale 改为 mul_ 原位缩放,b12x/ModelOpt 复用 packed weight 存储。另覆盖 ROCm AITER MXFP4 shuffle、Kimi-K3 AMD _w_fold 与 bf16x3 router split、fused MoE gate 的 _combined_gate_weight、DFlash/DSpark drafter 的 context K/V 权重与 norm、Muse Spark residual 系数等数十个 site。

## 关键改动

```text
vllm/model_executor/model_executor/test_utils.py 侧: refill_derived(新,in-place 回填普通属性)
vllm/model_executor/layers/fused_moe/experts/*.py: 各 experts 类声明 kept_constants(gpt-oss clamp/alpha/beta、CuteDSL per_token_global_scale 等)
vllm/model_executor/kernels/linear/{mixed_precision/{cutlass,marlin},scaled_mm/{cutlass,triton},nvfp4/b12x}.py: 派生 scale/azp_adj 改 replace_parameter / 原位 mul_
tests/model_executor/model_loader/test_reload_moe_kernels.py: 新增,reload 后地址不变 + 输出不变
```

## 性能声明

本 PR 主打 reload 正确性而非加速;冷启动无回退:12 个 model/backend 配置上 GSM8K、吞吐(GB200,vllm bench serve random 1024 in/256 out,concurrency 64)与 KV cache 均在噪声内(吞吐 +-2%、GSM8K +-0.1)。正确性数据:reload 后 GSM8K,main 在 8/17 模型崩(gpt-oss-120b 0.803→0.006、Kimi-K3 0.954→0.246),本 PR 全部保持(0.807→0.813、0.954→0.948);Beam DFlash2 接受长度在无此修复时 7.03→4.89,有修复保持。其余见正文 17 模型表格(GB200/GB300,NCCL weight update,3 次 full-checkpoint 更新)。

## 验证与回归风险

新增/扩展测试:test_reload_moe_kernels.py(SM100 才跑)、test_keep_param_objects.py 的 test_direct_reload_keeps_kernel_tensor_addresses(15 个 quant/kernel case)、test_draft_weight_reload.py、test_muse_spark_logits_local.py、Kimi-K3 AMD fold 测试;CI 接入 B200 Quantized MoE step 与 spec-decode draft-model steps。回归面:改动横跨几乎所有 quant method 与 linear kernel 的 post-load 路径; Humming MoE 的 post-load 每 quant method 跑一次,direct reload 会拒绝它(main 行为);gpt-oss Marlin MXFP4 的 begin_direct_reload 清 padding 是 stopgap,长期靠 #64352。

## 关联

前序:RFC #54477(direct weight reload)、#64336(direct reload core)、#64357(post-load 重跑);后续占位:#64352(padding 回填)、#64418(load plan)。
