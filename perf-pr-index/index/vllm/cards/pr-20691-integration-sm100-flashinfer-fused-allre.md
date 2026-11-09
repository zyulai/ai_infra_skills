# PR #20691: Integration SM100 FlashInfer fused allreduce RMSNorm

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ilmarkov |
| 合并 | 2025-07-12 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/20691 |

## 一句话结论

新增 Inductor pass 把 TP 的 allreduce + RMSNorm 融合为 FlashInfer trtllm one-shot allreduce,B200 上 Llama-70B TP=4 TPOT 提速约 10-15%。

## 动机

TP 推理时每层 decoder 都执行 `tensor_model_parallel_all_reduce` 后接 RMSNorm(或 fused_add_rms_norm),两个独立的 kernel launch 与一次中间张量往返;SM100(B200)上 FlashInfer 提供 TRTLLM 风格的 fused allreduce+RMSNorm one-shot kernel(Lamport flag、PDL launch、fp32 累加),可以把两步合成一次。此前 vLLM 的 `collective_fusion.py` 只有 AsyncTP(reduce-scatter/all-gather)模式,没有这条融合路径。

## 优化点

实现为 `AllReduceFusionPass`(VllmInductorPass,接入 pass_manager,由 `pass_config.enable_fi_allreduce_fusion` 开启,`fi_allreduce_fusion_max_token_num` 默认 1024):用 pattern matcher 匹配两类图模式——`all_reduce → rms_norm` 与 `all_reduce → fused_add_rms_norm`(epsilon 1e-5/1e-6 各注册一份),替换为自定义 custom op `flashinfer_trtllm_fused_allreduce_norm`(mutates allreduce_in/residual/norm_out,fake impl 供编译)。运行时按张量字节数判定:不超过 `_FI_MAX_SIZES`(world size 2/4 为 1 MiB,6/8 为 512 KiB)且不超过 max_token_num 限制时走 `flashinfer_comm.trtllm_allreduce_fusion` one-shot(`kARResidualRMSNorm` pattern,launch_with_pdl=True,fp32_acc=True);超限回退原 allreduce + norm 组合。pass 初始化时经 `trtllm_create_ipc_workspace_for_all_reduce_fusion` 建 IPC workspace(flashinfer 未安装或 world size 不支持时禁用),析构销毁。启用方式:`--compilation-config='{"pass_config": {"enable_flashinfer_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level":3}'`。

## 关键改动

```text
vllm/compilation/collective_fusion.py: AllReduceFusionPass、
  AllReduceRMSNORMPattern / AllReduceFusedAddRMSNormPattern、
  flashinfer_trtllm_fused_allreduce_norm custom op、_FI_MAX_SIZES
vllm/compilation/pass_manager.py: 注册该 pass
vllm/config.py: PassConfig.enable_fi_allreduce_fusion 与
  fi_allreduce_fusion_max_token_num
tests/compile/test_fusion_all_reduce.py: 2 GPU 替换验证
```

关键机制摘录(collective_fusion.py,模式替换):

```python
def replacement(input, rms_result, weight):
    residual = torch.zeros_like(input)
    allreduce = auto_functionalized(
        torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default,
        allreduce_in=input, residual=residual, norm_out=rms_result,
        rms_gamma=weight, rms_eps=self.epsilon, ...)
```

## 性能声明

meta-llama/Llama-3.1-70B-Instruct,TP=4,B200,`vllm bench serve` sonnet 数据集,60 秒窗口、request-rate 控制(QPS 1/5/10):

- TPOT:QPS 1 mean 13.210 → 11.875 ms,QPS 5 14.442 → 13.261 ms,QPS 10 17.834 → 15.352 ms——约 10-15% 提升;
- TTFT 基本持平(QPS 1 42.557 → 43.9 ms,QPS 10 54.465 → 52.725 ms,有升有降)。

其余数字(逐 QPS 的 median 等)见正文两张表;n sys 截图对比为正文附图(未收入 pack)。

## 验证与回归风险

测试 `tests/compile/test_fusion_all_reduce.py`:2 GPU(`multi_gpu_test`),SM100 且装 flashinfer 才跑,piecewise 编译级 + `+rms_norm` custom op,分别验证 `all_reduce → RMSNorm` 与 `all_reduce → fused_add_rms_norm` 两个模型在 pass 前仍含 `vllm.all_reduce`、pass 后被 `flashinfer_trtllm_fused_allreduce_norm` 替换(TestBackend before/after op 检查),fp16/bf16 两种 dtype。风险面:仅支持 world size ∈ {2,4,6,8} 与 SM100;张量超过 `_FI_MAX_SIZES` 自动回退(回退路径额外做一次 `copy_` 把结果写回原 buffer);custom op 通过 mutation 语义与 auto_functionalized 集成,依赖 FixFunctionalizationPass 的正确处理;workspace 显存随 max_token_num × hidden 分配。

## 关联

- 同区域:vLLM 索引中的 TP 通信优化,如 PR #20759(PyTorch symmetric memory all-reduce)、PR #2192(custom all-reduce kernels)、PR #55072(low-SM multimem reduce);后续 SM100 通信融合工作可与其对照

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
