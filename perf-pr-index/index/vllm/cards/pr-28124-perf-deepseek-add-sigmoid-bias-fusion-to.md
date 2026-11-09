# PR #28124: [Perf][DeepSeek] Add sigmoid+bias fusion to fused_grouped_topk from TRTLLM

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgoin |
| 合并 | 2025-11-08 |
| 分类 | Sampler / Detokenization(major,classify 结果,实为 MoE 路由 kernel) |
| Label | performance, ready, deepseek, nvidia |
| Diff | 素材未提供 diffstat 数字;pack diff 收录 kernel、binding 与 Python 封装改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/28124 |

## 一句话结论

把 DeepSeek 分组 top-k 的 sigmoid、e_score_correction_bias 与 FP32 输出转换全部融合进 fused_grouped_topk CUDA kernel,B200 中等并发 benchmark 提速约 4.5%。

## 动机

DeepSeek 系模型的 MoE 路由此前虽然用了 fused grouped top-k kernel,但 sigmoid 激活、`e_score_correction_bias` 加偏、以及 topk_values 转 float32 都发生在 Python 侧:先 `gating_output.sigmoid()`,再拼出 `scores_with_bias` 中间张量传入 kernel,kernel 返回后还要做 `.to(torch.float32)`/`.to(torch.int32)` 转换(issue #28086)。这些额外的 elementwise kernel launch 和中间张量在每层 MoE 都要付一次。

## 优化点

借鉴 TRTLLM 的 noAuxTc 实现,把三件事全部移进 CUDA kernel:kernel 内联 TRTLLM 的高精度 sigmoid 近似 `0.5*tanh(0.5x)+0.5`,`topk_with_k2` 与 `group_idx_and_topk_idx` 两个 kernel 在读取每个专家分数时原位施加 sigmoid(可由 `scoring_func` 开关)并加 bias,不再需要 `scores_with_bias` 中间张量;归一化仍用不含 bias 的 sigmoid 分数;`topk_values` 直接以 float32 输出,消除 Python 侧转换。`grouped_topk` 签名从 `(scores, scores_with_bias)` 改为 `(scores, bias, scoring_func)`。softmax 路径暂仍在 Python 侧预计算后以 `scoring_func=0` 进 kernel(留 TODO 在 kernel 内支持)。

## 关键改动

```text
csrc/moe/grouped_topk_kernels.cu: 新增 SCORING_SIGMOID 与 sigmoid_accurate,两个 kernel 融合 sigmoid+bias;topk_values 输出改 float32
csrc/moe/torch_bindings.cpp / csrc/moe/moe_ops.h: grouped_topk 签名改为 (scores, ..., bias, scoring_func)
vllm/_custom_ops.py: grouped_topk Python 封装同步签名
vllm/model_executor/layers/fused_moe/fused_moe.py: sigmoid 走全融合 kernel 路径,去掉 Python 侧 sigmoid/bias/转换
```

关键片段(grouped_topk_kernels.cu):

```cuda
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}
// 读取分数时:
T value = input[i];
if (scoring_func == SCORING_SIGMOID) value = apply_sigmoid(value);
value = value + bias[i];
```

## 性能声明

- E2E:4×B200,`vllm serve nvidia/DeepSeek-V3-0324-FP4 -tp 4`,`vllm bench serve` random 数据集 input 512/output 128、128 prompts、max-concurrency 32:输出吞吐 752.63→786.43 tok/s(+4.5%),Mean TTFT 839.33→699.63 ms,Mean TPOT 36.15→35.43 ms。PR 自述"约 4.5% speedup on a medium concurrency benchmark with nvfp4 deepseek on B200"。
- 精度:gsm8k(1319 题、5-shot,同 serve 配置)0.951→0.953,invalid responses 均为 0。
- 另附 dummy 模型 profile(`LLM(model="DeepSeek-R1", load_format="dummy", num_hidden_layers=4)`)前后截图,未给出具体数字。

## 验证与回归风险

验证为 gsm8k 评测 + E2E benchmark + dummy profile,diff 中未新增单元测试。风险面:影响所有走 CUDA `fused_grouped_topk` 的 DeepSeek 类模型路由;`grouped_topk` op 签名变化是内部接口,_custom_ops 封装同步更新;sigmoid 在 kernel 内以 tanh 近似实现(与 TRTLLM 一致),精度由 gsm8k 佐证;softmax 路径行为不变。

## 关联

修复 issue #28086。同区域(MoE 路由/top-k kernel):#22211(topk-softmax 融合 kernel)、#31246(topk sigmoid kernel)、#63076(ROCm triton sigmoid+bias top-k)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
