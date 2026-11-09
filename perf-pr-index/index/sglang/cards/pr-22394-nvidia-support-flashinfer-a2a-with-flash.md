# PR #22394: [NVIDIA] Support flashinfer a2a with flashinfer_trtllm_routed moe

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @trevor-m |
| 合并 | 2026-06-29 |
| 分类 | MoE |
| Label | documentation, run-ci, release-highlight |
| Diff | diffstat 缺失(pack 未提供 +X/-Y 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/22394 |

## 一句话结论

打通 flashinfer all-to-all 与 flashinfer_trtllm_routed MoE runner 的组合,并支持通信前 FP4 量化。

## 动机

此前 `--moe-a2a-backend flashinfer` 只允许搭配 flashinfer_cutlass 或 flashinfer_cutedsl runner(server_args 断言限制),flashinfer_trtllm_routed——即 Blackwell 上由 TensorRT-LLM 加速、消费 SGLang 侧 top-k expert 指派的 routed MoE 内核——无法与 flashinfer a2a 组合使用。

## 优化点

在 MoeRunner 中注册 `("flashinfer", "flashinfer_trtllm_routed")` 的 fused_func:利用 FlashinferDispatchOutput 与 StandardDispatchOutput 字段布局一致(hidden_states, hidden_states_scale, topk_output)这一点,直接复用已有 FP4/FP8/BF16 实现,再把结果包成 FlashinferCombineInput 走 flashinfer dispatcher 的 combine。dispatch 侧支持先做 NVFP4 量化再通信:当 dispatch 输出已带 hidden_states_scale 时跳过二次量化,输出 dtype 相应取 bf16,减少 a2a 传输字节量。同时移除零 local token 时的 dummy token workaround(flashinfer a2a 现已原生支持零 token),对空 batch 构造空 FP4 payload,并以 `invalid_token_expert_id=-1` 触发 kernel 内 sanitize,填充 token 不再浪费下游 MoE 计算。

## 关键改动

```text
python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py: 注册 fused_func,NVFP4 dispatch 输入免二次量化
python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py: 移除 dummy token 路径,零 token 空载荷 + invalid_token_expert_id
python/sglang/srt/server_args.py: moe-a2a-backend=flashinfer 放行 flashinfer_trtllm_routed
test/registered/ep/test_flashinfer_a2a.py: 新增 FP4(DeepSeek V3)与 FP8(Qwen3-Next)e2e 测试
```

## 性能声明

GB200 FP4 disagg(4xDEP4 prefill + 1xDEP32 decode,1k/1k workload,5120 条输入):新组合并发 4096 时 output token throughput 73488.77 tok/s、request throughput 79.75 req/s、mean TPOT 51.97 ms;baseline(deepep + flashinfer_cutedsl + SBO)同并发为 116179.28 tok/s、126.08 req/s、mean TPOT 19.25 ms——该配置下新组合吞吐低于 baseline(PRR 正文还给出并发 512/2048 两组数据,结论相同)。准确性:DeepSeek-R1-0528 FP8(tp4/ep4/dp4)GSM8K platinum 8-shot 0.984;DeepSeek-R1-0528-FP4-v2(modelopt_fp4)0.978。

## 验证与回归风险

新增 `test/registered/ep/test_flashinfer_a2a.py`(4-gpu-gb300 runner,FP4 DeepSeek-V3-0324 与 FP8 Qwen3-Next-80B 各跑 GSM8K,阈值 0.90/0.93)。回归面:combine 输出 buffer 的 dtype 判定改为 output_dtype(FP4 dispatch 下固定 bf16);moe_runner_backend 校验放宽;`SGLANG_MOE_NVFP4_DISPATCH` 自动开启条件扩大到 hybrid NVFP4 模型,均可能影响既有 flashinfer a2a 用户的路径行为。

## 关联

同区域:flashinfer a2a 通信栈的其他使用方,如 PR #21637(DCP fi_a2a 后端复用 FlashInfer MNNVL 通信);与 flashinfer_cutlass/flashinfer_cutedsl runner 的组合为既有路径。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
