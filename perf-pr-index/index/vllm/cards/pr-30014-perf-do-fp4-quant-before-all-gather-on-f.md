# PR #30014: [Perf] Do FP4 quant before All gather on flashinfer trtllmgen MOE

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jiahanc |
| 合并 | 2025-12-16 |
| 分类 | MoE(major) |
| Label | performance, ready, nvidia |
| Diff | diffstat 缺失(素材未提供 +additions/-deletions 与文件数;diff 本身可用,涉及 7 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/30014 |

## 一句话结论

把 FP4 量化挪到 DP All Gather 之前,DeepSeek-R1 上 all gather 消息体积缩小 2.4 倍,加速通信。

## 动机

FlashInfer TRTLLMGEN MoE 在 DP + EP 开启时,`forward_impl` 的 dispatch 流程先把 bf16 hidden_states 原样 all gather 到各 DP rank,再进 MoE kernel 做 FP4 量化。hidden_states 是 All Gather 里最大的负载,未量化就通信等于多传 4 倍以上的数据;All Gather kernel 时间直接受消息大小支配。前序 PR #29804 是本 PR 的阻塞项,合并后同样优化会加到 `flashinfer_trtllm_fp4_routed_moe`。

## 优化点

核心是"量化前移 + 附加张量随通信走":(1) `FusedMoE` 的 quant method 基类新增 hook `prepare_dp_allgather_tensor`,`ModelOptNvFp4FusedMoE` 实现为在 dispatch 前用 `flashinfer.fp4_quantize(hidden_states, a1_gscale, is_sf_swizzled_layout=False)` 把 hidden_states 变成 FP4 打包(uint8,hidden_size/2)+ scale factor(fp8,hidden_size/16),sf 作为 `extra_tensors` 一起传递;(2) all2all/device communicator/parallel_state 的 `dispatch` 签名统一加 `extra_tensors` 参数——BatchedAll2AllManager 用 `all_gatherv` 一并 gather,NaiveAll2AllManager 等不支持则显式 raise;(3) `flashinfer_trtllm_fp4_moe` 接受 `x` 为 tuple(已量化)时跳过重复量化,`apply` 里 `select_experts` 只取路由用的元数据。触发条件:`has_flashinfer_trtllm_fused_moe()` 为真、quant_method 是 `ModelOptNvFp4FusedMoE`、dp_size>1 且 use_ep。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/layer.py: dispatch 前按条件做 FP4 量化,sf 走 extra_tensors
vllm/model_executor/layers/quantization/modelopt.py: prepare_dp_allgather_tensor 实现 fp4_quantize
vllm/distributed/device_communicators/all2all.py: dispatch 支持 extra_tensors(Batched gather/Naive raise)
vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py: 接受已量化的 tuple 输入
vllm/utils/flashinfer.py: has_flashinfer_trtllm_fused_moe() 能力探测
```

关键片段(modelopt.py):

```python
hidden_states_fp4, hidden_states_sf = flashinfer.fp4_quantize(
    hidden_states, a1_gscale, is_sf_swizzled_layout=False)
extra_tensors: list[torch.Tensor] = [hidden_states_sf]
return hidden_states_fp4, extra_tensors
```

## 性能声明

PR 正文给出消息体积对比:原始 hidden_states 为 (num_tokens, hidden_size) bf16;改后为 (num_tokens, hidden_size/2) uint8 + (num_tokens, hidden_size/16) fp8,"On DeepSeek-R1 case, it is a 2.4x less message size"。未给出端到端或 kernel 级延迟数字。正确性:lm-eval gsm8k 5-shot(limit 0.5、batch 2048)在 `nvidia/DeepSeek-R1-0528-FP4-v2`(DP4/TP1/EP、fp8 KV cache)上 exact_match flexible-extract 与 strict-match 均 0.9712±0.0065。

## 验证与回归风险

验证为上述 gsm8k(serve 命令含 `--data-parallel-size 4 --enable-expert-parallel` 及 FI allreduce/attn fusion 编译选项)。风险面:`dispatch` 接口签名变动触及所有 device communicator 子类,不支持的实现必须显式报错;FP4 量化前移后 `select_experts` 只见路由元数据,任何依赖未量化 hidden_states 数值的路由路径需走 tuple 拆包;仅命中 TRTLLMGEN FP4 MoE + DP>1 + EP 组合,其他 MoE 后端不受影响。PR 未注明新增自动化测试。

## 关联

前序/阻塞:PR #29804(合并后对 `flashinfer_trtllm_fp4_routed_moe` 加同样优化)。同区域:flashinfer trtllm MoE 相关(PR #21525 为 decode attention 侧);PR 未注明其他编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
