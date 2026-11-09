# PR #9660: Single Batch Overlap for MoE Models

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Sulfur6 |
| 合并 | 2025-12-03 |
| 分类 | MoE |
| Label | deepseek, sgl-kernel, run-ci |
| Diff | diffstat 缺失(素材未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/9660 |

## 一句话结论

为 DeepSeek V3/R1 decode 引入 SBO:重叠 Shared Expert/Dispatch Recv 与 Down GEMM/Combine Send,提升小 batch 吞吐。

## 动机

Two-Batch Overlap(TBO)在低算力卡(H20)的 Decode 阶段收益不佳:Hopper 上 WGMMA 的 block_m 为 64,小 decode batch 下启用 TBO 会让 MLP GEMM 产生冗余计算,只有在较大 batch(如 64、128)才观察到正吞吐收益;而那种 batch 下 H20 已无法满足 TPOT/ITL 的 SLA。因此需要一个在小 batch 也能提升 Decode 吞吐的方案,即 Single Batch Overlap(SBO)。

## 优化点

SBO 在 DeepSeek-V3/R1 的 MoE 层做两处 overlap:Shared Expert 计算与 Dispatch Recv 重叠;Down GEMM 计算与 Combine Send 通信重叠。后者是 producer-consumer 信号模型:每个 local expert 按 block_m=64 个 token 分配一个 signal unit,Down GEMM 每算完一段就原子递增,Combine Send 轮询到阈值后发送对应 block_m 的结果。实现依赖 DeepEP(antgroup-opt 分支,deepseek-ai/DeepEP#483)与 DeepGEMM(sgl-project/DeepGEMM#14)的配套修改,Dockerfile 在 `HOPPER_SBO=1` 时安装固定 commit。SGLang 侧用 `--enable-single-batch-overlap` 开启(要求 `--moe-a2a-backend deepep`、`--moe-runner-backend deep_gemm`),MoE runner 经 running_state 传递 down_gemm/meta overlap args,DeepGEMM masked grouped GEMM wrapper 增加 enable_overlap/signal/max_block_n;非 Blackwell 上 `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` 默认降为 3,combine signal 改为按 block_m 分槽的 int32 张量。

## 关键改动

```text
python/sglang/srt/batch_overlap/single_batch_overlap.py: SboFlags 放开 Hopper(deep_gemm 且非 Blackwell)路径;combine_signal 按 block_m=64 分槽;CombineOverlapArgs 增加 block_m
python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py: masked grouped GEMM 接受 overlap_args(enable_overlap/max_block_n/signal)
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py: down GEMM 读取 overlap args;H20 且 expected_m<=64 时 max_block_n=160;回传 block_m/threshold
python/sglang/srt/layers/moe/token_dispatcher/deepep.py: 新增 DeepEPPDispatchHooks;combine 按 is_blackwell 传 comp_signal/block_m/threshold/num_sms
python/sglang/srt/models/deepseek_v2.py: forward_deepep 注册 dispatch/post_dispatch/post_combine hook 实现 Shared Expert 与 Dispatch 重叠
docker/Dockerfile: HOPPER_SBO=1 时安装 antgroup-opt 分支 DeepEP
```

```python
@classmethod
def enable_combine_down_gemm_two_stream_overlap(cls):
    return (
        is_sbo_enabled()
        and (
            get_moe_runner_backend().is_flashinfer_cutedsl()
            or (get_moe_runner_backend().is_deep_gemm() and not is_blackwell())
        )
    )
```

## 性能声明

5 节点 × 8×H20(3 个 prefill 节点 TP8,2 个 decode 节点 DP-Attn 16 + EP 16),输入 4096/输出 1536,PD 分离,deepep + deep_gemm,bs 32(rate 4.8/5.0 两轮):
- origin:request throughput 4.34/4.34 req/s,output token throughput 6667.06/6670.89 tok/s,mean ITL 73.11/73.13 ms。
- SBO:request throughput 4.63/4.67 req/s,output token throughput 7111.38/7168.55 tok/s,mean ITL 67.35/67.68 ms。
- GSM8K 1000 题:origin 0.951,SBO 0.950。

## 验证与回归风险

精度以 GSM8K bench_sglang 验证(0.951→0.950)。启用条件严格:Hopper + DeepSeek V3/R1 + deepep + deep_gemm;Blackwell 上 flashinfer_cutedsl 的 combine/down-gemm overlap 行为保留。Hopper 路径改变 combine signal 布局(每 expert 一个 uint32 → 按 block_m 分槽 int32)与 DeepEP combine 参数,依赖 DeepEP 特定 commit 与 DeepGEMM#14,默认 Docker 构建不受影响(`HOPPER_SBO=1` 才启用);`SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` 默认值在非 Blackwell 上从 32 变为 3,影响所有 SBO+非 Blackwell 组合。

## 关联

#10422(此前添加的 `--enable-single-batch-overlap` 参数)、two_batch_overlap.py(TBO);配套外部改动 DeepEP#483、DeepGEMM#14。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
