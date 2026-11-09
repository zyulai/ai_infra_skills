# PR #12921: [perf]optimize w4afp8 kernel on deepseek-v3-0324

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Bruce-x-1997 |
| 合并 | 2025-12-18 |
| 分类 | Quantization |
| Label | run-ci, sgl-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与 changed files 数),diff 正文可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/12921 |

## 一句话结论

细调 DeepSeek-V3-0324 w4afp8 CUTLASS MoE 的 SM90 tiling 并优化元数据 kernel,decode 小批量吞吐约 +5%、prefill 约 +10%。

## 动机

线上用 w4afp8 量化跑 DeepSeek-V3-0324 时,decode batch size < 32 的场景性能不佳。该路径走 `cutlass_w4a8_moe` 的 SM90 grouped GEMM,tile 配置按 m 粗分桶,小批量档位覆盖不足;同时 group GEMM 前的 problem sizes / expert offsets 元数据 kernel 用 atomicAdd 与单线程前缀和,也是小批量下的额外开销。

## 优化点

一是重排 `dispatch_w4a8_moe_mm_sm90` 的 tile 分桶:在 n=4096/k=7168 与 n=512/k=7168 两个 group gemm 中把 decode 档从 `m<=16` 扩到 `m<=32`(仍用 `SM90_CO<128,16,512,2,1,1>`),并新增 m<=4096 的 prefill 档(`SM90_CO<128,64,512,2,1,1>`,注释注明面向 seq_len 4096/8192);fallback 分支按 m<=32/1024/更大 区分 decode 与 prefill 配置。二是重写元数据 kernel:`compute_problem_sizes_w4a8` 对对齐的 topk_ids 用 int4 向量化读取统计 expert 出现次数,用 `cub::BlockReduce` 替代 atomicAdd 累加,删除 atomic_buffer 分配;expert offsets 用 `cub::BlockScan` 的分块 inclusive prefix sum 并行化,替换原单线程循环,并按 num_experts 选择 32~1024 的 block size。

## 关键改动

```text
sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu: SM90 tile 分桶重调 + INVOKE_GEMM_WITH_CONFIG 宏去重
sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu: int4 向量化计数 + cub BlockReduce/BlockScan
python/sglang/srt/layers/quantization/w4afp8.py: apply 传递逻辑微调
```

```cpp
// Optimized: vectorized memory access using int4 for better memory bandwidth
if (aligned) {
  const int4* vec_ptr = reinterpret_cast<const int4*>(topk_ids);
  for (int i = threadIdx.x; i < vec_length; i += BLOCK_SIZE) {
    int4 vec_data = vec_ptr[i];
    occurrences += (vec_data.x == expert_id) + (vec_data.y == expert_id)
                 + (vec_data.z == expert_id) + (vec_data.w == expert_id);
  }
}
```

## 性能声明

DeepSeek-V3-0324 w4afp8,TP4,flashinfer attention(precise GPU 型号 PR 未注明,tile 配置面向 SM90/Hopper):Prefill(8192 token 批)run dt 566.5~567.1 ms → 500.2~508.2 ms,约 10%;Decode(#running-req 18~20,cuda graph)gen throughput 482~484 tok/s → 553~565 tok/s,约 5%。bench_serving:case1(20 prompts,100-in/1000-out,cc20)output tok/s 354.28→364.82;case2(1000 prompts,1000-in/1-out)input tok/s 9112.65→9856.97,Mean TTFT 544.65→503.63 ms。精度:AIME25 0.4/0.4,AIME24 0.5/0.65/0.55,MMLU 0.8947。

## 验证与回归风险

验证以线上日志对比与 bench_serving e2e + 精度评测为主,pack 中未见新增单测;checklist 各项未勾选。风险面:tile 分桶按 (n,k) 形状硬编码(n=4096/7168、7168/2048、512/7168、7168/256 等 DeepSeek-V3 形状),其他模型形状走 fallback 分支的解码/预填充分档;元数据 kernel 从 atomic 改为 cub 归约属确定性等价替换。PR 基于 #10027 的 w4a8 实现。

## 关联

基于 #10027。同区域(w4afp8/deepseek quant kernel)其他 PR 素材未提供。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
