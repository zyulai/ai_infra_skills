# PR #59082: [CPU] Improve long-context MTP attention scheduling on AMX

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @tianmu-li |
| 合并 | 2026-10-21 |
| 分类 | Speculative Decoding(tier: major) |
| Label | performance, cpu |
| Diff | diffstat 缺失(素材未提供 +/− 统计);diff 素材截断(3 个文件未展示) |
| 链接 | https://github.com/vllm-project/vllm/pull/59082 |

## 一句话结论

CPU AMX attention 的 GQA 分组调度扩展到 MTP 验证等多 token 请求,长上下文 MTP 吞吐提升约 18%。

## 动机

CPU attention 已有的 batch 级 GQA fast path 要求整个 batch 全是 q=1 decode,因此 MTP 验证、短 prefill 这类多 token 请求拿不到 grouped 调度,长上下文 MTP 场景损失明显。此前 #46154 只修了混合 batch 中把 prefill 拆成单 token 的分类问题,没有覆盖 request 级的资格判定。

## 优化点

调度器按每个 request 的 query 长度、因果性与 AMX 几何评估 grouped query heads 与 split-KV 工作量,再经 cost model 选计划:可整除分组的请求允许任意 query 长度打包,其余只打包单次迭代能放下的 query,超大分组回退 MHA。工作项结构 AttentionWorkItemGroup 增加 kv_head_idx/q_head_start/q_head_num 并按 16 字节对齐;调度输出改为 Partitions + AttentionTaskSpan,替代原 cu_workitem_num_per_thread 前缀和;reduction scratchpad 改为全局偏移并按 cache line 分隔各 split 的 max/sum 统计。适用于 auto、FP8 E4M3、FP8 E5M2 KV cache。

## 关键改动

```text
csrc/cpu/cpu_attn_impl.hpp: request 级 GQA 分组资格、Partitions/task span 调度重构、scratchpad 布局
benchmarks/kernels/cpu/benchmark_cpu_attn.py: 透传 kv_cache_dtype
(另有 3 个文件在素材 diff 中被截断,未展示)
```

```cpp
const auto request_q_head_group = [&](const int32_t req_id) {
  if (original_q_head_per_kv > max_num_q_per_iter) return 1;
  if (max_num_q_per_iter % original_q_head_per_kv == 0 ||
      request_q_token_num(req_id) * original_q_head_per_kv <= max_num_q_per_iter)
    return original_q_head_per_kv;
  return 1;
};
```

## 性能声明

Qwen/Qwen3.5-4B,BF16 TP1,batch size 1,无 prefix caching,100k 输入 / 1k 输出(benchmark_serving random,--ignore-eos),取 3 次请求中位数:

| 配置 | main tok/s | 分支 tok/s | main TPOT | 分支 TPOT |
| --- | --- | --- | --- | --- |
| No MTP, auto KV | 6.73 | 7.08 | 79.29 ms | 75.49 ms |
| MTP3, auto KV | 8.01 | 9.48 | 48.72 ms | 34.33 ms |
| MTP3, FP8 KV(修复后) | 8.62 | 9.36 | 37.25 ms | 27.57 ms |

FP8 分支首次实现 TTFT 87.14 s(新 materializer 误处理普通 FP8 prefill),改走 legacy materializer 后 TTFT 81.20 s,与 main 差 0.58% 以内;FP8 MTP acceptance 98.7%(767/777)。

## 验证与回归风险

tests/kernels/attention/test_cpu_attn.py、tests/v1/attention/test_group_head_counts.py(原测试 1,637 passed);MRCR 正确性:Qwen3.5-4B MTP3 auto KV 40/40 通过且分数与 main 完全一致(0.9952/0.8444/0.7632);FP8 KV 40/40,n8 needle 0.7055 仍低于 0.7100 阈值(与 main 同样不达标,非本 PR 引起)。风险:改动集中在 CPU attention 调度,影响所有 CPU attention 路径(含非 MTP);native amx_fp8 与 ARM 硬件未实测,rebase 后未重跑完整 kernel 计时矩阵。

## 关联

相关:#46154(混合 batch GQA decode 分类修复,PR 内声明非重复)、issue #46131。同分类 Speculative Decoding 卡:#34457(sparse MLA+MTP full cudagraph)、#63873(bench 投机状态)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
