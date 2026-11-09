# PR #4232: [Feature] Integrate DeepEP into SGLang

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @liz-badada |
| 合并 | 2025-03-19 |
| 分类 | MoE(major) |
| Label | high priority |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/4232 |

## 一句话结论

把 DeepSeek 的 DeepEP all-to-all 通信库接入 SGLang MoE,新增 `--enable-deepep-moe` 路径;当前性能尚低于既有 EP(token permutation 未优化)。

## 动机

DeepSeek-V3 等大规模 MoE 模型在 DP attention 下需要高效的专家并行 token 分发。DeepEP(https://github.com/deepseek-ai/DeepEP)是 DeepSeek 开源的 MoE dispatch/combine 通信库,提供 NVSHMEM 之上的 intranode/internode all-to-all 与低延迟模式;本 PR 将其接入 SGLang 框架,使 DP-attention + EP-MoE 可以使用 DeepEP 的通信路径。标注 WIP。

## 优化点

机制上新增强 `--enable-deepep-moe`(强制绑定 `--enable-dp-attention`,model_runner 启动时断言):新增 `DeepEPDispatcher`(改编自 Megatron-Core 的 `MoEFlexTokenDispatcher`)维护全局单例 `deep_ep.Buffer`——normal 模式按 dispatch/combine config hint 预算 NVL/RDMA 字节,做 `get_dispatch_layout → dispatch → combine`;low-latency buffer(`get_buffer_low_latency`,decode 用,含 H20 cooperative launch 的已知 workaround 注释)已准备好但暂未启用(Todo)。permute/unpermute 取自 Megatron-Core,新增 `deepep_run_moe_deep_preprocess`(stable sort topk_ids、`searchsorted` 求 seg_indptr、用 `num_minus_one` 修正无效专家的 src2dst)与 `deepep_permute_triton_kernel`/`deepep_post_reorder_triton_kernel` 两个 Triton kernel;新增 `DeepEPMoE(EPMoE)` 用 `GroupedGemmRunner` 跑 normal 前向,`forward_deepgemm_masked` 的 deep_gemm masked grouped GEMM 路径留作 decode/低延迟 dispatch 之后的 TODO。另修 weight loading:`load_merged_column_weight` 显式接收 `tp_rank/tp_size` 以支持预切分权重。

## 关键改动

```text
python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py: 新增文件;DeepEPDispatcher + normal/low-latency Buffer 管理 + permute/unpermute
python/sglang/srt/layers/moe/ep_moe/kernels.py: 新增 deepep_run_moe_deep_preprocess 与 permute/post_reorder Triton kernel
python/sglang/srt/layers/moe/ep_moe/layer.py: 新增 DeepEPMoE(normal grouped GEMM + deepgemm masked 占位)
python/sglang/srt/model_executor/model_runner.py / schedule_batch.py: enable_deepep_moe 开关联动与 DP-attention 断言
python/sglang/srt/layers/parameter.py / linear.py: 权重加载传入 tp_rank/tp_size(预切分权重支持)
```

```python
def deepep_run_moe_deep_preprocess(topk_ids, num_experts):
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(num_experts + 1, ...)
    torch.searchsorted(reorder_topk_ids, expert_ids, out=seg_indptr)
    num_minus_one = seg_indptr[0]           # 无效专家(-1)占位计数
    seg_indptr = seg_indptr - num_minus_one
    ...  # src2dst: dst_id - num_invalid 对齐到剔除无效后的槽位
    return reorder_topk_ids, src2dst, seg_indptr
```

## 性能声明

PR 明确说明"Current performance is below expectations as token permutation is not yet optimized"(permute triton kernel 有 bug,暂回退 PyTorch 原生 permute):

- H20,DeepSeek-V3,TP8/DP8,`--disable-cuda-graph --disable-radix-cache`,bench_serving random 1000 in/1000 out,512 prompts,并发 128:DP Attn + DeepEP 输入/输出吞吐各 436.69 tok/s、总计 873.38 tok/s;对照组 DP Attn + EP 为 617.90/617.90、总计 1235.79 tok/s(DeepEP 路径当时慢约 29%)。

无精度评测结果;正文以单条 chat completion 验证单机(TP8/DP8)与双机(TP16/DP16)功能可用。

## 验证与回归风险

验证方式为单机/双机 server 启动 + OpenAI 接口冒烟(test_openai.py),DeepEP 需按其 NVSHMEM 依赖安装指南部署(提供 Dockerfile.deepep)。已知限制:必须 `--disable-cuda-graph`(CUDA graph 捕获报 "capturing stream has unjoined work");low-latency dispatch/combine 未启用,decode 走 normal dispatch;H20 上 low-latency 需修改 DeepEP `internode_ll.cu` 的 warp 配置才能工作。回归风险:DeepEPMoE 的 dynamic 量化 scale 每次 forward 重算 max;`enable_deepep_moe` 与 radix cache/chunked prefill 等组合未经正文验证。

## 关联

本 PR 是 SGLang DeepEP 系列的起点(后续 TBO、低延迟 decode 等工作在其上叠加);对照基线为既有 `--enable-ep-moe` 路径。pack 未列出具体后续 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
