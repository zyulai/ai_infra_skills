# PR #37695: [Perf] Use torch compile to fuse pack topk in trtllm moe

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @wzhao18 |
| 合并 | 2026-03-27 |
| 分类 | MoE (tier: major) |
| Label | performance, ready, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/37695 |

## 一句话结论

用 torch.compile 融合 trtllm MoE 的 pack topk ids/weights,MiniMax-M2.5 TP2 并发 64 1K/1K 下 FP8/NVFP4 约 +2%。

## 动机

trtllm MoE 在调用 FlashInfer kernel 前需要把 `topk_ids` 与 `topk_weights` 打包进单个 int32 tensor(格式 `(expert_id << 16) | weight_bf16.view(int16)`)。此前 FP8 与 NVFP4 两个 expert 实现各自内联这段打包表达式,由若干逐元素 op 组成。PR 把它抽成公用函数并用 torch.compile 融合,减少这些小 op 的开销。

## 优化点

在 `vllm/model_executor/layers/fused_moe/utils.py` 新增 `trtllm_moe_pack_topk_ids_weights`,用 `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)` 装饰,内部保留原打包表达式 `return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(torch.int16)`。随后 `trtllm_fp8_moe.py` 与 `trtllm_nvfp4_moe.py` 的 `apply` 都改为调用该函数,删除各自内联的位运算打包,保持送给 `trtllm_fp8_block_scale_routed_moe` / `trtllm_fp4_block_scale_routed_moe` 的 `packed_topk_ids` / `packed_tensor` 格式不变。改动仅涉及打包路径:两个 `apply` 中原注释均说明 `trtllm_fp8_block_scale_routed_moe` / `trtllm_fp4_block_scale_routed_moe` 不支持 autotuning,故在 dummy run 阶段跳过对应 kernel;本次替换保留这两处跳过逻辑与注释,不动 MoE kernel 本身。FP8 侧打包结果变量名为 `packed_topk_ids`,NVFP4 侧为 `packed_tensor`,二者送入 FlashInfer routed MoE 前的格式不变。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/utils.py: 新增 @torch.compile(dynamic=True) 的 trtllm_moe_pack_topk_ids_weights,把位运算打包融合为编译 kernel
vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py: apply 内联打包替换为调用公用函数
vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py: apply 内联打包替换为调用公用函数
```

关键代码(新增的 torch.compile 融合打包函数):

```python
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def trtllm_moe_pack_topk_ids_weights(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """
    Pack topk_ids and topk_weights into a single int32 tensor.
    Format: (expert_id << 16) | weight_bf16.view(int16)
    """
    return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(
        torch.int16
    )
```

## 性能声明

正文 Purpose 称 "From benchmarking, this gives ~2% speedup in Minimax M2.5 TP=2 Concurrency 64 1K/1K across FP8 and NVFP4"。Test Result 条件:`vllm serve MiniMaxAI/MiniMax-M2.5`(NVFP4 臂为 `MiniMax-M2.5-NVFP4` 私有 checkpoint)`--trust-remote-code --stream-interval 20 --tensor-parallel-size 2`,`vllm bench serve --dataset-name random --random-input 1024 --random-output 1024 --num-prompt 1280 --max-concurrency 64 --num-warmups 100 --ignore-eos`(PR 正文未注明硬件与精度)。FP8:Main Output token throughput 2716.55 tok/s、Total 5433.09 tok/s、duration 482.50 s、Mean TPOT 23.19 ms;Branch 2764.70 tok/s、Total 5529.39 tok/s、474.09 s、Mean TPOT 22.71 ms。NVFP4:Main 4115.97 tok/s、Total 8231.94 tok/s、318.45 s、TPOT 15.28 ms;Branch 4192.77 tok/s、Total 8385.54 tok/s、312.61 s、TPOT 14.90 ms。Accuracy(gsm8k):FP8 0.9303 / 0.9287(flexible / strict);NVFP4 0.9257 / 0.9212。两臂均 1280 successful / 0 failed。其余 serving 指标:FP8 Peak output token throughput 173.00 → 199.00 tok/s、Mean TTFT 390.04 → 468.97 ms、Median TTFT 350.18 → 373.37 ms、Mean ITL 456.29 → 446.70 ms、Median ITL 446.20 → 437.64 ms、P99 ITL 1108.22 → 1055.59 ms、Peak concurrent requests 128.00 → 127.00;NVFP4 Peak output token throughput 254.00 → 261.00 tok/s、Mean TTFT 284.88 → 386.67 ms、Median TTFT 231.55 → 413.34 ms、Mean ITL 300.62 → 293.09 ms、Median ITL 293.73 → 288.91 ms、P99 ITL 737.18 → 582.89 ms。

## 验证与回归风险

Test Plan 为用 minimax m2.5 的 fp8 与 nvfp4(私有 checkpoint)做精度与性能测试;正文给出上述 serving benchmark 与 gsm8k 表格,未列专门单元测试命令。风险面:打包函数被 FP8 与 NVFP4 两个 trtllm MoE expert 共用,`@torch.compile(dynamic=True)` 引入编译缓存行为;正文未列其他模型或量化组合的回归测试。

## 关联

无(PR 正文未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
