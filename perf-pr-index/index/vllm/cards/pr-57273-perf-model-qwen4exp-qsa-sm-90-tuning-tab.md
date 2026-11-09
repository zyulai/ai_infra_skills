# PR #57273: [Perf][Model] Qwen4Exp QSA: sm_90 tuning table for _select_config

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ShuoleiWang |
| 合并 | 2026-09-17 |
| 分类 | Model-specific Fast Path |
| Label | performance, ready, qwen |
| Diff | diffstat 缺失(pack 未提供) |
| 链接 | https://github.com/vllm-project/vllm/pull/57273 |

## 一句话结论

为 Qwen4Exp QSA 的 Triton split-K sparse attention kernel 增加 sm_90(H100/H200/H20)调优表,fp8 cache 下大并发区域单 kernel 快 31-53%。

## 动机

`_qsa_sparse_paged_gqa_splitk_kernel`(Qwen3.8-Flash-Next QSA 层对每 query 的 2048 个选中 token 做 sparse attention)的 (BLOCK_N, target_splits, num_warps) 由 `_select_config()` 按键 `base_programs = query rows x kv_heads per rank` 选择,默认表在 GB300 上调优(#55557 另加了 sm_120 表);在 H20 上测量发现 GB300 条目在多数 decode 区域是错误的几何配置。

## 优化点

在 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` 新增 `_select_sm90_config()`,与 sm_120 helper 同形,`_select_config()` 按 `_is_sm90()`(device capability == (9,0),lru_cache)分派;默认区域已最优的返回原 GB300 条目(表中标注 unchanged)。选型依据:小 `base_programs` 时 kernel 按 stride 把 tile 交给 split programs,(32,64,4) 会让某个 program 拿到 65 tiles/64 splits 的两 tile 尾巴,launch 需等它,而 64 宽 tile 得到 33 tiles/33 splits 每程序一个;几百 programs 以上 kernel 是 gather-bound,要 resident CTAs——BLOCK_N 64 需 68-76 KB shared memory(3 CTAs/SM),BLOCK_N 32 + 1 warp 只需 35-41 KB(6 CTAs/SM);fp8 的 in-kernel dequant 增加每 tile 工作量,是 64 宽条目在 fp8 下损失近半时间的原因。

## 关键改动

```text
vllm/models/qwen4_exp/nvidia/ops/qsa.py: _is_sm90 + _select_sm90_config(bf16/fp8 两套 base_programs 区间表)
benchmarks/kernels/benchmark_qsa_select_config.py: 单 shape 约 10 秒的复测脚本(在裁剪部分)
tests/models/qwen4_exp/test_qsa_reference.py: select_config_regions 与 sm90_ 相关测试(在裁剪部分)
```

```python
def _select_sm90_config(base_programs, use_prefill_config, is_fp8):
    if base_programs > 2048:
        return (32, 1, 1) if use_prefill_config else (64, 1, 2)
    if is_fp8:
        if base_programs <= 24: return 64, 64, 2
        ...
        return 32, 4, 1
    if base_programs <= 32: return 64, 64, 2
    ...
```

## 性能声明

单 kernel 延迟变化(H20,CUDA-graph replay,3 次交错序 x 9 replays 中位数,hot 与 L2-evicted;按 group 3/6/12 三种 head 几何):bf16 cache 各区域 -4% 到 -21%(如 ≤24 区域 4 行 -11%/-12%/-14%,16 行 -12%/-19%/-21%;65-128 区域 -14%/-13%/-9%);fp8 cache:≤24 区域 -8% 到 -22%,257-512 区域 -43%/-43%/-31%,513-2048 区域 -53%/-45%/-43%。sweep 示例:512 行 group 3 fp8 330 vs 577 µs(GB300 条目慢 75%)。e2e:该 kernel 占此模型 decode step 的 3-9%,bf16 cache 下端到端吞吐变化远小于 1%;fp8 cache 约为 C128 decode step 的 6%。serving(8x H20,TP8/EP8,MTP3,bf16,bf16 K/V):C1 263.6 → 259.1 tok/s、C128 1649.0 → 1674.8 tok/s(在 1-3% 的本机 stock-vs-stock 漂移内,仅证明无回归)。GSM8K(0-shot chat,300 题):85.0% ± 2.1 → 86.3% ± 2.0。

## 验证与回归风险

单测:select_config_regions/sm90 相关 11 passed,全文件 83 passed;每个候选通过 FP32 参考的 `rtol=atol=2e-2`;任何几何上无区域回归超出 run-to-run noise。风险:split count 变化改变 split-K 归约顺序,greedy 输出在近平局处可能分歧(每个分歧点 top-1/top-2 margin 0-0.75 logprob,另一候选 token 在 top-5 内);所有延迟测量在 H20 上,H100/H200 未实测;prefill/decode >2048 区域部分条目未测量(沿用默认)。

## 关联

#55557(前序:加 sm_120 表与 fp8 路径)、#55430(tile-union prefill kernel on SM121)、#55122(deterministic persistent top-k)——同区域正交工作。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
