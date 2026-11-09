# PR #31781: [Kernel] Support bias type in grouped_topk kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @xyang16 |
| 合并 | 2026-01-07 |
| 分类 | Attention / Kernel (tier: major) |
| Label | performance, ready, deepseek, nvidia |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files);pack 末尾无 "file(s) dropped from diff" 说明,diff 完整保留 3 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/31781 |

## 一句话结论

grouped_topk kernel 支持独立 fp32 bias 类型以消除 cast,DeepSeek-R1-0528 TP8 e2e 输出吞吐 +5%。

## 动机

`grouped_topk` CUDA kernel 此前只支持 bias 与 input 同 dtype,但 DeepSeek-V2 里 `e_score_correction_bias` 恒为 fp32(正文引 `deepseek_v2.py` L269-270),因此在 `fused_moe.py` L1382 被 cast 成 input dtype。该 cast 由 torch.compile 融合成 `triton_poi_fused__to_copy_0` kernel,profiling 显示额外耗时 1.6 µs。PR 让 kernel 模板的 bias 类型独立于 input 类型,从而去掉这次 cast。

## 优化点

把 `topk_with_k2` 从 `template <typename T, ScoringFunc SF>` 改为 `template <typename T, typename BiasT, ScoringFunc SF>`,bias 指针由 `T const*` 改为 `BiasT const*`,并在读取处用 `static_cast<T>(bias[i])` 显式转换(experts-per-group 大于 / 不大于 WARP_SIZE 的两条 top-2 分支都改)。这样 fp32 bias 可直接传入 kernel,无需在 Python 侧提前 cast 到 input dtype,也就不再触发 `triton_poi_fused__to_copy_0`。配套更新 `fused_moe.py` 的调用与 `test_grouped_topk.py` 用例(含 scoring_func=0 的已算分路径)。

## 关键改动

```text
csrc/moe/grouped_topk_kernels.cu: 核心。topk_with_k2 增加 BiasT 模板参数,bias 以 BiasT const* 传入并 static_cast<T> 后参与打分
vllm/model_executor/layers/fused_moe/fused_moe.py: 传 fp32 e_score_correction_bias,去掉 input dtype 的 cast
tests/kernels/moe/test_grouped_topk.py: 覆盖独立 bias 类型的 grouped_topk 调用
```

关键代码(bias 以独立类型参与打分):

```cuda
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);
```

对应模板签名改为 `template <typename T, typename BiasT, ScoringFunc SF>`、`BiasT const* bias`。

## 性能声明

正文声明 "e2e output token throughput improves 5%"。Benchmark 条件:`vllm serve deepseek-ai/DeepSeek-R1-0528 --tensor-parallel-size 8 --max-num-seqs 8 --trust-remote-code --compilation_config '{"compile_sizes": [1,2,4,8]}'`,`vllm bench serve --dataset-name sharegpt --sharegpt-output-len 300 --max-concurrency 8 --num-prompts 1000 --num-warmups 20 --ignore-eos`(PR 正文未注明硬件与精度)。Main:Output token throughput 473.91 tok/s、Total 836.32 tok/s、Request throughput 1.58 req/s、Mean TTFT 339.07 ms、Mean TPOT 15.80 ms;PR:498.94 tok/s、Total 880.50 tok/s、1.66 req/s、Mean TTFT 236.31 ms、Mean TPOT 15.30 ms;两臂各 1000 successful / 0 failed。Accuracy(`lm_eval` gsm8k):Main flexible-extract 0.9545±0.0057 / strict-match 0.9530±0.0058;PR 0.9606±0.0054 / 0.9591±0.0055。Profiling:改动前多一个 `triton_poi_fused__to_copy_0`(约 1.6 µs),改动后不再 launch。两臂其余 serving 指标:Peak output token throughput 528.00 → 552.00 tok/s、Median TTFT 289.77 → 259.43 ms、P99 TTFT 4721.53 → 295.77 ms、Median TPOT 15.73 → 15.35 ms、P99 TPOT 16.27 → 15.84 ms、Mean ITL 15.80 → 15.30 ms、Median ITL 15.72 → 15.00 ms、P99 ITL 16.08 → 15.40 ms;两臂 Total input tokens 229418、Total generated tokens 300000、Peak concurrent requests 16.00、Successful requests 1000 / Failed 0 相同,Benchmark duration 633.03 → 601.27 s。

## 验证与回归风险

`pytest -s -v tests/kernels/moe/test_grouped_topk.py`:Tests passed。PR 正文的 Accuracy Testing 另用 `python3 -m lm_eval --model local-completions --model_args model=deepseek-ai/DeepSeek-R1-0528,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 --tasks gsm8k` 对比两臂,数值见性能声明;Test Plan 仅列这一条 pytest 命令,Test Result 仅写 "Tests passed"。风险面:kernel 模板签名变化影响所有 `grouped_topk` 调用方与不同 input / bias dtype 组合,正文未说明新 `BiasT` 模板参数对其他调用方、其他量化 dtype 或其他硬件的影响范围;除 DeepSeek-R1-0528 TP8 的 e2e 吞吐、TPOT / TTFT 与 gsm8k 外,正文未列其他模型或硬件的测试。

## 关联

无(PR 正文仅链接 `deepseek_v2.py`、`fused_moe.py` 的代码行,未引用其他 PR 编号)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
