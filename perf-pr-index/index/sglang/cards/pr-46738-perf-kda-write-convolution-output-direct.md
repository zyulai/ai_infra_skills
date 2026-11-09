# PR #46738: [Perf] kda: Write convolution output directly into contiguous QKV planes

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-11-02 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | run-ci, jit-kernel, run-ci-extra, run-ci-nv-extra |
| Diff | pack diffstat 缺失;diff 含新测试 test_causal_conv1d_split_output.py 及卷积/后端/graph 三处修改 |
| 链接 | https://github.com/sgl-project/sglang/pull/46738 |

## 一句话结论

KDA 卷积直接把输出写入连续 Q/K/V 平面,消除 FlashInfer prefill 的三次 layout 拷贝,16K-token 该路径快 2.66x。

## 动机

KDA 已在一次调用中计算 packed Q/K/V 卷积,但其 token-major 输出使三个 channel slice 非连续,FlashInfer prefill 路径在消费前必须物化三份拷贝。在 GLM-5.3-Flash TP4 16K-token shape 下,这些拷贝的开销超过卷积本身。

## 优化点

给现有 Triton 卷积(`causal_conv1d_fn`)增加可选 split-output layout:卷积在单次分配中直接存入连续 Q/K/V 平面,以运行时 total-token count 作为 plane pitch(kernel 内按 `idx_feats // OUTPUT_SPLIT_SIZE` 选择平面、`idx_feats % OUTPUT_SPLIT_SIZE` 定列),算术与卷积状态更新完全不变。启用条件:eager KDA prefill(`kda_backend.py::_convolve_prefill`)与 captured KDA prefill(`kda_prefill_graph.py::kda_extend_in_graph`)都仅在 FlashInfer 后端且 Q/K/V 宽度相等时传 `output_split_size=layer.q_dim`;其他后端与不等宽度保留原 layout 与 split。无 server 参数、graph size、精度、MoE 或内存预算改动,且不依赖单独的 empty-view fix #46737 或社区 PR #45369/#45583。plane pitch 不做 per-shape specialization,避免每次 ragged prefill 触发重编译。

## 关键改动

```text
python/sglang/kernels/ops/mamba/causal_conv1d_triton.py: kernel 增加 OUTPUT_SPLIT_SIZE 存储布局与 output_split_size 参数
python/sglang/srt/layers/attention/linear/kda_backend.py: _convolve_prefill 仅 FlashInfer+等宽时 split 输出,返回 (Q,K,V) 元组
python/sglang/srt/layers/attention/linear/kda_prefill_graph.py: 图捕获路径同样启用 split 输出
test/registered/kernels/ops/mamba/test_causal_conv1d_split_output.py: 新增 bit-exact 回归测试
```

```python
if OUTPUT_SPLIT_SIZE:
    # Store each channel group as a contiguous [tokens, channels] plane.
    o_ptrs = (
        o_ptr
        + (idx_feats // OUTPUT_SPLIT_SIZE).to(tl.int64)
        * total_tokens * OUTPUT_SPLIT_SIZE
        + output_token * OUTPUT_SPLIT_SIZE
        + idx_feats % OUTPUT_SPLIT_SIZE
    )
else:
    o_ptrs = o_ptr + output_token * stride_o_token + idx_feats * stride_o_dim
```

## 性能声明

单 B300,BF16 输入,FP32 width-4 weights,no bias,local Q/K/V width 各 2048,input row stride 6416;CUDA-event 中位数(9 samples,交替 baseline/candidate 顺序),每 sample 含 100 次 CUDA-graph replay 调用;两臂使用同一 patched operator(原 layout + 三次 contiguous copies vs split-output):

| Total tokens | 原卷积 + Q/K/V copies | Split-output 卷积 | Speedup |
| ---: | ---: | ---: | ---: |
| 1024 | 24.945 us | 10.272 us | 2.43x |
| 8192 | 180.952 us | 68.585 us | 2.64x |
| 16384 | 347.838 us | 130.748 us | 2.66x |

PR 明确注明:以上是 local path 测量,非 full-model latency 或 request throughput;并非"卷积 kernel 本身快 2.66x",收益来自消除三次 layout 拷贝;无 combined serving-throughput 结果归属本 PR。

## 验证与回归风险

新增 `test_causal_conv1d_split_output.py`:bit-exact 输出与状态、不等请求长度、strided input、padded block-table rows、changed-input graph replay(5 tests + 12 subtests,B300);现有 FlashInfer KDA graph tests 按原数值容差通过。Cleanup validation(2026-10-29):CPU fixture 5 tests + 11 subtests(GDN、Triton KDA、等宽 FlashInfer、不等宽 fallback);RTX PRO 6000 Blackwell(SM120)4 tests + 8 subtests,4 个 FlashInfer graph tests 因需 SM100/SM103 被 skip(不计通过)。PR 注明 validation limitation:孤立模型级 accuracy 未完成——早前 GLM-5.3-Flash 实验(合并 empty-view fix)GSM8K 1281/1319 vs baseline 1286/1319,不能确立本 patch 单独的准确性结论。风险面:仅 FlashInfer 等宽路径改变布局,SM120 上 FlashInfer graph 路径未被 CI 覆盖。

## 关联

#32474(以独立 output-layout kernel 打包卷积)、#34299(native checkpoint backend)、#45992(Triton KDA normalization 路径处理拷贝)、#46737(empty-view fix,本 PR 不依赖)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
