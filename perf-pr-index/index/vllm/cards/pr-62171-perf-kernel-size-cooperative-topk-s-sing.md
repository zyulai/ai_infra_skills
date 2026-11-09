# PR #62171: [Perf][Kernel] Size cooperative_topk's single pass by row length, not stride

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @haosdent |
| 合并 | 2026-10-23 |
| 分类 | Speculative Decoding(major) |
| Label | ready |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/62171 |

## 一句话结论

cooperative_topk 按行实际长度而非 logits 行 stride 选择单 pass 路径,宽 buffer 下的 decode top-k 最高快约 8%。

## 动机

DSA indexer 用 `max_model_len` 列数分配 decode logits,`cooperative_topk` 拿到的 stride 远大于行实际长度(`seq_len`)。此前 kernel 的 `per_block`(每个 CTA 处理的元素数)按 stride 计算:长 `max_model_len` 下每块元素数超过单 pass staging 上限,即使行上下文本身很小,所有行都被迫走两遍 TMA 流式读取的 two-pass 路径。例如 16 行用 4-CTA cluster,256K stride 得到每块 64K 元素,超过 48K 单 pass 上限,而 40K token 的行每块只需 10K。

## 优化点

`per_block` 改为 `ceil(seq_len / CS)`,即按每行自身长度分块,让每行独立选择 single-pass 或 streaming 路径;同一次 launch 中不同行可走不同路径,barrier setup 已跟随所选路径。`large_topk` 按 16-byte 单元切分行,而单 pass 上限(`kFusedMax`)与 stage 大小均为该单元的倍数,因此路径与 stage 数与精确切分一致。两条路径计算的是同样的精确 radix top-k,选择结果不变。

## 关键改动

```text
csrc/cooperative_topk.cuh: per_block 由 stride 改为按行长度 sl 计算
tests/kernels/test_top_k_per_row.py: 新增 test_cooperative_topk_rows_shorter_than_stride
```

## 性能声明

B300 microbenchmark(k=2048,rows=requests x 4 MTP,CUDA graph replay):4 行/40K context/1M stride 10.99→10.27 µs,16 行/40K/1M 12.18→11.24 µs;作者称 decode top-k 在行适配单 pass staging 且 stride 大于 per-block staging 时最高快 8%(如 max_model_len 1M、1-8 行,或约 200K、9-64 行)。InferenceX AgentX,GLM-5.3-NVFP4,4x GB200,与 #62170、#62172 合并测量:decode time per turn c1/c4/c8/c16 分别 -2.8% / -2.5% / -1.5% / -1.5%(本 PR 约占一半);`vllm bench forward`(GLM-5.2-NVFP4,TP4,B300,基于 #62170)在 1/3/7 请求、25K/94K context 下 -0.58% 至 -1.36%。MRCR 8 needles:0.903 → 0.890,配对差 -0.013 [-0.038, +0.012]。

## 验证与回归风险

测试:`tests/kernels/test_top_k_per_row.py` 265 passed/37 skipped(GB200 SM100);rebase 到 #59645 后 402 passed/44 skipped(B300 SM103)。新测试覆盖所有 cluster size(1/4/8/33/64 行)、64K 与 1M stride、300 至 300K 行长,单次 launch 混合 trivial/register/single-pass/streaming 行,与 `torch.topk` 精确比对(atol=0)。回归风险:仅 SM90+(非 SM12x)的 cooperative_topk 路径;按行长度分块的 16-byte 单元对齐假设依赖 kFusedMax/kSizePerStage 为单元倍数;精度由 exact radix top-k 保持,无精度回归声明。

## 关联

同一组测量中的姊妹 PR:#62170、#62172(gain 拆分约 1/6、1/2、1/3)、#62173、#62174(含 FlashInfer 依赖 #6630/#6631/#6632);本 PR rebase 依赖 #59645 的 overflow recovery。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
