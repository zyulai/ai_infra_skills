# PR #56301: [ROCm][Perf] W4A16: pad gfx11 weight and activation strides

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mgehre-amd |
| 合并 | 2026-10-06 |
| 分类 | Other (tier: major) |
| Label | performance, rocm, ready |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 1 个 kernel 文件 + 1 个 Python kernel 文件 + benchmark + 测试 |
| 链接 | https://github.com/vllm-project/vllm/pull/56301 |

## 一句话结论

gfx11 上把 W4A16 GEMM 权重行与激活行的 stride 移出 1024/2048 字节 L2 通道混叠悬崖,TTFT 最多降 23%、TPOT 最多降 3%。

## 动机

在 gfx11(RDNA3/3.5)上,当 W4A16 GEMM 某个操作数的行 stride 恰为 1024 字节的倍数时,各行会混叠到相同的 L2/DDR 通道,访问串行化,损失大量本可达成的读带宽;把 stride 偏移一个 128 B cache line 即可避开。packed 权重行(K/2 字节)与激活行(K x 2 字节)各有自己的悬崖(1024 B 与 2048 B),此前都按 dense 连续布局存储,命中即劣化。

## 优化点

双操作数分别处理。权重:新增 `pack_skinny_int4` 作为 skinny 权重布局的唯一入口,在 load 时(`process_weights_after_loading`)对落在 1024 B 悬崖的行一次性 pad 128 B(32 个 int32 列),int8 视图保留非连续 stride,int32 视图在 apply 时恢复 `[N, K//8]`。激活:W4A16 apply 算子内对行 stride 落在 2048 B 悬崖的激活做显式拷贝 pad(每步一次),正文说明该拷贝成本低于 GEMM 收益,decode 路径不 pad 激活以保住 decode 收益。kernel 侧:HIP skinny kernel(`wvSplitK_int4_hf`/`_sml_`)与 Triton prefill kernel 的寻址都从硬编码 `K_packed`/`K`/`K8` 改为按传入的 `B_row_stride_bytes`/`stride_bn`/`stride_am` 计算行偏移,并新增权重字节数与 stride 的 TORCH_CHECK(防 undersized view 越界读)。padding 逻辑由 `_weight_pad_bytes`/`_act_pad_bytes` 门控且仅对 1024/2048 的倍数加一条 cache line,并断言 pad 后不再落回悬崖;整体仅在实测过的 gfx1151 上启用,其他 gfx11 型号留待测量后续开启。

## 关键改动

```text
vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py: 悬崖常量与 _weight_pad_bytes/_act_pad_bytes、pack_skinny_int4(load 时权重 pad)、_pad_activation_rows(apply 时激活 pad)、Triton kernel 接受行 stride
csrc/rocm/skinny_gemms_int4.cu: wvSplitK_int4 系列内核按 B_row_stride_bytes 寻址,入口校验 stride/字节数
tests/kernels/quantization/test_rdna_hybrid_w4a16.py: padding 规则单测、padded vs dense 逐位一致(rtol=0/atol=0)、process_weights 后的 stride 断言
benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py: 改用 do_bench(强制从 DRAM 读权重,避免 MALL 驻留掩盖 stride 效应),权重走 pack_skinny_int4 生产布局
```

关键寻址改动(HIP kernel):

```cuda
const int64_t b_row_stride_bytes = in_a.stride(0) * in_a.element_size();
...
const uint8_t* B_ = &B_packed[(m + 0) * B_row_stride_bytes + k_ / 2];
```

## 性能声明

AMD Strix Halo(gfx1151),`--output-len 128 --num-prompts 10 --max-num-seqs 1 --request-rate 0.3`,median,n 如注:Qwen3-8B-quantized.w4a16 128/128:TTFT 119.99 → 101.80 ms(-15.2%),TPOT 23.91 → 23.20 ms(-3.0%);Qwen3-8B 1920/128:TTFT -8.0%,TPOT -2.5%;Llama-3.1-8B-Instruct-w4a16 128/128:TTFT 131.50 → 101.75 ms(-22.6%),TPOT -2.3%;Llama-3.1-8B 1920/128:TTFT -6.5%;gemma-3-4b-it-quantized.w4a16 1920/128:TTFT -5.9%,TPOT 无效应;Qwen3-4B 1920/128:TTFT -0.8%,TPOT 无效应。视觉模型 gemma-4-31B(AWQ 4-bit,gs32,328 in/256 out + 640x480 图):TPOT -2.83%,加 MTP(3 speculative tokens)后 TPOT -7.78%、TTFT +1.33%。总体表述为 TTFT 最多 23%、TPOT 最多 3%。精度:GSM8K 200 题 5-shot greedy(Qwen3-8B-quantized.w4a16,gfx1151,n=3/臂)两臂均 88.00%,输出 token 数一致(23209-23214)。

## 验证与回归风险

`tests/kernels/quantization/test_rdna_hybrid_w4a16.py` 95 tests 在 gfx1151 全过,其中新增:padding 规则单测(悬崖上才 pad、pad 后不回悬崖、4864 B 行 pad 实测 -14% 故不 pad)、HIP decode 与 Triton prefill 两路径下 row-padded 权重对 dense 权重输出逐位一致(rtol=0, atol=0,fp16/bf16 x 有无 zero-points x M=1/M=MAX+1)、`process_weights_after_loading` 后 stride 断言与 int32 视图存活、undersized weight view 拒绝。风险面:仅 gfx1151 启用,gfx11 其他型号与 gfx12 未测量(代码路径在非 gfx1151 上归约为原行为);激活 pad 引入每步一次拷贝(仅 prefill apply 路径);受影响 shape 覆盖面因模型而异(Qwen3-4B 仅 1/4 GEMM 命中,收益小)。

## 关联

同区域:PR #55090(dense fp16/bf16 权重行 pad 的姊妹 PR,同源 L2 aliasing 问题,不同张量与代码路径);ROCm 量化 GEMM 相关见 pr-41394(ROCm native W4A16 kernel)、pr-59478(AITER FlyDSL INT4 quick reduce)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
