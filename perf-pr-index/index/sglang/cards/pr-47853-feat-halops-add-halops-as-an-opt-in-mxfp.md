# PR #47853: [Feat] halops: Add HALOps as an opt-in MXFP8 GEMM backend for Blackwell and support mxfp8 megamoe

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @Wanglongzhi2001 |
| 合并 | 2026-11-04 |
| 分类 | MoE(tier: major) |
| Label | documentation, quant, run-ci |
| Diff | diffstat 缺失(素材未提供);diff 实际涉及约 10 个文件(素材另注明 5 个文件因超长被裁剪) |
| 链接 | https://github.com/sgl-project/sglang/pull/47853 |

## 一句话结论

接入 HALOps 作为 SM100 上 opt-in 的 MXFP8 dense GEMM 第三后端并支持 MXFP8 MegaMoE;GPU 内核时间持平,收益来自 host 侧 dispatch 开销下降。

## 动机

HALOps 是 PaddlePaddle 团队面向 NVIDIA Blackwell(SM100)的高性能 kernel 库。SGLang 已通过 DeepGEMM 与 FlashInfer 服务 MXFP8 dense linears,本 PR 加入第三个后端,经 `--fp8-gemm-backend halops` 显式选择;默认行为不变(`auto` 不会解析到 HALOps,不设 flag 则不导入包)。同时集成 MXFP8 MegaMoE 路径:MXFP8 experts 复用既有 DeepGEMM MegaMoE W8A8(`fp8xfp8`)kernel,scale 布局保持 canonical(每 32 一组的 UE8M0),由 MegaMoE 自行派生所需布局。`halops` wheel 只发布在 PaddlePaddle 的 CUDA 13.2 包页(cp312/Linux),官方 Docker 镜像在并行 build stage 安装。

## 优化点

`halops_mxfp8.py` 提供惰性加载器(缺 wheel 时 raise 带安装提示,使失败发生在 load 而非首个 token)、CUTLASS SfAtom 128x4 scale swizzle(与 `group_wise_quant(use_swizzle=True)` 逐字节一致,故已量化的 checkpoint 只做一次量化和一次重排)、load 期在线 weight quantizer。`halops_mxfp8_linear.py` 是 W8A8 MXFP8 dense linear runner:无 tile 的形状、K 非 128 对齐、非 bf16/fp16 激活、非 SM100 设备回落 FlashInfer CUTLASS(用层上保留的 canonical scale),`input_scale` 无受支持 producer 故直接拒绝。`fp8_utils.py`/`fp8.py` 完成后端枚举、解析与 dispatch;block-FP8 checkpoint 配 halops 直接 raise 而非静默跑 flag 未要求的 kernel。`mega_moe.py` 按 E4M3 expert weights 选择 `fp8xfp8` MMA type,把 UE8M0 scale 字节转为 fp32(`(sf.to(int32) << 23).view(float32)`),该类型下 K 按 un-packed 读取。

## 关键改动

```text
python/sglang/srt/layers/quantization/halops_mxfp8.py: 惰性加载器、SfAtom swizzle、在线量化
python/sglang/srt/layers/quantization/halops_mxfp8_linear.py: W8A8 MXFP8 dense linear runner + FlashInfer fallback
python/sglang/srt/layers/quantization/fp8_utils.py: HALOPS 后端枚举/解析/dispatch
python/sglang/srt/layers/moe/mega_moe.py: mxfp8 megamoe(fp8xfp8 MMA、UE8M0→fp32 scale)
python/sglang/srt/layers/quantization/fp8.py: load 期 scale 准备与 canonical MoE 量化
```

```python
# 摘录省略 docstring,见 halops_mxfp8.swizzle_scale_to_sfatom
def swizzle_scale_to_sfatom(scale_u8: torch.Tensor) -> torch.Tensor:
    m, k_groups = scale_u8.shape
    return (
        scale_u8.contiguous()
        .view(m // 128, 128, k_groups // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(m // 128, k_groups // 4, 4, 32, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(m, k_groups)
        .contiguous()
    )
```

## 性能声明

4x SM100,DeepSeek-V4-Flash,TP4 / attention-DP4 / EP4,EAGLE,`--moe-a2a-backend megamoe --moe-runner-backend deep_gemm --quantization mxfp8`,三臂仅 `--fp8-gemm-backend` 不同。bench_serving(512 random prompts,mean input 1024 / output 128,concurrency 128):HALOps 23.94 req/s、1576 out tok/s、TTFT 592 ms;DeepGEMM 20.23、1332、1212 ms;FlashInfer 19.51、1285、865 ms。但作者明确注明差异在 host 侧而非 kernel:同一 prefill trace 三后端执行相同 8296 个 kernels,GPU kernel 总时长 421.1/417.3/424.3 ms(±2% 内),"this PR does not claim a GPU-level speedup";CPU-op busy 1528.8/1812.6/2956.7 ms,CUDA runtime calls 22,955/17,127/51,213。accuracy:GSM8K 全 1314 题、5-shot、greedy、--max-tokens 4096:halops 0.895、deep_gemm 0.897、flashinfer_cutlass 0.879。

## 验证与回归风险

新增 CPU contract 单测 `test/registered/unit/layers/quantization/test_halops_mxfp8.py`:SfAtom 布局对 CUTLASS 128x4 atom 参考实现、传参契约、dtype 处理、FlashInfer fallback(不依赖 halops 可导入)、checkpoint scale 不被重量化、hot reload 时 SfAtom 跟随新 scale 且保持 graph 已捕获的 storage。默认路径零影响(不选不导入)。约束:SM100-only、cp312 Linux wheel、K%128、N%128(SfAtom row align)、带 bias 时 M%256。pack 中 CI:Base 通过,Extra 与 AMD ROCm 10 失败标记。

## 关联

正文未引用其他 PR 编号;HALOps 与 DeepGEMM、FlashInfer 并列为 MXFP8 dense GEMM 第三后端。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
