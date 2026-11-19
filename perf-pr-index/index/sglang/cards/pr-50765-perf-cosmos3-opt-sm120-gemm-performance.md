# PR #50765: [Perf] cosmos3: opt sm120 gemm performance on bimanual

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @xupinjie |
| 合并 | 2026-11-17 |
| 分类 | Attention / Kernel(major) |
| Label | run-ci, diffusion, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/50765 |

## 一句话结论

为 Cosmos3-Nano-Sim-Bimanual 在 SM120(RTX PRO 6000 Blackwell)选型 QuACK 与 FlashInfer cuBLASLt 后端替换 PyTorch 默认 BF16 GEMM(opt-in 环境变量),微基准单形状最高 1.409×,端到端 901 帧生成请求 59.244 s → 56.418 s,节省 2.825 s(4.77% 延迟下降)。

## 动机

PyTorch 默认 BF16 GEMM 在 Cosmos3-Nano-Sim-Bimanual 的小 M 投影上留有明显性能空间(RTX PRO 6000 Blackwell Server Edition):224×4096×4096 63.04 µs(有效 BF16 峰值利用率 41.3%)、448×6144×4096 139.01 µs(56.2%)、448×24576×4096 472.92 µs(66.0%);利用率按 `2MNK / 完整 GPU 调用时间 / dense BF16 峰值`(288.8 TFLOP/s @ 1.5 GHz SM 时钟,188 SM)计,含 reduction 内核,非 NCU 指令吞吐百分比。

## 优化点

- opt-in `SGLANG_OPT_COSMOS3_SM120_GEMM=1`,限定 Cosmos3 GEN BF16、TP=1、实测的 188-SM SM120 配置(`(12, 0, 188)` 设备检查)。
- gate/up 投影选 QuACK,选定的 QKV/output/down 投影选 FlashInfer cuBLASLt 后端;后端选择策略留在 Cosmos3 模型侧,执行、首次使用库调优与安全原生回退共享在新的 `gemm/sm120_bf16.py`(`Sm120Bf16Linear`:非空连续 BF16、N/K 整除 8、无 bias/autograd、非编译期、非 HIP、SM120;不满足即返回 None 走原生 `UnquantizedLinearMethod`)。
- CUDA graph capture 之外调优、复用库调优结果、独立调用/graph 用 allocator 管理的 scratch(每调用 64 MiB workspace,避免 `mm_bf16` 进程级 scratch 在并发流/图下不安全)。
- 按 quality 分层:`quality=lossless` 允许优化路径,`quality=exact` 保持原生 PyTorch GEMM(bitwise identical)。
- `_backend_ranges` 按 (N, K) 与 M 区间选后端(如 (6144,4096):M 193–384 QuACK、385–448 cuBLASLt、449–640 QuACK、1024–2048 QuACK;区间外保持原生)。

## 关键改动

```python
# python/sglang/kernels/ops/gemm/sm120_bf16.py (新增)
def linear_bf16_quack(x, weight):
    from quack.gemm_interface import gemm_out
    out = x.new_empty((x.shape[0], weight.shape[0]))
    gemm_out(x, weight.T, out, tuned=True, split_k=None)
    return out

def linear_bf16_cublaslt(x, weight): ...  # FlashInfer AutoTuner 选 tactic, allocator 管 scratch

# cosmos3video.py (新增)
class _Sm120LinearMethod(UnquantizedLinearMethod):
    _backend_ranges = {
        (24576, 4096): ((64, 2048, linear_bf16_quack),),
        (4096, 12288): ((64, 512, linear_bf16_quack),),
        (6144, 4096): ((193, 384, linear_bf16_quack), (385, 448, linear_bf16_cublaslt), ...),
        (4096, 4096): ((128, 2048, linear_bf16_quack),),
    }
```

## 性能声明

(全部照抄 PR 正文,未独立核实;RTX PRO 6000 Blackwell Server Edition 96GB,PyTorch 2.14.1+cu130,CUDA 13.0,cuBLAS 13.1.1.3,FlashInfer 0.7.3,QuACK 0.6.5,请求 1500 MHz SM 时钟)

- GEMM 微基准(CUDA 事件计时完整 CUDA Graph 调用,含 split-K/reduction;8 组轮换权重,每样本 128 次调用,调优/热身后 5 样本取中位):M=448 N=6144 K=4096 cuBLASLt 98.66 µs vs PyTorch 139.01 µs = 1.409×(利用率 56.2%→79.1%);M=224 N=4096 K=4096 cuBLASLt 51.76 µs = 1.218×;M=224 N=24576 K=4096 QuACK 232.03 µs = 1.151×;M=448 N=24576 K=4096 QuACK 411.65 µs = 1.149×。另有微基准赢但未启用的后端;当前策略只启用在真实模型激活上验证过的 5 组后端/形状对。
- 端到端:Cosmos3-Nano-Sim-Bimanual BF16 `quality=lossless` TP=1,901 帧 640×352、30 FPS、4 步去噪、seed 2026;原生 59.244 s → 本 PR 56.418 s(1.050×),省 2.825 s(4.77% latency reduction)。每变体各一次计时请求,编译与调优在 warmup 支付。
- 精度:15 个真实激活 GEMM(层 0/12/23 的 5 个启用形状)对 FP64:最差候选/原生 RMS 误差比 0.8203、最大绝对误差比 1.0106,全部通过 1.5× 误差门;`exact` 路径对 `F.linear` bitwise identical。

## 验证与回归风险

新增 `test_sm120_bf16_linear.py`(SM120 必需,注册 `1-gpu-small`)覆盖精度门、特殊值传播、独立 CUDA graph scratch、冷捕获回退、非 Cosmos 形状、cuBLASLt 按精确 M profile 等;`test_sm120_bf16_linear_dispatch.py`(diffusion-1-gpu-5090)覆盖 exact/lossless 回退、构造期 opt-in 条件、区间外不保留 workspace。风险面:opt-in 默认关闭;仅验证 188-SM SKU 与 5 组形状对;`lossless` 不代表生成视频 bit-identical。Base/Extra/AMD CI 均显示失败图标(pack 未提供失败细节)。

## 关联

QuACK 0.6.5、FlashInfer 0.7.3(cuBLASLt 后端)、`multimodal_gen` Cosmos3 管线(`dreams.py` 把 forward_batch 传入 `run_fixed_step_sde`/`commit_clean_kv` 以支持 `request_allows_lossless`)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
