# PR #53396: [Kimi K3][Kernel] Support DS conv-state layout in fused KDA decode kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @gcanlin |
| 合并 | 2026-08-27 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, performance, ready, kimi, k3 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files),实际改动为 2 个 CUDA 源文件(CUDA/ROCm)、模型 dispatch、测试与 benchmark |
| 链接 | https://github.com/vllm-project/vllm/pull/53396 |

## 一句话结论

让融合 KDA decode kernel 同时支持 SD/DS 两种 conv-state cache 布局:生产 DS 部署从 Triton fallback 链回到融合路径,E2E 吞吐 +3.84%。

## 动机

`ops.fused_kda_decode` 此前只接受 SD conv-state cache 布局(入口 `TORCH_CHECK` 直接拒绝其他 stride),而生产 Kimi K3 部署(2 节点 P/D 分离,NIXL 点对点状态传输)固定 `VLLM_SSM_CONV_STATE_LAYOUT=DS`,于是整个模型 69 个 KDA 层全部静默回退到多算子 Triton decode 链。实测回退链在 B200(TP8 shape,H_local=12,K=V=128)上比融合 kernel 慢 1.16-1.58x(依 batch 而定),每层 decode 链 launch 数约为 3 倍(8-9 vs 3 个 kernel)。

## 优化点

机制:为两种物理布局各派发到独立的 fused-kernel 模板实例化,把 conv-state 的内层 stride(channel 方向、tap 方向)作为非类型模板参数(NTTP)在编译期固写——SD 按 (tap, channel) 打包即 `<1, 3*H*128>`,DS 按 (channel, tap) 打包即 `<kConvWidth-1, 1>`;launch 时从 conv_state 的 stride 检测布局并在 head-count dispatch 内解析。这样避免了对 runtime inner stride 做通用寻址带来的额外整数指令与寄存器压力(动态 stride 版本寄存器从 64 涨到 72/thread,两个特化版本均保持 64),代价是 kernel 实例数翻倍、扩展二进制增大 392 KB(0.95%)。CUDA 与 ROCm 两个 kernel 文件同步改造,入口校验放宽为接受 SD 或 DS stride 组合,模型侧 `is_fused_kda_decode_supported` 移除对 `is_conv_state_dim_first()` 的否决。此前的 layout enum 写法按 review 意见改写为 stride NTTP,128 个 `kda_decode_fusion_many_heads_kernel` 实例的 SASS 经 cuobjdump 归一化比对逐位一致(同为零 spill),性能按构造等同。

## 关键改动

```text
csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu: kernel 模板新增 kConvStateChannelStride/kConvStateTapStride 两个 NTTP,launch 侧按 conv_state_is_ds 选择 <kConvStateWidth,1> 或 <1,3*H*kDimK>
csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu: ROCm 版同步改造
vllm/models/kimi_k3/nvidia/kda.py: is_fused_kda_decode_supported 不再因 DS 布局拒绝
tests/models/kimi_k3/test_kda.py: 精度测试参数化 SD/DS 双布局 x H=12/24/48/96
benchmarks/kernels/benchmark_kimi_k3_kda_decode.py: 增加 --layout 参数
```

关键寻址改动:

```cuda
const int cs_idx =
    hk * kConvStateChannelStride + w * kConvStateTapStride;
```

## 性能声明

Microbench(B200,CUDA graph replay,H_local=12,K=V=128,50 warmup/300 measured,两组中位数均值):DS 特化融合路径 vs Triton fallback,batch 1-512 提速 1.22x-1.87x(batch 8: 10.480 → 5.610 us,1.87x;batch 512: 171.455 → 140.055 us,1.22x)。布局特化本身:SD 相比动态 stride 版消除 +2.25% 至 +11.53% 回归(geomean +5.04%,回到与原 SD 基线持平);DS 特化相比动态 stride 提升 0.72%-10.95%(batch 128 处 10.95%,geomean 4.75%)。生产 shape 检查(69 组 per-layer state buffer,B=128,DS):37.04 vs 49.33 us/层(1.33x)。E2E(2 节点 x 8xB200,TP8+PP2,FP8 KV cache,DS 布局,FULL_DECODE_ONLY cudagraphs;`vllm bench serve`,random,128 输入/512 输出,128 请求,concurrency 64,temperature 0):输出吞吐 2069.73 → 2149.26 tok/s(+3.84%),Mean TPOT 29.73 → 28.59 ms(-3.84%),Mean TTFT 基本不变(-0.40%);GSM8K flexible-extract 0.9689 / strict-match 0.9697。kernel trace:DS 布局下每个 KDA decode 层都跑融合 kernel(窗口内 5076 次调用,平均 7.2 us/次,占窗口 kernel 时间 1.26%),fallback 三件套(约 3.5% kernel 时间)从 decode 热路径消失。

## 验证与回归风险

`tests/models/kimi_k3/test_kda.py::test_fused_kda_decode_correctness` 参数化 SD/DS 双布局、H=12/24/48/96、output normalization 与 lower-bound decay:10 passed / 51 deselected,stride-NTTP 重写后复跑同样通过。数值对相同输入逐位一致(正文声明),TP4 collectives 无关;二进制膨胀 392 KB(0.95%)为已声明的代价。风险面:入口 stride 校验同时接受两种布局,若上游引入第三种 stride 组合会显式报错而非静默;E2E 数字采集于 enum 版本 commit `40459de9d3`,作者以 SASS 逐位一致论证数字可迁移至 stride-NTTP head,并承诺在其上补跑一次 E2E。适用范围仍受 fused kernel 既有条件约束(H∈{12,24,48,96}、head_dim 128、bf16、num_spec==0 等)。

## 关联

同区域:Kimi K3 模型支持见 pr-50000;fused GDN MTP/KDA 相关见 pr-65874(fused GDN MTP DSpark support)、pr-61855(ROCm fused GDN MTP triton port)、pr-59735(non-spec GDN decode 标准路径);DS/SD conv-state 布局由 `VLLM_SSM_CONV_STATE_LAYOUT` 控制,服务于 P/D 分离的 NIXL 状态传输(见索引 KV connector 相关条目)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
