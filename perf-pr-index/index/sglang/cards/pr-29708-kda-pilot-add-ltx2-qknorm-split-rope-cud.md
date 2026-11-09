# PR #29708: [KDA-Pilot] Add LTX2 QKNorm split-RoPE CUDA fast path

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @BBuf |
| 合并 | 2026-07-01 |
| 分类 | Attention / Kernel (tier: major) |
| Label | diffusion, jit-kernel, run-ci, run-ci-extra |
| Diff | diffstat 缺失(pack 记为 +None/-None across None files;diff 本身可用,5 个文件,含 4 个新文件) |
| 链接 | https://github.com/sgl-project/sglang/pull/29708 |

## 一句话结论

为 LTX-2.3 的 Q/K RMSNorm + split-RoPE 预处理新增原生 CUDA JIT 快路径,kernel 几何平均提速 5.84x,LTX-2.3 HQ 端到端降噪阶段缩短 9.75%。

## 动机

LTX-2.3 attention 预处理的热点模式是 `apply_split_rotary_emb(q_norm(q), (q_cos, q_sin)).to(bf16)` 对 Q/K 各跑一遍:先 `torch.nn.RMSNorm`(bf16 autocast 下内部 fp32),再 fp32 split-RoPE,最后一次性舍入到 bf16。eager 路径要分别物化 RMSNorm 与 RoPE 的中间张量,多次整层读写。该任务来自 MIT HAN Lab Kernel Design Agents(KDA-Pilot)的 `b200_ltx2_qknorm_split_rope__bitwise`,要求与 BF16 attention 输入契约 bitwise 一致。

## 优化点

新增 JIT CUDA kernel `ltx2_qknorm_split_rope.cuh`:每个 token 行一个 block(128 线程,4 warp),第一遍向量化(fmaf x4)累加平方和、warp shuffle 归约后算 `rstd`,第二遍对每对 `(idx0, idx0+half_dim)` 做归一化乘权与 split-RoPE 旋转(`fmaf(-sin, x1, x0*cos)`),全程 fp32 计算、仅在最终输出做一次 `__float2bfloat16_rn` 舍入,与 eager oracle 逐位一致。host 侧用 TensorMatcher 校验形状/stride(支持非连续 cos/sin、独立 Q/K 序列长度、head_dim 64/128)。Python 侧注册为带 fake impl 的 eager custom op,`torch.compile(fullgraph=True)` 可直接图覆盖。`LTX2Attention` 默认尝试该 CUDA 路径(TP=1、`nn.RMSNorm`、4D split-RoPE 时),不支持或 JIT 加载/launch 失败一次后全局禁用并回落原实现。

## 关键改动

```text
python/sglang/jit_kernel/csrc/diffusion/ltx2_qknorm_split_rope.cuh: 新增,fused RMSNorm+split-RoPE CUDA kernel。
python/sglang/jit_kernel/diffusion/ltx2_qknorm_split_rope.py: 新增,JIT 加载、custom op 注册、can_use 判定(SM100+)。
python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py: LTX2Attention forward 接入快路径与回退。
test/registered/jit/diffusion/test_ltx2_qknorm_split_rope.py: 新增,bitwise/拒绝非法输入/torch.compile 测试。
test/registered/jit/benchmark/diffusion/bench_ltx2_qknorm_split_rope.py: 新增,14 条 LTX-2.3 生产形状基准。
```

## 性能声明

KDA final k22 运行(空闲 B200):14 条生产形状(LTX-2.3 two-stage 与 HQ 行,head_dim 64/128,video/audio/cross,seq 126-32640)全部 `torch.equal` bitwise 通过;相对 task-local destination-passing PyTorch baseline,等权几何平均 speedup 5.84x,最小 4.22x,最大 7.34x,无回退行;生产网格 fallback_count == 0。集成基准 CI-small(B200):stage1_video_self_q16_k16_d4096 为 191.15 us -> 33.98 us(5.625x),stage1_audio_to_video_q16_k8_d2048 为 176.30 -> 32.34 us(5.451x)。E2E(B200,LTX-2.3 HQ A/B,无 torch.compile):E2E 17022.27 -> 15462.38 ms(-9.16%),denoise stage 12941.22 -> 11679.22 ms(-9.75%),refinement stage -11.47%;fused 路径 hit=16384、fallback=0。视频输出对比(seed 42,1920x1088,121 帧):SSIM 1.000000,PSNR=inf。

## 验证与回归风险

新增注册单测(B200,SM<10 跳过):bitwise 相等、非法输入拒绝(fp32 输入、错误 stride)、`torch.compile(fullgraph=True)` custom-op 覆盖,本地 5 passed;语法/format/lint 检查通过。pack 中 CI Base/Extra 均为 pending(hourglass)。回归风险:(1) 仅 SM100+(B200)启用,其他 GPU 自动回落原 eager;(2) 融合分支要求 `nn.RMSNorm` 且 q/k norm eps 与 `norm_eps` 一致、TP=1,不满足时静默回落;(3) 一次 launch 异常会全局置 `_LTX2_QKNORM_SPLIT_ROPE_CUDA_DISABLED=True`,此后进程内不再尝试;(4) `pe` 为 3D(interleaved RoPE)时走原路径,该路径未改动。

## 关联

同区域 diffusion fused attention 预处理:PR #27096(Cosmos3 fused qknorm rope,本批次);同模型 LTX2 其他融合:residual_gate_add(同文件已有)。本批次内无直接前序 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
