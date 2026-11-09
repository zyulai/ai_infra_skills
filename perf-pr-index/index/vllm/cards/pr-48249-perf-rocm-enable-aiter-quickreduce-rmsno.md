# PR #48249: [Perf][ROCm] Enable AITER QuickReduce + RMSNorm fusion

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @mjkvaak-amd |
| 合并 | 2026-09-17 |
| 分类 | Attention / Kernel（tier: major） |
| Label | rocm, ready |
| Diff | diffstat 缺失（pack 未记录增删行数）；diff 显示改动 `vllm/_aiter_ops.py` 与 `tests/distributed/test_quick_all_reduce.py` 两个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/48249 |

## 一句话结论

ROCm prefill 的 fused all-reduce+RMSNorm 改走 AITER QuickReduce 融合内核，该 kernel 本身约 2.7x 加速，TP=2 逐层通信开销下降带动吞吐最高 +29.3%。

## 动机

vLLM 在 ROCm 上已有的 AITER fused all-reduce + RMSNorm 实现只覆盖 1-stage（decode 尺寸）路径，非 1-stage（prefill 尺寸）的 fused AR+RMSNorm 仍走 `custom_fused_ar_rms` 的常规路径，通信 + norm 两次落盘开销在 prefill 阶段占比可观。上游 ROCm/aiter#4104 新增了 QuickReduce + RMSNorm 单融合 kernel，本 PR 把符合条件的 prefill 调用路由过去。作者明确注明：融合只适用于 `max_num_batched_tokens<=8192`，>8k 的场景需要后续 vLLM 侧改动，不在本 PR 范围。

## 优化点

在 `_rocm_aiter_fused_allreduce_rmsnorm_impl` 的非 1-stage 分支前插入新的分派逻辑：当 vLLM 的 `qr_comm` 可用且未禁用、`should_quick_allreduce` 判定可走 QR、输入/residual/weight 三者 dtype 一致且为 bf16/fp16、weight 为长度等于 hidden_dim 的一维向量、且单行字节数 `row_size <= 32KB` 并能整除 32KB tile 时，改调 `aiter.qr_all_reduce_rmsnorm` 一步完成 all-reduce + RMSNorm，直接返回 out 与 residual_out，不再进入 `custom_fused_ar_rms`。该 kernel 需要 AITER 自己的 DeviceComms 实例而非 vLLM C++ QR 缓冲，因此新增 `_get_or_create_aiter_qr_rmsnorm_comm`：在同一 TP CPU group 上惰性 `init_custom_qr` 建 AITER QR 通信器，经 `all_gather_object` 交换 handle 后缓存到 device communicator 复用，不按调用创建；world_size 仅支持 2/4/8，任何失败路径都会写回 `False` 缓存并回退原路径。

## 关键改动

```text
vllm/_aiter_ops.py: 新增 AITER QR 通信器惰性创建/缓存，与 prefill 尺寸 fused QR+RMSNorm 分派逻辑
tests/distributed/test_quick_all_reduce.py: 新增 prefill 分派与 decode 保留 1-stage 两个回归测试
```

```python
fused_qr_rmsnorm_ok = (
    qr_comm is not None
    and not getattr(qr_comm, "disabled", True)
    and qr_comm.should_quick_allreduce(input_)
    and input_.shape == residual.shape
    and input_.dtype in (torch.bfloat16, torch.float16)
    and row_size > 0
    and row_size <= 32 * KB
    and (32 * KB) % row_size == 0
)
```

## 性能声明

测试条件：`amd/Qwen3.5-397B-A17B-MoE-MXFP4`，TP=2，`max_num_batched_tokens=8192`，`kv_cache_dtype fp8`，`ROCM_AITER_FA`，`fuse_allreduce_rms` 等 pass 开启（AITER>=v0.19.0）。

- ISL=8192/OSL=1024，并发 16/32/64/128：Output tok/s +8.5%/+11.3%/+16.8%/+20.3%，Mean TTFT −29.8%~−30.3%，Mean TPOT −5.4%~−14.5%，E2EL −7.9%~−17.0%。
- ISL=61440/OSL=60，并发 4/8：Output tok/s +28.1%/+29.3%，Mean TTFT −22.3%/−22.8%。
- Trace（ISL=8k, OSL=32, Conc=4）：prefill 的 fused QR+RMSNorm kernel 较 baseline 约 2.7x faster；decode 保持 1-stage 路径不变。
- GSM8K：flexible-extract 0.8158→0.8044，strict-match 0.8499→0.8560，无回归。

## 验证与回归风险

新增/依赖测试：`tests/distributed/test_quick_all_reduce.py` 中 `test_rocm_aiter_fused_rmsnorm_uses_aiter_qr_rmsnorm_for_prefill`（验证 prefill 尺寸输入分派到 `aiter.qr_all_reduce_rmsnorm` 及各参数传递）与 `test_rocm_aiter_fused_rmsnorm_keeps_1stage_decode_path`（验证 decode 尺寸仍走 1-stage）；另跑了 GSM8K 精度 A/B。风险面：仅影响 ROCm + AITER + 开启 `fuse_allreduce_rms` 的 TP 2/4/8 场景；分派条件（dtype、行大小 ≤32KB、`max_num_batched_tokens<=8192`）不满足或 AITER 缺少 `qr_all_reduce_rmsnorm` 时回退原路径；GSM8K flexible-extract 有 0.0114 的小幅波动（INT4/量化 QR level）。

## 关联

依赖上游 kernel：ROCm/aiter#4104（需 AITER>=v0.19.0）。同区域其他 vLLM PR 编号本 PR 未提及，无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写，性能数字未独立核实。
