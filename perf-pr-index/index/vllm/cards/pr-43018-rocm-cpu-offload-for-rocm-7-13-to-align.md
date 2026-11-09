# PR #43018: [ROCm] Cpu offload for ROCm 7.13+ to align the hipMemcpyBatchAsync params and perf in 7.14x

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @hongxiayang |
| 合并 | 2026-08-21 |
| 分类 | KV Cache / Memory (tier: major) |
| Label | rocm, ready, v1, nvidia |
| Diff | diffstat 缺失(pack 未记录);diff 覆盖 4 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/43018 |

## 一句话结论

修复 ROCm 7.13+ 上 hipMemcpyBatchAsync 的 numAttrs 参数与 >8192 描述符段错误,KV CPU offload 批量拷贝提速至 19 倍。

## 动机

前序 PR #40549 在 ROCm 上启用了 KV CPU offloading,但发现 hipMemcpyBatchAsync API 的缺陷:ROCm 7.2.1-7.2.3 在 numAttrs>0 时直接返回 hipErrorNotSupported,只能以 numAttrs=0 调用;而实测 7.2.x 的批量接口根本没有做 batching(性能与逐条 hipMemcpyAsync 相同),真正优化的批量实现在 ROCm 7.14+ 才打包发布。另外 7.14/7.15 上发现 count>8192 时 hipMemcpyBatchAsync 会触发 GPU memory access fault(segfault),需要绕过。

## 优化点

三层兼容与修复:(1) 按运行时 HIP 版本动态决定 numAttrs——`_num_attrs_for_hip_version` 以版本号判定,≥7.13 传 1(携带 srcAccessOrder 属性),7.2.x 及未知版本保守传 0;(2) 描述符分块——ROCm 默认每批上限 8192(`kRocmDefaultMaxBatchDescriptors`),超过则循环分块调用,CUDA 保持不设上限,`VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS` 环境变量可在任意平台覆盖;(3) HIP runtime 加载健壮化——`_load_hip_runtime` 依次尝试 `libamdhip64.so`/`.so.7`/`.so.6`,兼容仅有版本化 soname 的 runtime-only/wheel 安装。C 侧 `cache_kernels.cu` 的 `swap_blocks_batch` 与 Python 侧 `cuda_mem_ops.py` 的 `copy_blocks` 同步加入分块逻辑。

## 关键改动

```text
vllm/v1/simple_kv_offload/cuda_mem_ops.py: HIP 版本门控 numAttrs、8192 描述符
  分块、HIP runtime 多 soname 回退加载
csrc/libtorch_stable/cache_kernels.cu: swap_blocks_batch 增加 resolve_max_
  batch_descriptors 与分块循环(CUDA/ROCm 同构)
vllm/envs.py: 新增 VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS
tests/v1/simple_kv_offload/test_hip_mem_ops.py: 版本门控/加载回退/分块单测
```

关键机制摘录(cuda_mem_ops.py,版本门控):

```python
def _num_attrs_for_hip_version(version: int) -> int:
    # ROCm 7.2.1-7.2.3 reject numAttrs > 0; 7.13+ accept it.
    major, minor = version // 10_000_000, (version // 100_000) % 100
    return 1 if (major, minor) >= (7, 13) else 0
```

## 性能声明

微基准(hip 程序,hipMemcpyBatchAsync(numAttrs=1) 对比逐条 hipMemcpyAsync):ROCm 7.2.3 上两者约 1×(无 batching);ROCm 7.15 nightly wheel(8× MI350X,gfx950)上批量接口对逐条循环提速 6-19×:store(D→H) 4KiB 块 25.4 vs 1.6 GB/s(×15.4),64KiB 块 48.9 vs 6.3 GB/s(×7.7);load(H→D) 4KiB 块 31.5 vs 1.7 GB/s(×19.1),64KiB 块 49.0 vs 5.8 GB/s(×8.4)。回归:gsm8k 200 题 offload 0.950 / baseline 0.955;serve 冒烟为 amd/DeepSeek-R1-0528-MXFP4-Preview TP8 + VLLM_USE_SIMPLE_KV_OFFLOAD=1(kv-offloading-size 600)。

## 验证与回归风险

新增 `test_hip_mem_ops.py`:版本→numAttrs 参数化表(7.1 至 8.0 共 11 档)、HIP 加载三级回退、描述符分块计数(8192 恰好单批、超 1 溢出、Kimi bs=1 shape 8235 描述符→4096×2+43、CUDA 不分块)与空输入 no-op;该目录在 NVIDIA CI 也默认执行(monkeypatch,无需真实 HIP runtime)。回归测试覆盖源码构建(ROCm 7.2.3 nightly)、53/53 单测、gpt-oss-20b eager+lazy 集成、serve 冒烟与 gsm8k。风险点:分块循环在描述符极多时引入多次 launch 开销;7.14/7.15 的 count>8192 段错误为运行时缺陷,修复未进 7.14x,依赖上限常量;未知 HIP 版本走保守 numAttrs=0,7.13+ 的 srcAccessOrder 属性将被忽略。

## 关联

- 前序:PR #40549(ROCm CPU offloading 首次启用)、issue #44341
- 同区域:KV CPU offload / simple_kv_offload 相关 PR(如 KV connector 类优化)

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
