# PR #33506: [Fix] [AMD] dsa: Fix Triton cache flooding in aiter backend

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-mvarjoka |
| 合并 | 2026-11-15 |
| 分类 | CUDA Graph / Compiler(major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 additions/deletions 与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/33506 |

## 一句话结论

修复 DSA `aiter` 后端里 `_get_valid_kv_indices_kernel` 把 batch size 声明为 `tl.constexpr` 导致的 Triton 编译缓存暴涨:去掉从未被内核体引用的 `bs` constexpr,内核从"每个 batch size 编译一次"变为每个 `topk` 只编译一次;配套 aiter 修复后基准扫描的容器文件系统变更数从约 334,000 降到约 94,000,总 token 吞吐平均约 +5%。

## 动机

用 DSA 模型(DeepSeek-V3.2、GLM-5.x FP8)配 `--attention-backend dsa --dsa-prefill-backend aiter --dsa-decode-backend aiter` 服务时,`/tmp/torchinductor_root/triton/` 下的 Triton 编译缓存无界增长——一次基准扫描写入数万个内核目录;缓存填满后文件系统操作拖慢容器。根因:`_get_valid_kv_indices_kernel` 声明 `bs: tl.constexpr`,Triton 把每个 constexpr 值烘焙进独立编译产物,continuous batching 下 decode/extend batch size 几乎每步都变,于是持续重编译、刷爆缓存。更糟的是 `bs` 在内核体中从未被引用——launch grid `(bs,)` 已通过 `tl.program_id(0)` 编码 batch 维,纯属死特化。另一个 constexpr `topk`(= `dsa_index_topk`)每模型固定且真实用于 `tl.arange(0, topk)`,只编译一次,保持不动。该问题不出现于 `--dsa-*-backend tilelang`(不调用此内核)或非 DSA 模型如 DeepSeek-R1(不进该后端)。

## 优化点

- 从 `python/sglang/kernels/ops/attention/dsa/triton_kernel.py` 的内核签名删掉 `bs: tl.constexpr`,并从 `dsa_backend.py` 的两个调用点(`_forward_aiter_decode`、`_forward_aiter_extend`)移除传参;grid 仍用 `bs`。
- 内核现在(每个 `topk`)只编译一次,不再按 batch size 编译。
- 同类 bug(把 batch size 当 `tl.constexpr`)也存在于 aiter 仓库的 `_fwd_kernel_stage2_asm`(经 `mla_decode_fwd` 调用),由配套 aiter PR #4545 单独修复;两处都修才能完全止住该服务路径的缓存增长。

## 关键改动

```python
@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
-    bs: tl.constexpr,
    topk: tl.constexpr,
):
```

调用侧 `get_valid_kv_indices` 同步移除 `bs` 实参;diff 仅此两处小改。

## 性能声明

(全部照抄 PR 正文,未独立核实)

- 用 `docker diff sglang`(统计容器内文件系统变更,以 Triton 缓存为主)跑完整基准 sweep:修复前约 334,000,两个补丁(本 PR + aiter PR)都打上后约 94,000;两个补丁缺一不可,各堵住该路径上一个按 batch size 编译的内核。
- "两个补丁都包含时,上述基准 sweep 的 total token throughput 指标平均约 5% uplift";"所有基准场景性能应当持平或更好"。
- 修复验证方法:清空旧缓存、跑 sweep 后确认 `/tmp/torchinductor_root/triton/` 中 `_get_valid_kv_indices_kernel` 目录稳定在小固定数,decode 输出不变。

## 验证与回归风险

验证依赖人工复现脚本(lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x 镜像,DeepSeek-V3.2 TP8,16 组 isl/osl/并发配置的 bench_serving sweep)。PR 未提供单元测试,未提供模型级精度评测;风险极低(仅删一个未使用的 constexpr 参数),但需与 aiter PR #4545 配套才能完全消除缓存增长。CI 状态:Base/Extra/AMD ROCm 10 三个 run 显示失败图标(pack 未提供具体失败原因)。

## 关联

aiter PR #4545(同批修复 aiter 侧 `_fwd_kernel_stage2_asm` 的同类 constexpr bug)。涉及文件:`python/sglang/kernels/ops/attention/dsa/triton_kernel.py`、`python/sglang/srt/layers/attention/dsa_backend.py`(调用点改动在 diff 中未展开)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
