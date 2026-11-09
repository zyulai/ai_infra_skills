# PR #55368: [ROCm][MoE] Pad the AITER MoE intermediate size at allocation time, and round the expert-group count to a kernel that exists

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @sshlyapn |
| 合并 | 2026-09-30 |
| 分类 | MoE (tier: major) |
| Label | rocm, ready, verified |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/55368 |

## 一句话结论

修复 ROCm AITER MoE 两类启动失败:intermediate size 按 CK kernel tile 宽度在分配期对齐补齐,expert-group 数取整到 AITER 已编译 kernel 支持的值。

## 动机

IFM/K2-Horizon-375B-A23B 在 ROCm AITER 路径上两类独立启动失败:(1) AITER CK 2stages MoE kernel 按 `inter_dim <= 192` 分派(以下两 stage 都用 64 宽 tile,以上至少一个 stage 用 128 宽),CK `IsSupportedArgument` 拒绝不能整除的 intermediate——`moe_intermediate_size=1792` 在 TP=8 下为 224,报 `device_gemm ... does not support this GEMM problem`;(2) `biased_grouped_topk` 只为 `NUM_GRP ∈ {1,2,4,8}` 实例化,96 个 expert 算出 group 数 3,launch 失败。

## 优化点

机制一:在既有的 `maybe_roundup_sizes` hook(量化方法已普遍覆写)中按对齐规则 `64 if inter_dim <= 192 else 128` 向上取整 intermediate,发生在 `create_weights` 之前,使 padded 尺寸进入分配;loader 按 checkpoint 范围写真实行。对齐规则镜像 AITER 自身阈值而非一律 128:全 148 个 AITER config 文件(11279 行)中阈值规则移动 0 行 tuned 配置,flat 128 会把 207 行挤到启发式 fallback;且取整不会跨越 192 边界,选中的对齐始终与 dispatcher 一致。pad 通道在任一维度 padded 时用 `torch.zeros` 分配并在转换前清零重载残留(`silu(0)*0=0`,数值惰性);未 padded 层保持 `torch.empty` 零成本。仅作用于 `UnquantizedMoeBackend.AITER`。机制二:`_aiter_get_num_expert_group` 把 group 数取整到能整除 num_experts 且满足 32/group 上限的最大支持值(分组在此路径是 no-op,路由等价);无解时保留原值,由新增 `_aiter_can_use_biased_grouped_topk` 门控回退通用路径。

## 关键改动

```text
vllm/model_executor/layers/fused_moe/oracle/unquantized.py: aiter_moe_intermediate_alignment + AITER 分支 + padding 清零
vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py: group 数取整与 _aiter_can_use_biased_grouped_topk 门控
tests/kernels/moe/test_rocm_aiter_moe.py: 新增 Weight alignment 8 个测试
tests/kernels/moe/test_rocm_aiter_num_expert_group.py: 新文件,5 个 group-count 测试
```

## 性能声明

无正向性能声明(目标为修复启动失败)。已知代价:pad 后 `intermediate_pad` 非零而 CK partial kernel 不吃 pad 参数,kernel 跑满 padded GEMM 含零行——TP=4 时 448→512、TP=8 时 224→256,MoE GEMM **最多约 14% 额外计算**;PR 未提供修复前后端到端吞吐数字。测试结果:gfx950 (MI355X) 上 112 passed、2 skipped,唯一失败为父提交即存在的无关 fp8 accuracy 测试。

## 验证与回归风险

8 个 alignment 测试(阈值表、padded 不跨分支、96→128/224→256/448→512 等端到端、其他后端不 pad、零初始化、bias 形状、数值位等同)+ 重载两次清零 NaN pad 的测试;CDNA 上跨 block_m(32/64/128)验证真实 kernel 接受性;router 测试覆盖 14 种 expert 数与 grouped_topk 参考路由一致性(atol=1e-4)。风险面:仅 unquantized AITER 后端(ROCm);量化 AITER 路径(fp8 需 256 等)有自己的覆写不受影响;TRITON/FLASHINFER_CUTLASS 保持不 pad;无支持 group 数时故意回退通用路由。

## 关联

同区域:#55251(fp8 AITER 路径的同类 zero-init 工作);mxfp4 与 quark nvfp4 已有的零分配;AITER MoE oracle/unquantized 后端相关 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
