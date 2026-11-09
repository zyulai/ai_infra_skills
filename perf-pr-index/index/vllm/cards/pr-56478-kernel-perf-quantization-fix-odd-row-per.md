# PR #56478: [Kernel][Perf][Quantization] Fix odd-row performance cliff in per-token-group quantization

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ayush1399 |
| 合并 | 2026-09-13 |
| 分类 | Quantization (tier: major) |
| Label | performance, nvidia, quantization, verified |
| Diff | diffstat 缺失(pack 中为 +None/-None across None files);diff 完整可用(2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/56478 |

## 一句话结论

修复 per-token-group 8-bit 量化在奇数行数时的性能悬崖:按下一个偶数行数选择 launch 几何并用边界检查处理尾部,H100 上最坏形状内核快约 4×。

## 动机

在排查 Gemma 4 31B FP8 TP8 H100 上异常差的 TPOT 时发现:13,359-token prefill 前向中约 35 ms 额外耗时来自激活量化,足以抵消 FP8 GEMM 省下的约 30 ms。根因是现有启发式在总 group 数为奇数时选择每 block 一个 group,只启动 16 线程;TP8 下 down-projection 输入宽度为 2688,奇数行数使量化比相邻偶数情形慢约 4×,宽度 5376 的输入也走约 2× 慢的路径。

## 优化点

机制:不再用实际 group 数决定 block 几何,而是按"下一个偶数行数"的 group 数(`(num_rows + (num_rows & 1)) * groups_per_row`)选择 `GetGroupsPerBlock` 的 launch 策略,实际 block 数改为 ceil-div;新增模板参数 `CHECK_BOUNDS`,仅当 `num_groups % groups_per_block != 0` 时启用边界检查(valid_group 判定),整块时该检查被编译掉,尾部组跳过量化但仍参与 block 与 PDL 同步;不做任何 tensor padding 或拷贝,量化算术与 scale 布局不变;helper 同时服务 FP8 与 INT8,两者都获得回归覆盖;单独的 packed kernel 不变。

## 关键改动

```text
csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu: launch 几何按偶数行数策略 + CHECK_BOUNDS 模板参数处理尾部组
tests/kernels/quantization/test_per_token_group_quant.py: FP8/INT8 测试扩展空输入、奇数尾、FP16/BF16/FP32、group 64/128、各 scale 布局
```

核心策略改动:

```cpp
// Odd row counts inherit the next even row count's launch geometry.
const int64_t policy_num_groups =
    (num_rows + (num_rows & 1)) * groups_per_row;
const int groups_per_block = GetGroupsPerBlock(policy_num_groups);
const bool needs_bounds_check = num_groups % groups_per_block != 0;
const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
```

## 性能声明

内核延迟(µs,CUPTI + CUDA graphs,BF16 输入、FP8 输出、group size 128、列主序 FP32 scale UE8M0,三轮 30 样本取中位,cold 冲刷 L2):

- H100:32767×128 hot 41.76→10.45(约 4×);13359×2688 hot 339.63→81.46(约 4.2×);65537×128 hot 81.18→41.82(约 2×);偶数行 32768×128 保持 10.46→10.43 不变。
- GB10:32767×128 hot 49.23→20.93;13359×2688 hot 517.30→487.12(收益小);65537×128 hot 108.24→67.47;偶数行不变。
- H100 生产源 18 形状:几何平均 hot −39.5%、cold −38.0%;259 形状原型中 112 个奇数形状与 next-even 对照差 hot ≤2.9%、cold ≤4.2%。
- GB10 182 个通用形状:几何平均 hot −6.7%、cold −0.2%;91 个奇数/next-even 对几何平均差异 <0.4%。
- 说明:两次测量均未锁频;H200/B200/B300/ROCm 未测;E2E TP8 TPOT 尚未用本补丁复测。

## 验证与回归风险

测试:FP8/INT8 per-token-group 量化测试扩展(空输入、奇数尾、三种输入 dtype、group 64/128、column-major/tma_aligned/UE8M0 布局);isolated production-source tests 678 passed(H100 与 GB10);GB10 Compute Sanitizer memcheck/synccheck 零错误;18 个 H100 形状与 194 个 GB10 形状量化输出字节级一致、scale 一致;GB10 上 Gemma-4-31B-it-FP8-block serving 冒烟通过(更广的精度评测未跑)。风险面:影响所有走 shared-memory per-token-group 量化路径的 FP8/INT8 激活量化(含 spec/FI 各处的 odd-M 场景),偶数行路径与 packed kernel 行为不变;ROCm 未测但同一 launch 宏被修改。

## 关联

相关:#55330(寄存器驻留内核、固定 256 线程的另一种方案,二者互补);同分类相关卡:pr-24342(per-token-group quantization 扩展)、pr-18844(fp8 quant 调优)、pr-21476(int8 per-token-group CUDA kernel)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
