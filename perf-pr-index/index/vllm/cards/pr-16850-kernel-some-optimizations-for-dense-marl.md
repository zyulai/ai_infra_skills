# PR #16850: [Kernel] some optimizations for dense marlin and moe marlin

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jinzhen-lin |
| 合并 | 2025-05-05 |
| 分类 | MoE |
| Label | performance, ready, ci/build |
| Diff | diffstat 缺失(pack 标注 +None/-None);diff 规模大,22 个文件被截断 |
| 链接 | https://github.com/vllm-project/vllm/pull/16850 |

## 一句话结论

把 MoE marlin 的优化迁移到 dense marlin(去 max_par 限制、m8n16k16 模拟、AWQ fma 融合),MoE marlin 索引/A 缓存优化最高 +10%,并新增 FP8 marlin。

## 动机

marlin 系 W4A16/W8A16 kernel 在三个层面有余量:dense marlin 未享受 #14447 为 moe marlin 引入的优化(大 batch 受 `max_par` 限制、m<=8 无专属 MMA 路径、AWQ 反量化多两次浮点运算);moe marlin 读 A 时的索引计算与重复 IO 有浪费;fp8 marlin 独立于 gptq marlin、不支持 block quant,也无法跑 DeepSeek 的 W-FP8-A-FP16。

## 优化点

(1) dense marlin(迁移自 #14447):改造 workspace 使用逻辑去掉 `max_par` 上限,加速 m>1024 大 batch;用 `m16n8k16` MMA 经转置模拟 `m8n16k16`,优化 m<=8;AWQ 路径把 `mul(sub(qw, zp), scale)` 融合为 `fma(qw, scale, -zp*scale)`(`-zp*scale` 可预计算);删除无用 kernel 缩小 wheel、把 kernel 拆多文件加速编译。(2) moe marlin:读 A 的索引计算改为尽量缓存行/列信息(shared 里新增 `sh_rd_block_sorted_ids` 预除 `top_k`),最高 +10%;用剩余 shared memory 尽量缓存矩阵 A——同一 threadblock 处理相同 M 不同 N 的 slice 时(`should_load_a` 判定)跳过重复的 A 全局读,主循环改为 stage-group 双层循环复用已缓存的 A tile。(3) FP8 marlin:把 fp8_marlin 合并进 gptq_marlin kernel 并支持 block quant;为 moe marlin 增加 fp8(生成器新增 `kFE4M3fn`,仅 channelwise 或 group_size=128),使 DeepSeek W-FP8-A-FP16 可跑。同时 `has_act_order`/`has_zp` 从模板参数改为由 `w_type`/`group_blocks` 推导的 constexpr,减少实例化;`dequant_data` 对 (has_zp && !is_zp_float) 复用 B8/B128 无 zp 反量化路径以提升数值精度。

## 关键改动

```text
csrc/moe/marlin_moe_wna16/marlin_template.h: 主 kernel——A 缓存复用、索引预除、dequant_data、FP8 支持
csrc/moe/marlin_moe_wna16/generate_kernels.py: kernel 生成矩阵调整(新增 kFE4M3fn、收紧 thread config)
csrc/moe/marlin_moe_wna16/kernel.h: 模板签名瘦身(has_zp/has_act_order 移除)+ max_shared_mem 参数
csrc/moe/marlin_moe_wna16/ops.cu: get_kernel_cache_size 计入 m_block_size_8 与 rd_sorted_ids
```

```cpp
int sh_a_max_row =
    ((max_shared_mem - 1024) / 16 - shm_size_used) / (thread_k_blocks * 2);
...
if (should_load_a) {
  int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe_a;
  ...
  sorted_row = sh_rd_block_sorted_ids[row];  // 预除 top_k,避免热循环内除法
```

## 性能声明

- moe marlin 读 A 的索引计算优化:performance improvement of up to 10%(PR 正文唯一量化声明,未附基准表,PR 未注明测试条件)。
- 其余各项(大 batch 去 max_par、m<=8 MMA 模拟、AWQ fma 融合、A 缓存省 IO)仅作定性描述,PR 未给出分项数字。

## 验证与回归风险

diff 中 22 个文件被截断,pack 未见新增/修改的测试文件,PR 正文未列测试结果与基准数据。风险面:kernel 生成矩阵收紧(小 batch 只保留 (128,128,256)、大 batch 只保留 (64,256,256))可能移除部分此前可达的 shape 配置;A 缓存复用依赖 `max_shared_mem` 动态参数与 `sh_a_max_row` 计算,`static_assert` 保证 reduction 复用不超过 B 的 shared 用量;AWQ fma 融合与 `dequant_data` 的无 zp 反量化改变中间数值路径(作者注明"both weight and zero point are dequanted using this logic, the final dequanted weight would be correct");FP8 仅支持 channelwise 与 group_size=128。

## 关联

同区域:前序 PR #14447(moe marlin 优化,本 PR 向 dense marlin 迁移其方法);索引内 MoE kernel 卡 PR #46340(batched MoE TD operand loads)、PR #36022(FlashInfer MoE A2A)、PR #28373(AMD Triton MoE tuning);FP8 block quant 方向同 PR #55330(per-token-group FP8 量化)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
