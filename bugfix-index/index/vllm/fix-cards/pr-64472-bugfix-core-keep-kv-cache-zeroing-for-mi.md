# PR #64472: [Bugfix][Core] Keep KV cache zeroing for mixed-precision UniformTypeKVCacheSpecs groups

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#64472](https://github.com/vllm-project/vllm/pull/64472) |
| 作者 | zyongye |
| 合并日期 | 2026-11-02 |
| labels | bug, documentation, ready, ci/build, kv-cache-manager |
| 热度 | 👍 0 · 💬 27 |
| 分类 | KV Cache / Memory |
| bug 类型 | correctness |

## 一句话结论
混合精度 KV cache 组被 scheduler 归约后 zeroing 决策丢失、PP worker 间不一致;修复为把决策持久化并在全部 worker 间 OR 汇聚。

## 问题现象
混合精度 KV cache(如 `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers 0` 这类跳层配置)下,新分配的 attention 块可能不被清零:跨组复用的块在不同精度下被重新解释,把旧字节解码成 NaN/Inf。开启 PP 时问题更具体:验证中 Qwen3-0.6B + MRV2 + PP=2 的 Worker_PP1 断言 `kv_block_zeroer is not None` 失败,首个 completion 返回 HTTP 500。

## 根因
正文披露两处:
1. `generate_scheduler_kv_cache_config` 会把每个 `UniformTypeKVCacheSpecs` 组归约为一个 layer spec,在此归约之后重新计算混合精度会丢失信息,zeroing 决策在 scheduler 投影中丢失。
2. 只保存 worker 0 的混合精度结果会让 PP worker 之间不一致:纯 uniform 精度的 stage 跳过 zeroer 初始化,却仍会收到 scheduler 下发的 `new_block_ids_to_zero`,导致 MRV2 在没有 zeroer 的 stage 上收到清零任务。

## 修复方案
- `KVCacheConfig`(vllm/v1/kv_cache_interface.py)把 `needs_kv_cache_zeroing` 从 property 改为 `field(init=False)` 的存储字段,在 `__post_init__` 中由 `has_mamba_layers or has_mixed_precision_kv_cache` 解析——即在各 worker 的完整配置上初始化,`has_mixed_precision_kv_cache` 仍是本地事实。
- `get_kv_cache_configs`(vllm/v1/core/kv_cache_planning.py)在最终 replanning 后对所有 worker 的决策做 OR:`needs_kv_cache_zeroing = any(...)`,再写回每个 cfg。
- `generate_scheduler_kv_cache_config`(kv_cache_utils.py)的 deepcopy 投影因此保留该决策。
- 权衡:某个 stage 需要清零时,uniform 精度的 stage 也会对新 attention 块清零;uniform 精度、纯 attention 的配置继续跳过清零。

## 影响与回归风险
- 影响窗口:正文未说明具体版本范围。
- 回归面:所有 KV cache 配置的初始化与 scheduler 配置生成路径(`kv_cache_interface.py`、`kv_cache_planning.py`、`kv_cache_utils.py`);uniform 精度配置行为不变,混合精度 + PP 下 uniform stage 多付一次清零开销。正文声明这是 #63232(MiniMax-M3.1-Flash)拆出的独立改动,验证覆盖 Qwen 路径而非 MiniMax 原生算子,未做性能声明。

## 关联
#63232(拆分来源), #49623(元数据保留修复的扩展), #62313, #55085, #58716, #59432(正文说明与本 PR 无重复的开放修复)

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
