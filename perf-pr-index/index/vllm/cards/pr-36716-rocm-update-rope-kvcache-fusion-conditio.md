# PR #36716: [ROCm]: Update rope+kvcache fusion conditions and disable custom op by default

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @Rohan138 |
| 合并 | 2026-03-25 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, rocm, ready |
| Diff | diffstat 缺失(pack 未提供加减行数);diff 实际改动 5 个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/36716 |

## 一句话结论

实测 vLLM RoPE custom op 在 MI355X 上最高回退 5%,默认关闭该 custom op 并把 fuse_rope_kvcache 从 O1 移到 O2,修正其启用条件。

## 动机

PR #35601 引入 ROCm/AITER 的 RoPE + KV cache 写入融合(`fuse_rope_kvcache`),依赖 vLLM RoPE custom op。进一步测试发现,vLLM RoPE custom op 本身在 MI355X 上有最高 5% 的性能回退;该融合此前是随 `rotary_embedding` custom op 默认开启而被动开启的。在拿到 "common head sizes x (torch-native RoPE, vLLM RoPE, AITER RoPE)" 的微基准数据之前,先默认关掉,避免默认配置吃亏。

## 优化点

三处机制调整:(1) `vllm/platforms/rocm.py` 中删除 "AITER 启用时自动追加 `+rotary_embedding` custom op" 的平台默认逻辑,即默认不再启用 vLLM RoPE custom op(用户仍可显式开启)。(2) `vllm/config/vllm.py` 的默认 pass_config 中 `fuse_rope_kvcache` 直接置 False(原为动态的 `enable_rope_kvcache_fusion`);同时 `enable_rope_kvcache_fusion` 的条件修正为 Inductor graph partition 开启、或 splitting_ops 不包含 kv_cache_update ops——为此新增 `splitting_ops_contain_kv_cache_update()`,处理 Dynamo 分区且 splitting_ops 为 None 时 kv_cache_update ops 会被后补进 splitting_ops 的边界情况(issue #33267)。(3) `vllm/config/compilation.py` 中当 splitting_ops 为 None 且未启用 Inductor graph partition 时,自动禁用 `fuse_rope_kvcache` 并输出警告;`fuse_rope_kvcache` 从 O1 移入 O2(文档同步)。另删除 `set_splitting_ops_for_v1` 中仅限 VLLM_COMPILE mode 的断言。

## 关键改动

```text
vllm/platforms/rocm.py: 删除 AITER 下默认追加 +rotary_embedding custom op 的逻辑
vllm/config/vllm.py: 默认 fuse_rope_kvcache=False;enable_rope_kvcache_fusion 条件改用 splitting_ops_contain_kv_cache_update()
vllm/config/compilation.py: splitting_ops 为 None 且无 Inductor 分区时自动禁用 fuse_rope_kvcache 并警告;新增 splitting_ops_contain_kv_cache_update()
docs/design/fusions.md / optimization_levels.md: fuse_rope_kvcache 改为 O2(ROCm/AITER only),收益标注 2-4%
```

关键代码摘录(节选,<=15 行):

```python
if not self.use_inductor_graph_partition:
    if self.pass_config.fuse_rope_kvcache:
        logger.warning_once(
            "fuse_rope_kvcache is enabled, but splitting_ops is None "
            "and Inductor graph partition is not enabled."
            "Disabling fuse_rope_kvcache. ...")
        self.pass_config.fuse_rope_kvcache = False
    self.splitting_ops.append("vllm::unified_kv_cache_update")
    self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
```

## 性能声明

- vLLM RoPE custom op 在 MI355X 上存在最高 5%(up to 5%)的性能回退(PR 正文,未附具体基准表)。
- 文档中将 fuse_rope_kvcache 融合的收益从 TBD 更新为 2-4%。
- 本 PR 为回退/禁用性改动,pack 未提供禁用前后的 A/B 数字。

## 验证与回归风险

- pack 中 Test Plan / Test Result 为空,未列明测试;验证主要依赖后续微基准计划(尚未进行)。
- 影响路径:ROCm + AITER 用户默认行为变化——不再默认走 vLLM RoPE custom op 与 rope+kvcache 融合,回落 torch-native RoPE;显式设置 `+rotary_embedding` 或 `fuse_rope_kvcache=True` 的用户不受影响,但 fused 模式下若 splitting_ops 为 None 且无 Inductor 分区会被强制关闭并警告。torch.compile/splitting_ops 交互逻辑有配套修正,错误配置从静默变为警告+禁用。

## 关联

- 前序:PR #35601(引入 RoPE+KV cache 融合);相关 issue #33267。
- 同区域:索引库已有 pr-42832(ROCm GPT-OSS fuse rope/static q/fp8 quant)、pr-16457(rotary emb 优化)等 RoPE 相关卡片。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
