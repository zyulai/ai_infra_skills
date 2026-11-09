# PR #44657: [Perf] [AMD] dsv4: Add a separate MegaMoE decode instance

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @RolaoDenthu |
| 合并 | 2026-10-16 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | amd, run-ci |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files),diff 内容可用 |
| 链接 | https://github.com/sgl-project/sglang/pull/44657 |

## 一句话结论

给 AMD FlyDSL MegaMoE 增设小容量 decode 实例,小 token 数下每层 MoE 时间最多降至 1.31x,e2e TTT 约 -3%。

## 动机

Decode 此前与 prefill 共享按 prefill 规模配置的 MegaMoEV2 实例。即使 decode batch 很小,大 MTPR(max tokens per rank)也使其走大实例的 dispatch/combine 路径,无法享受针对小 token 数优化的 kernel 配置。因此引入一个更小的 decode 专用实例,启用 small-MTPR 执行路径,同时保留主实例服务 prefill。

## 优化点

新增环境变量 `SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR`(默认 0,关闭)。开启后维护两个缓存的 `MegaMoEV2` 实例:主实例用 `SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR`(默认 8192)服务含 prefill 的 step;decode 实例用更小的 MTPR 服务 decode-only step 中的 decode、target verify 与 draft extend。两个实例在 graph capture 之前按固定顺序在所有 EP rank 上一起构建(构建会同步设备并 barrier),各自持有独立通信缓冲与同步状态,通过 per-layer rebinding 共享模型权重。实例选择只看全局同步的 step 类型(`is_extend_in_batch` 在 DP 组 max-reduce;capture 模式直接选 decode 容量),与各 rank 本地 batch 大小、graph/eager 无关,保证跨 rank 线上协议一致。启动校验要求 decode MTPR 为不大于主 MTPR 的 2 的幂,且覆盖单 rank 最大 decode workload(max running requests x 每 request token 数),不支持 attn-cp-size>1。

## 关键改动

```text
python/sglang/srt/layers/moe/mega_moe_flydsl.py: _decode_mtpr/_capacities/_select_mtpr 与双容量实例构建、权重 rebind
python/sglang/srt/arg_groups/mega_moe_hook.py: validate_flydsl_decode_capacity 启动容量校验
python/sglang/srt/environ.py: 注册 SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR(EnvInt(0))
test/registered/unit/layers/moe/test_mega_moe_decode_capacity.py: 容量选择/构建顺序/启动校验单测
```

```python
def _select_mtpr(forward_batch) -> int:
    mtpr = _mtpr()
    decode_mtpr = _decode_mtpr()
    if not decode_mtpr or not _is_decode_step_mode(forward_batch):
        return mtpr
    if get_is_capture_mode():
        return decode_mtpr
    return mtpr if get_is_extend_in_batch() else decode_mtpr
```

## 性能声明

Kernel 级:MegaMoEV2 每层时间(µs,8 rank 取 max;MI355X,EP8,DeepSeek-V4-Pro shapes,a8w4,uniform routing,CUDA-graph;相对 16384 compact 列):512 fixed-slot 实例在 7 tokens 时 254 µs(1.31x)、56 tokens 343 µs(1.31x)、112 tokens 362 µs(1.30x)、224 tokens 449 µs(1.28x)、448 tokens 647 µs(2% slower)、672 tokens 装不下;对照 8192/16384 compact 列分别为 331-333/331-333、454/450、472/469、566/573、632/633、797/790 µs。

E2E:cc=256 的 agentx benchmark 上 TTT 约 -3%。精度:gsm8k 1319 得分 0.939。

## 验证与回归风险

新增 `test_mega_moe_decode_capacity.py`(CPU 单测,register_cpu_ci):覆盖跨 rank 一致容量选择、decode step 忽略本地 token 数与 graph bucket、draft extend/IDLE 共享 decode 容量、任一 rank 有 extend 即整 step 回大容量、首次调用按序构建全部容量、权重 rebind 不复制、禁用/非法容量行为、启动校验(需 --max-running-requests、拒绝 attn-cp>1)。风险:默认关闭;仅作用于 HIP + FlyDSL MegaMoE 路径;decode 容量不足会在运行时 assert 失败而非静默切大容量(测试明确此语义);448 tokens 处有小幅回退(2% slower);Checklist 各项未勾选,CI 状态显示 Extra/AMD ROCm 尚未通过。

## 关联

属于 AMD FlyDSL MegaMoE(`SGLANG_AMD_USE_FLYDSL_MEGA_MOE`)路径的容量扩展;正文未列出其他关联 PR 编号。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
