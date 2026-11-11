# PR #63893: [ROCm][Perf][DSv4.1] Mono: tuned K1 at TP4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-11-10 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/63893 |

## 一句话结论

DSv4.1 Mono 在 gfx950 的每个 TP4 layer 打开 tuned K1,decode step c1 快 284 µs(−5.2%)。

## 动机

#63027 只在 TP2 对 target 的 compress-ratio-2 layers 调优 K1/K2,TP4 仍跑 generic K1。而其 tuned K1 路径(flag copies、norms/gates 移出 wqkv CTAs、fused fold and slice、wide normed stores、batched norm polls)不读任何 routing dims,在 TP4 上对所有 ratio、fold off、index sources 乃至 DSpark draft 同样适用,却没被启用。部署形态:DeepSeek-V4.1-Flash 低并发、MI355X TP4、`--monokernel`。

## 优化点

新增 `MonoBuild.tuned_front = tuned or (not GFX942 and not generic and tp == 4)`,固定于构建期、无 env var:gfx950 每个 TP4 layer(含 draft)的 K1(attention seam 与 front)启用 tuned 路径;K2 在 TP4 保持 generic(早期实验中 TP4 tuned K2 在 FULL capture 时 hang,该代码已不在 #63027 head),TP8 保持 generic K1(#64152 后续把 gate 扩到 TP8)。TP2、TP8、`generic=True` 与 gfx942 构建不变(TP8 的 161/161 builds 与 main binary-identical)。K1/K2 同 width 共享 scratch layout,K1 regions 跟随 `tuned_front`;runner 按 `(width, tuned, front)` 缓存 scratch,避免与 FFN launch 的 generic layout 混叠,代价每 width 至多 +8,960 B(48 rows:11,153,024→11,161,984 B)。

## 关键改动

```text
vllm/models/deepseek_v41/amd/mono/layer.py: 新增 tuned_front 属性;scratch_layout/scratch_bytes 增加 front 参数;K1 构建改用 tuned_front
vllm/models/deepseek_v41/amd/mono/runner.py: scratch 缓存键改 (width, tuned, front);tuned() 改为 build() 返回 MonoBuild
tests/models/test_dsv41_mono_numerics.py: TP4 也对照 generic runner 做 bit-for-bit 比较;新增 ratio 0 的 7-row 用例
```

```python
@property
def tuned_front(self) -> bool:
    from vllm.platforms.rocm import get_cdna_version

    return self.tuned or (
        get_cdna_version() == 4 and not self.generic and self.tp == 4
    )
```

## 性能声明

ABBA 对 #63027 rebased,4 rounds,4x MI355X TP4 `--monokernel`,DSpark k=6,AV off;random 32k in / 1024 out、`--ignore-eos`、synthetic AL 3.87;decode step(ITL p50):

| 并发(目标 rows) | base | this PR | delta |
|---|---:|---:|---:|
| c1(7 rows) | 5.449 ms | 5.164 ms | −284 µs(−5.2%) |
| c2(14 rows) | 6.720 ms | 6.395 ms | −326 µs(−4.8%) |
| c4(28 rows) | 8.676 ms | 8.482 ms | −194 µs(−2.2%) |

每轮均更快,|mean| > t×SE(t=3.18,df=3)且大于 base A/A spread;c1/c2 为 gate 条件,c4 仅报告。Profiler 归因:7 rows 时 K1 −277.2 µs/step(target −259.9、draft −17.3;占 ABBA step delta 的 98%),14 rows 时 −301.2 µs(93%);K2 +7.5/+28.7 µs。

## 验证与回归风险

输出按设计不变:路径来自 #63027(TP2 bit-exact);numerics test 现也在 TP4 做 bit-for-bit 比较(新增 generic runner 对照),mutant(TP4 K1 residual store 改一个 bf16 ulp)如期 fail。GSM8K-1319(8-shot,greedy,`--parallel 6`):base 0.9185(AL 3.580)vs PR 0.9158(AL 3.583),差 −0.0027,噪声内。Compile-only:TP4 K1 148–174 VGPRs(#63027 rebased 为 142–240)、no spills、LDS 不变。限制:decode step 只 timed 7/14/28 rows(29–48 rows 未 timed);draft K1 无 GPU bit check(acceptance-count 检查被噪声 confound);gfx942 仅 compile-checked;base 是尚未合入的 #63027 rebase。

## 关联

#63027(base,stacked;本 PR 前三个 commit 即其 rebase)、#64152(stacked on this,TP8 tuned K1)、#63892(sibling,K2 侧:batch TP4 K2 的 X8 copies)、#63486(scratch 读取需适配 `(tuned, tuned_front)`)、#61934(K2 FFN-seam hoist 因无 e2e gain 关闭,ABBA gate 的由来)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
