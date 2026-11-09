# PR #63885: [ROCm][Perf][DSv4.1] Build the SWA decode metadata in one launch

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-10-30 |
| 分类 | Speculative Decoding(major) |
| Label | rocm, ready, ci/build, deepseek, DSv4.1 |
| Diff | diffstat 缺失(素材包未提供 +x/-x);diff 含 3 个 vllm 源文件与 1 个测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/63885 |

## 一句话结论

把 #60731 的单 launch SWA decode metadata 内核从 gfx942 放宽到 gfx950 (MI355X),DSv4.1 decode 每步少 49 次 launch,c1 ITL p50 −100 µs(−1.75%)。

## 动机

--monokernel 下 DeepSeek-V4.1 decode(≤32 行)此前每个 SWA 组要用 vLLM 的 8 次 launch 构建 metadata,共 7 组(4 target + 3 DSpark draft)= 56 次;#60731 的 swa_decode_meta 单 Triton launch 内核只在 gfx942 启用,但该内核没有任何 arch 专属代码,可直接用于 gfx950。本 PR 与并行的 #63994 汇合(采用其 gate)。

## 优化点

ops/__init__.py 新增 swa_decode_meta_enabled():--monokernel 且 ROCm、get_cdna_version() in (3, 4)(即 gfx942 或 gfx950)时为真;swa_decode_meta.build 的门从 enabled() 换成它,record_monokernel 同时清两个 gate 的缓存。其余 gfx942 helper 保持 enabled()。行为变化仅限 gfx950 + --monokernel 的 decode 步(内核最多 32 行;AV off、k=6 时 ≥5 个请求的步保持 vLLM 原 build,AV on 按 32 行截断);prefill、混合批、其他 arch 不变,无新旗标。

## 关键改动

```text
vllm/models/deepseek_v41/amd/ops/__init__.py: 新增 swa_decode_meta_enabled() gate
vllm/models/deepseek_v41/amd/ops/swa_decode_meta.py: build 改用新门
tests/models/test_dsv41_mono_decode.py: 新增 arch × --monokernel gate 测试
```

## 性能声明

DeepSeek-V4.1-Flash,MI355X TP4,--monokernel,DSpark k=6(合成 AL 3.87,AV off),32k in / 1k out:decode launch 数 114→65(−49,torch-profiler 统计);SWA metadata launch 56→7;被计内核 GPU 时间 136.5→36.8 µs/step(profiler 开启下);ITL p50 c1 5.684→5.584 ms(−1.75%)、c2 6.891→6.812 ms(−1.16%);噪声 0.3–1.0%。GSM8K-1319 与 acceptance length(AL 3.577)与 base 一致。

## 验证与回归风险

内核测试 test_dsv41_gfx942_decode_meta.py(-k swa_decode_meta)2 passed 并加入 MI355 CI 步骤;gate 测试 82 passed(fail-before:parent 上 81 passed,1 failed);gate 测试杀死 6/6 mutants;VLLM_GPU_SYNC_CHECK=error 无 sync 报告。风险:仅 monokernel + CDNA3/4 的 ≤32 行 decode 步;AV-on 部署现在也走该路径(已验证精度与 AL);与 #61311 的 monokernel→specialized_decode 重命名冲突,后落地方需 rebase。

## 关联

#60731(内核来源)、#63994(相同改动,本 PR 收敛后可关闭)、#58352(增益重叠,若先落地约剩 20 次 launch)、#61311(monokernel 改名 specialized_decode)、#63486/#63891(draft,CI/文件冲突)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
