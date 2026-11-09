# PR #63027: [ROCm][Perf][DSv4.1] Mono: tuned K1/K2 for the TP2 compress-ratio-2 layers on MI355X

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @chao-xu-spec |
| 合并 | 2026-11-05 |
| 分类 | Speculative Decoding \| tier: major |
| Label | rocm, ready, deepseek, DSv4.1 |
| Diff | 素材未提供 diffstat(pack 标注 +None/-None, None files)；pack 末尾注明 6 file(s) dropped from diff |
| 链接 | https://github.com/vllm-project/vllm/pull/63027 |

## 一句话结论

为 MI355X 上 TP2 compress-ratio-2 层调优 K1/K2，层快 8.6-18.1%，E2E TPOT p50 降 3.3%。

## 动机

为 gfx950（MI355X）上 DeepSeek-V4.1-Flash mono decode layer 的 compress-ratio-2 target layers（TP2）做更快的 K1/K2。正文说明用静态 build flag `MonoBuild.tuned` 选择新路径，其输出与 main bit-identical，且其它 build（ratio 0/1、TP4、DSpark draft、gfx942、FFN launch）编译出与 main 相同的 MLIR；每个 commit 独立可 build 且 bit-exact。正文未展开更早的具体瓶颈。

## 优化点

(1) Tuned builds of ratio-2 layers：`MonoBuild.tuned` gate 带独立 scratch、MoE stages 改用 unsigned index math、polled ready flags 复制到单独行、independent loads 提前于 waits 发出；(2) Down stage：两个 down CTAs 通过 LDS 流水各轮，shared-expert down rows 下放到下一个 one-task CTAs；(3) FFN seam 与 attention hand-offs：更短的 seam→norm→router chain、whole-sector attention partials、per-tile `wo_a` ready flags。diff 层面 common/mx.py 新增 udiv/umod（用 shift / mask 或 unsigned division 替代 signed floor division 的 fix-ups）与 oob / oob_rsrc（越界读不产生内存流量）。

## 关键改动

```text
vllm/models/deepseek_v41/amd/mono/attention/back.py、front.py、plan.py: tuned K1/K2、down stage 与 attention hand-offs
vllm/models/deepseek_v41/amd/mono/common/mx.py: udiv/umod 无符号索引数学与 oob 辅助
vllm/models/deepseek_v41/amd/mono/layer.py: MonoBuild.tuned gate 与 scratch 布局
vllm/models/deepseek_v41/amd/mono/README.md、tests/models/test_dsv41_mono_numerics.py: 说明与数值测试
```

```python
def udiv(x, d, unsigned=True):
    if isinstance(x, int) or not unsigned:
        return x // d
    if d & (d - 1) == 0:
        return x >> (d.bit_length() - 1)
    return fx.Int32(fx.Uint32(x) // d)


def umod(x, d, unsigned=True):
    if isinstance(x, int) or not unsigned:
        return x % d
    if d & (d - 1) == 0:
        return x & (d - 1)
    return fx.Int32(fx.Uint32(x) % d)
```

## 性能声明

"Result vs main: ratio-2 layer K1+K2 8.6-18.1% faster (M = 1..48), index-source launches included; E2E TPOT p50 −3.3% (c32) and −3.7% (c4)"。Layer microbench（TP2、ratio 2、K1+K2 µs/layer、cold L2、max over ranks、3 对 GPU 上 11 次 fresh runs 中位）：M=1/6/12/18/24/36/48 时 main 89.1/134.4/194.1/238.2/292.8/402.1/472.2 → PR 73.0/118.6/173.0/208.3/259.1/360.2/431.8，变化 -18.1/-11.7/-10.9/-12.6/-11.5/-10.4/-8.6%；按 c32/c4 的宽度加权 -10.7%/-11.6%。Index-source launches -8.9% 至 -17.3%（加权 -10.9%/-11.4%），`krdy` wait 本身在 main 与 PR 上均 <0.6%。E2E（TP2 + MTP、agentic-coding trace、每 run 1 h、CI server 参数、两 lane 同时跑并在第二波对调）：TPOT p50 -3.31%(c32)/-3.74%(c4)、p90 -1.22%/-6.84%、interactivity p50 +3.49%/+3.78%、output tok/s/GPU +0.53%/-0.66%、per-request ITL ratio 0.973/0.976 与 0.970/0.960；MTP acceptance length 每次 3.51、TTFT 在噪声内。测试条件：8x MI355X(gfx950)、镜像 vllm/vllm-openai-rocm:nightly-rocm100-5d35ce39（AITER 0.1.26、FlyDSL 0.4.0，仅 Python 树不同），模型 DeepSeek-V4.1-Flash，workload 层微基准 M=1..48 与 agentic-coding trace E2E c32/c4 + MTP；正文注明 E2E 数字测于 rebase 之前（相同 kernel code）。

## 验证与回归风险

Repo tests（`pytest -v tests/models/test_dsv41_mono_numerics.py tests/models/test_dsv41_mono_layers.py tests/models/test_dsv41_mono_decode.py`）：4 GPUs 下 main 与 PR 均 80 passed / 2 skipped（gfx942 only），2 GPUs 下均 78 passed / 4 skipped；numerics 测试把 tuned layer 与 generic=True runner 在 12/36/48 rows 上逐 bit 比对。Correctness vs main：TP2 下 115/115 cases bit-identical（92 个在 tuned builds；含全部 K1+K2 输出与 SWA cache、按 rank 哈希、index-source 与 no-fold launches），所有非 tuned build 的 MLIR 与 main 逐字节相同（168/168 gfx950、76/76 gfx942）。回归风险：仅 gfx950 / TP2 / compress ratio 2 / target routing（384 experts、top-6）/ SWA window 生效，gate 为静态、无新 flag 或 env var；tuned builds 的 scratch 布局不同，runner 按 (width, tuned) 键控，最多约 20 MB 每 tuned width 每 rank（peer memory 不变）；gfx942 与 DSpark draft builds 未在 GPU 上跑（MLIR 相同）、TP4 未调优；rebase 到 main 8977b13a1c 含 #63576 / #63689，tuned 路径共享 stage_combine 的 krdy wait 与相同 mx_code。

## 关联

#61482（把这些 kernel 移到 vllm/models/common/amd/mono/）、#63576（K2 的 combine 在 index-source launches 中先等 krdy 再读 klen）、#63689（vLLM 对 zero / subnormal MXFP8 groups 的 scale code）；正文另说明 rebase 到 main 8977b13a1c。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
