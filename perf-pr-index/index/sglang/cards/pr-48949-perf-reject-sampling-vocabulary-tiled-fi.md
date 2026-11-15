# PR #48949: [Perf] reject_sampling: Vocabulary-tiled final-token pass at small batch

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @rohithrr9 |
| 合并 | 2026-11-12 |
| 分类 | Speculative Decoding(tier: major) |
| Label | run-ci, jit-kernel |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/sgl-project/sglang/pull/48949 |

## 一句话结论

`speculative_sampling_classic_kernel` 的 final-token pass 在小 batch 下按 (requests x vocabulary tiles) 网格拆成两 kernel:GB300 上 202,048 词表 batch 1–16 每 verify step 72–113 µs 降到 17–23 µs(最高 6.65x),端到端 decode step −2.4%/−1.1%,输出与 stock 路径 bit-for-bit 一致。

## 动机

`speculative_sampling_classic_kernel` 每请求一个 program,该 program 要把整个词表扫两遍:先求 residual 分布的和,再走它的 CDF。一次 launch 在 1 个请求和很多请求时代价差不多;小 batch 时设备近乎闲置,少数 program 各自读几十万条目。202,048 词条表下,GB300 上 batch 1–16 这个 final-token pass 每 verify step 花 72–113 µs。

## 优化点

- split 路径在 (requests x vocabulary tiles) 网格上做同样的算术:`_split_sum_kernel` 重复 chain walk 并规约一个 tile;`_split_cdf_kernel` 按 tile 顺序加上各 tile 的和,恢复 normalizer 与自己 tile 之前的 running sum,再套用 stock 的 tile step。stock 循环取第一个命中的 tile 及其第一个命中下标(即该行最小命中下标),所以用 atomic 保持跨 tile 最小值即可复现 stock token。
- stock kernel 不变。launcher 仅在 CUDA/ROCm 设备、`block_verification` 关闭、词表跨多于一个 tile、且 batch 小于 multiprocessor 数时走 split 路径。
- 每行用来选出收尾 program 的 arrival counter 由该 program 自行重置,scratch 可按 stream 顺序复用、launch 可安全 capture。scratch 按 (device, stream, tile count) 建一次、每 multiprocessor 一行(gate 的 batch 上界)、永不替换,captured graph 里的指针保持有效;无 scratch 的 stream 上 capture 则保留 stock kernel。
- 新增 `test_reject_sampling_split.py`:各输出 buffer 与 stock 路径 bit-for-bit 对照(跨 batch、chain depth、词表大小,含 129,280 词条表),batch 1 建的 scratch 能服务 gate 拆分的最大 batch 且不被替换,非 CUDA 设备保留 stock kernel。

## 关键改动

```text
python/sglang/kernels/ops/speculative/reject_sampling.py: 新增 _split_chain_walk/_split_residual_tile/_split_sum_kernel/_split_cdf_kernel/_split_scratch/_should_split_final_sampling/_chain_speculative_sampling_split;chain_speculative_sampling_triton 在 gate 命中时走 split
test/registered/kernels/ops/speculative/test_reject_sampling_split.py(新增): 与 stock 路径逐位对照 + scratch/graph/非 CUDA 设备用例,注册 1-gpu-small
```

```python
    tl.atomic_min(Result + pid, v_start + match_idx, sem="relaxed", scope="gpu")

    tl.debug_barrier()
    arrived = tl.atomic_add(Arrived + pid, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_TILES - 1:
        final_token = tl.atomic_add(Result + pid, 0, sem="relaxed", scope="gpu")
        tl.store(Predicts + last_idx, final_token)
        tl.atomic_xchg(Arrived + pid, 0, sem="relaxed", scope="gpu")
```

## 性能声明

- Kernel 微基准(每路径一次完整 `chain_speculative_sampling_triton` 调用,每请求 8 slots:root + 7 drafts,fp32 target/draft 概率,GPU 时间用 CUDA events 且每次调用排在比其 host launch 更长的 GPU sleep 之后以排除 launch 开销,路径交替、取中位数;GB300 152 SMs,L2 warm):词表 202,048 时 batch 1: 86.7→17.0 µs(5.09x)、2: 72.3→17.0(4.25x)、4: 113.2→17.0(6.65x)、8: 107.1→17.0(6.29x)、16: 113.2→23.2(4.89x)、32: 111.2→31.4(3.55x)、64: 165.6→53.9(3.07x)、96: 189.1→80.5(2.35x);词表 262,144 时 batch 1: 62.1→17.0(3.65x)、2: 127.6→17.0(7.50x)、4: 118.4→17.0(6.96x)、8: 135.8→19.1(7.12x)、16: 139.9→25.2(5.55x)、32: 148.6→34.4(4.32x)、64: 219.8→66.2(3.32x)、96: 241.3→96.9(2.49x)。
- 端到端(GB300 x4,镜像 `lmsysorg/sglang:nightly-dev-cu13-20261106-858073e4` 加本 PR 文件,`meta-models/Muse-Spark-Open-816B-A42B-NVFP4` TP4/EP4,DFLASH 8 draft tokens,real acceptance,sampled verification T=1.0;`--dataset-name random-ids --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 --temperature 1.0 --flush-cache`,server `--max-running-requests 16`):decode step 中位数 concurrency 1: 9.055→8.834 ms(−2.4%;两 repeat −0.9%/−4.3%),concurrency 4: 14.680→14.521 ms(−1.1%;−0.2%/−3.0%);预期收益为 batch 1–16 每 verify step 省 55–95 µs,即此处一步的 0.4–1%。两台 stock server 同法互差 0.6% 以内;同 server 两 repeat 间差距可达 2.7%。
- GSM8K(sgl-eval,1319 题,top-p 1.0,max 16384 tokens,seed 42,stock 与 PR 交替):max-running 4: 97.42% → 97.65%;16: 98.03% → 97.57%。

## 验证与回归风险

`test_reject_sampling_split.py` 在 GB300 上 51 passed(含 review fixes);launcher 的另一测试 `test_block_verification.py` 25 passed;batch 1 capture 的 CUDA graph 在以最大 split batch 的 eager 调用之后 replay,每次 replay 与 stock kernel 一致;基准在 16 个配置 x 7 种输入变体(随机接受、drafter 70% 提议 target top token、全接受、首个 draft 被拒、draft 行等于 target 行(零 residual, last-index fallback)、final coin 0 与接近 1)上检查 `predicts`、`accept_index`、`accept_token_num` 全部相等。风险:两 kernel 结构多一次 launch 与 tile-sum round trip,gate 保证只在 stock 网格填不满设备时启用;scratch 常驻(每 (device, stream, tile count) 一份,rows = SM 数)。

## 关联

无直接 PR 引用(验证环境为 DFLASH sampled verify;正文说明该 pass 是 verify step 的组成部分)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
