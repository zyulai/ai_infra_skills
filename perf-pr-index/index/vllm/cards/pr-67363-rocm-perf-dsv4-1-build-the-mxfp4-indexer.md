# PR #67363: [ROCm][Perf][DSv4.1] Build the MXFP4 indexer decode metadata in one launch

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @okorzh-amd |
| 合并 | 2026-11-19 |
| 分类 | Quantization(tier: major) |
| Label | rocm, ready, deepseek |
| Diff | diffstat 缺失(pack 未提供 +/− 统计) |
| 链接 | https://github.com/vllm-project/vllm/pull/67363 |

## 一句话结论

ROCm 上 DeepSeek-V4/V4.1 MXFP4 稀疏 indexer(gfx950)的 decode metadata 原先每步用一串小的 eager torch op 构建,本 PR 改为每个 indexer KV group 一次 Triton launch(`RocmMxfp4DecodeRowsKernel`),每步 builder GPU 时间 C1 下 38.6 → 16.2 us(-58%),TP8 1K/1K C1 ITL p50 降 23 us(-0.55%),精度不变。

## 动机

`RocmMxfp4IndexerMetadataBuilder` 在 graph replay 前的每个 decode 步都用一串小的 eager torch op 构建 decode metadata,launch 数多、builder GPU 时间长(C1 每步 17 次 launch / 38.6 us;adaptive verification 56 rows 时 49 次 / 99.3 us)。目标是在不改变 CUDA 与其他 builder 行为的前提下,把这条路径压成单次 Triton launch。

## 优化点

- 共享的 `DeepseekV32IndexerMetadataBuilder` 加文档化的 `_build_decode_rows()` 钩子(默认返回 None)与 `should_build_custom_decode_rows()`;`build()` 在"非 varlen decode、非 DCP"的 decode 步调用它,返回 None 时保持原路径,故 CUDA 等其他 builder 完全不变;varlen decode 与 DCP 跳过钩子。
- ROCm 侧 `RocmMxfp4IndexerMetadataBuilder` 覆写钩子:一次 `RocmMxfp4DecodeRowsKernel`(ROCm-only ops 模块)写出 `decode_seq_lens_buffer`、`expanded_block_table_buffer`、`decode_lens_buffer`、`per_req_decode_lens_buffer`、`context_lens_buffer`、`decode_block_ends_buffer`,并返回三个 view。
- Reader 读到的整数与 main 相同;只有从不被读的 padding 行不同——被清零。
- JIT warmup:`enable_jit_warmup` 时对 kernel 注册 warmup(compress_ratio 与 block_cols 为 compile key);fused 时跳过 `torch.floor_divide` 的 context_lens 计算与 `decode_block_ends` 的 add/floor_divide。

## 关键改动

新 kernel 声明(照抄 pack diff):

```python
# vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py
class RocmMxfp4DecodeRowsKernel(
    VllmTritonJitKernel["RocmMxfp4DecodeRowsKernel.CompileKey"]
):
    TAIL_BLOCK = 256

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        compress_ratio: int
        block_cols: int
```

kernel 主体(照抄 pack diff):

```python
        pid = tl.program_id(0)
        if pid < num_rows:
            req = _token_request(query_start_loc_ptr, tl.full((1,), pid, tl.int32), num_reqs)
            req = tl.sum(req, 0)
            valid = pid < tl.load(query_start_loc_ptr + num_reqs)
            start = tl.load(query_start_loc_ptr + req)
            end = tl.load(query_start_loc_ptr + req + 1)
            seq_len = tl.load(seq_lens_ptr + req)
            length = tl.maximum(seq_len - (end - start) + pid - start + 1, 0)
            length = tl.where(valid, length, 0) // COMPRESS_RATIO
            tl.store(row_lens_ptr + pid, length)
            tl.store(decode_lens_ptr + pid, 1)
```

```diff
 # vllm/v1/attention/backends/mla/indexer.py:build() 走 custom_rows 分支
+            custom_rows = self.should_build_custom_decode_rows(common_attn_metadata)
+            if not (self.supports_varlen or custom_rows):
                 torch.diff(
```

## 性能声明

- Builder 微基准(1 GPU、2 轮,60 步中位数,真实 aiter cache 布局):C1(1 req x 7 rows, 32K ctx)17 → 4 次 launch,GPU 39.2 → 16.0 / 38.0 → 16.3 us,均值 -22.5(-58%),host 228 → 152 us;adaptive verification 56 rows 49 → 6 次,97.3 → 22.9 / 101.2 → 22.8 us,均值 -76.4(-77%);336 rows 47 → 6 次,119.1 → 30.8 / 116.2 → 30.9 us,均值 -86.8(-74%)。
- TP8 1K/1K 服务(DeepSeek-V4.1-Flash,MI355X,`vllm/vllm-openai-rocm:nightly-rocm100-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d`,ABBA 4 台 server):C1 ITL p50 4.219 → 4.195 ms(-23 us,-0.55%,A/A gap 0.02%);按 prompt 配对 -23.8 us [95% CI -36.4, -11.2],16 个 prompt 中 14 个更快;C2-C64 无减速但增益未被解析(-1.8% 至 +0.1%,低于 0.45-2.6% 的 A/A gap)。
- 32K 端到端(3 reps x 6 prompts):全部 18 对 ITL p50 4.2512 → 4.2197 ms(-31.5 us,-0.74%);8 对同输出对 -25.8 us(-0.61%)[-27.1, -24.5]。torch profiler:每步 builder launch 17 → 4,72.2 → 18.7 us/步(all-rank median,profiler 膨胀)。
- 精度:GPQA-diamond 86.11% → 86.11%(配对 +0.00 pts,McNemar p = 1.0);RULER 32K-128K 4 任务 2640 items +0.04 pts [-0.24, +0.33],McNemar p = 1.0;GSM8K-1319 C64 91.74% → 91.58%(-0.15 pts [-1.45, +1.15],p = 0.908)。

## 验证与回归风险

- 新测试 `test_decode_metadata_matches_torch_path` 对 `DeepseekV41RocmMxfp4IndexerMetadataBuilder`(ratio 2/1)与 `RocmMxfp4IndexerMetadataBuilder`(ratio 4)、uniform/adaptive 两种步型,逐 buffer 与 torch 路径做 `torch.equal` 断言(含 data_ptr 一致)。
- 风险:TPOT/吞吐列受 DSpark greedy 输出不可复现影响(两台同臂 server 的 AL 相差最多 19%),表中 C1 的 ITL 增益是唯一同时清过两侧 ABBA gap 的格子;TP decode-row shard 在约 23% 的 sweep decode build 上生效(仅 C16-C64 到 64 rows)。

## 关联

- 与 #66047 的 TP decode-row shard 交互(GSM8K C64 触发该 shard);复用 `vllm/v1/attention/ops/rocm_aiter_mla_sparse` 的 `_token_request` 等。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
