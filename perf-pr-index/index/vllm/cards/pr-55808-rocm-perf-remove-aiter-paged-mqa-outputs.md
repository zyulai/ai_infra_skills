# PR #55808: [ROCm][Perf] Remove AITER paged-MQA outputs guard for DeepSeek-V4

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @shen-shanshan |
| 合并 | 2026-09-08 |
| 分类 | Attention / Kernel |
| Label | rocm, deepseek, DSv4 |
| Diff | diffstat 缺失(pack 未提供 +N/-N 与文件数;diff 内容可用,覆盖 2 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/55808 |

## 一句话结论

移除 DeepSeek-V4 AITER paged-MQA logits 之后的 nan_to_num 保护内核,高并发吞吐最高 +2.06%。

## 动机

为修复一个 NaN 输出问题,vLLM #49714 在 `deepgemm_fp8_paged_mqa_logits` 调用之后额外加了一次 `out_logits.nan_to_num_(float("-inf"))` 逐元素内核。这是纯防御性开销:每次 sparse MLA decode 都多扫一遍 logits 张量。该 NaN 问题后来已在 AITER 侧修复(ROCm/aiter#4774,已包含在 `v0.1.21.post1`),因此这个 guard 可以移除。

## 优化点

单一改动:删除 `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` 中 `rocm_fp8_paged_mqa_logits`(GFX950 路径)返回前的 `out_logits.nan_to_num_(float("-inf"))`,让 AITER 内核直接返回 logits,省去一次逐元素内核 launch 与内存遍历。同时删除专门验证该 guard 的单测 `test_paged_mqa_logits_do_not_contain_nan`——该测试通过 monkeypatch 令 AITER 内核填 NaN 再断言 guard 清除,前提(AITER 会产出 NaN)已随上游修复不再成立,保留意义不大。

## 关键改动

```text
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py: 删除 out_logits.nan_to_num_(-inf) 后处理
tests/kernels/attention/test_rocm_triton_attn_dsv4.py: 删除对应 NaN guard 单测
```

```python
            KVBlockSize=block_size,
            WavePerEU=2,
        )
-       out_logits.nan_to_num_(float("-inf"))
        return out_logits
```

## 性能声明

(SA InferenceX `8k1k` workload;功能性测试为 DeepSeek-V4-Pro,TP8,`--moe-backend aiter`,kv-cache fp8,FULL_AND_PIECEWISE cudagraph,64 并发 gsm8k 20-shot 请求全部正确处理;ROCm 硬件型号未注明)

| 并发 | Output token throughput | Mean TPOT | Mean TTFT |
| --- | --- | --- | --- |
| 1 | 62.04→62.03 tok/s(-0.02%) | 15.70→15.63ms(-0.45%) | 405.86→471.35ms(+16.14%) |
| 4 | 202.34→202.94(+0.30%) | 18.77→18.70(-0.37%) | 498.23→507.68(+1.90%) |
| 16 | 560.66→567.07(+1.14%) | 26.78→26.47(-1.16%) | 806.20→793.03(-1.63%) |
| 64 | 1093.95→1116.49(+2.06%) | 55.60→54.43(-2.10%) | 1757.46→1760.17(+0.15%) |

作者结论:并发 4/16/64 输出吞吐分别 +0.30%/+1.14%/+2.06%,并发 1 基本不变;TPOT 各档均降且高并发降幅更大,说明负载越重收益越明显。

## 验证与回归风险

功能验证:DeepSeek-V4-Pro 服务端 64 并发、3 波共 192 请求全部完成、无失败。回归风险集中在 GFX950/DeepSeek-V4 的 AITER sparse MLA 路径:若用户 AITER 版本低于 `v0.1.21.post1`(不含 ROCm/aiter#4774 修复),被删除的 NaN guard 不会兜底,理论上可能复现原始 NaN 问题(PR 未注明对 AITER 最低版本有无强制校验)。TTFT 在并发 1 处 +16.14% 的波动 PR 未解释。

## 关联

#49714(引入该 guard 的前序 PR)。同区域为 DeepSeek-V4 ROCm AITER attention/MoE 优化线;pack 素材未列其他关联 PR。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
