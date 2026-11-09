# PR #48666: [Kernel] Gemma-4 FA4 FP8 Kernel

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jhaotingc |
| 合并 | 2026-08-14 |
| 分类 | Speculative Decoding (tier: major) |
| Label | documentation, speculative-decoding, ready, ci/build, v1, mrv2, verified |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 9 个文件(flash-attention tag 升级、FA backend、MTP speculator 等) |
| 链接 | https://github.com/vllm-project/vllm/pull/48666 |

## 一句话结论

把 FA4 的 SM90 FP8-KV-dequant 路径接入 vLLM,让 Gemma-4(full_attention 头维 512)在 SM90 上从 FA3 升级到 FA4 且滑动/全注意力层都用 FP8 KV cache,decode 吞吐较 FP8 TRITON 最高 2.21x。

## 动机

Gemma-4 在 sliding_attention 用 256 宽 head、full_attention 用 512 宽 head;SM90 上 full-attention 层此前只能走 FA3,FP8 KV cache 无法同时覆盖 FA3 滑窗层与 FA4 全注意力层。另外 Gemma-4 MTP 的 BF16 draft 模型用 Q-only attention 读目标模型的 FP8 KV cache,仅 alias cache 张量会让 draft attention 模块保留默认 K/V scale=1.0,读校准后的 FP8 cache 结果错误。

## 优化点

接入 flash-attention#164 的 FA4 FP8-KV-dequant kernel:SM90 上 FP8 KV 在 kernel 内反量化,K/V descale 以 (batch, num_kv_heads) f32 传入,Q 保持 BF16/FP16 原生 dtype(不再要求 quant query input),输出直接写原生 dtype。配套机制:fa_utils 放行 SM90 上 fa_version 3/4 的 fp8 KV cache;FlashAttentionBackend 增加 _get_sm90_fa4_fp8_kv_block_size(SM90+FP8 cache+head_size 512+FA4 时 kernel 使用 64-token TMA tile/page,并据此报告精确 block size 契约),新增 get_*_for_config classmethod 让 block size 选择拿到具体 engine config;MTP 侧 Gemma4Speculator._setup_gemma4_kv_sharing 改为按 layer index 直接解析目标层名,并对每个共享 cache 的 draft 层复制 target 的 device/host K/V scales(register buffer clone,不 alias)。使用约定:FLASH_ATTN BF16 需显式 version=4,FP8 不设或 version=3(仅 head dim 512 的 FP8 全注意力层自动用 FA4)。

## 关键改动

```text
cmake/external_projects/vllm_flash_attn.cmake: vllm-flash-attn tag 升级至 617264c1
vllm/vllm_flash_attn/flash_attn_interface.py: FA4 分支增加 fa4_fp8_kv_dequant(fp8 e4m3 K/V kernel 内反量化,转发 K/V descale)
vllm/v1/attention/backends/flash_attn.py: SM90 FA4 FP8-KV 的 64-token page 契约与 *_for_config block size 接口
vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py: _copy_target_kv_scales,draft 层复制 target K/V scales
vllm/v1/attention/backends/fa_utils.py: SM90 允许 fa_version 3 或 4 的 FP8 KV cache
```

## 性能声明

Serving(4x H200,TP=4,25k shared prefix + 3k unique input,OSL=200,concurrency 1-64):decode tok/s/GPU,FP8 FA4 27.3/186.5/261.6(conc 1/16/64),BF16 FA4 24.3/168.4/241.1,FP8 TRITON 17.9/105.0/118.5,BF16 TRITON 15.0/102.5/127.5;FP8 FA4 比 BF16 FA4 快 8-12%,比 FP8 TRITON 最高快 2.21x。Accuracy(gemma-4-31B-it,4x H200,TP=1/DP=4,max len 262144):AIME 2025 87.29/87.14/86.88、GPQA Diamond 85.42/85.39/86.55、MMLU-Pro 86.44/86.25/86.64、GSM8k 94.62/94.62/94.47(BF16/FP8 TRITON/FP8 FA4),各基准差距 <=1.2pp。MTP(2x H200,TP=2,BF16 四层 assistant、3 draft tokens、conc 32、512 条 SPEED-Bench):FP8+scale 修复后 acceptance 3.6370(BF16 基线 3.6543,差 0.0173 token/0.58pp),输出吞吐 4693.33 vs 4370.34 tok/s(+7.4%),ITL 6.64 vs 7.13ms。

## 验证与回归风险

_accuracy/serving/MTP 三组测试即验证;pre-commit 检查 speculator.py;MTP 两臂各 512/512 请求无错。回归面:FA4 FP8-KV 路径仅限 SM90 + head_size 512;SM100 等 FA4 路径仍要求 Q/K/V 同 FP8 dtype(supports_quant_query_input 在 SM90 FP8-KV 下显式关闭);block size *_for_config 接口改动影响所有 attention backend 的默认 block size 选择链路;BF16 FLASH_ATTN 用户需显式 version=4,FP8 用户不得显式设 version=4,否则行为偏离。

## 关联

上游:vllm-project/flash-attention#164(FA4 FP8-KV-dequant kernel)。同区域:PR #53147(Gemma4 Triton sliding-window 剪枝,本索引库同批次)。
