# PR #60229: [Spec Decode][GLM-5.3-Flash] Support adaptive verification

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @jonahsamost |
| 合并 | 2026-10-29 |
| 分类 | Model-specific Fast Path (tier: major) |
| Label | speculative-decoding, ready, deepseek, kimi, k3, glm |
| Diff | diffstat 缺失(pack 未记录 +None/-None,None files);diff 内容可用 |
| 链接 | https://github.com/vllm-project/vllm/pull/60229 |

## 一句话结论

让 GLM-5.3-Flash 的 GDN 与 k-pool 路径支持 GPU 端逐请求选择验证 draft token 数(adaptive verification),深验证(kmax≥4)吞吐提升 6.5%-12.7%。

## 动机

adaptive verification 在 GPU 上为每个请求选择本次验证多少个 draft token;FULL CUDA graph replay 时物理 token buffer 与捕获的控制流固定,而活的请求边界可变。GLM-5.3-Flash 有两个与此不兼容的假设:GDN 捕获的是普通单 token decode 路径,但 replay 可能把 token 行归并为更少请求、每请求多验证 token;k-pool scatter 路径假设每个物理 token 行都属于活请求,而 replayed graph 含额外 padding 行。

## 优化点

机制:GDN 后端声明 `supports_device_cpu_query_lens_mismatch`,允许 replay 时 per-request query 长度由 GPU 决定(`get_varlen_cudagraph_max_query_len` 返回配置的最大 decode query 长度);metadata 保留 CPU batch plan、执行 GPU ragged plan,adaptive 模式下跳过 uniform spec sequence length 优化。GDN 的 speculative path 在 varlen graph 捕获时即按最大验证宽度冻结 spec 路径(捕获 dummy 行是单 token,replay 可能是 k+1 token 行)。k-pool indexer/sparse_indexer 的 decode scatter 把 graph padding token 映射到追加的 dummy request(`idx=num_requests`),intra 索引钳到 0,结果裁掉 dummy 行——全程不把长度同步回 CPU;indexer builder 在 adaptive 下强制 `max_decode_len = next_n`、`write_is_uniform = False`。

## 关键改动

```text
vllm/models/glm5next/common/sparse_indexer.py: scatter 索引引入 dummy request 吸收 graph padding
vllm/v1/attention/backends/gdn_attn.py: device/cpu query len mismatch 支持 + 捕获期走 spec 路径
vllm/v1/attention/backends/mla/indexer.py: adaptive 下 decode 写宽取 next_n 上界
tests/v1/attention/test_sparse_indexer_decode_seq_lens.py: padding/CUDA graph replay scatter 测试
```

## 性能声明

GLM-5.3-Flash + MTP,4x B300(TEP4),kmax=1..5 × 18 个 batch size(B=1..256)× prefix S=512/8192/32768,adaptive 相对同最大深度 fixed verification 的 decode 吞吐(对 (B,S) 取无加权平均):kmax=1 0.965×;kmax=2 0.958×;kmax=3 0.965×;kmax=4 1.065×;kmax=5 1.127×。即浅验证略亏(~3.5-4%),深验证(kmax=4/5)提升 6.5%/12.7%。

## 验证与回归风险

新增测试:GDN fused MTP 的 adaptive ragged/padded 用例(device_query_lens 模拟重分配)、GDN metadata builder 的 device query boundaries 与捕获期走 spec 路径、DeepseekV32 indexer 的 `write_max_decode_len == next_n` 断言、sparse indexer 的 scatter padding(CPU)与 CUDA graph replay 后 live budget 变小(设备端);kimi_k3 KDA metadata 测试补充 `enable_adaptive_verification=False`。风险面:仅 `enable_adaptive_verification=True` 时行为改变(默认关闭);触及 GDN(GLM/Mamba 系)与 k-pool/sparse indexer(GLM-5.3-Flash)两条路径;uniform spec sequence length 优化在 adaptive 下被跳过,非 adaptive 路径不变。

## 关联

同区域:#58454(kpool tail ring corruption 修复,同属 GLM-5.3-Flash kpool/indexer 区域)、Kimi K3/KDA spec-decode 相关 PR(labels 覆盖 deepseek/kimi/k3)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
