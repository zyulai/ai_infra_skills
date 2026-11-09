# PR #33771: [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @aabbccddwasd |
| 合并 | 2026-02-08 |
| 分类 | Speculative Decoding (tier: major) |
| Label | ready, v1, nvidia |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 仅含 vllm/v1/attention/backends/flashinfer.py 一处改动 |
| 链接 | https://github.com/vllm-project/vllm/pull/33771 |

## 一句话结论

修复两个回归提交使 GLM-4.7-GPTQ decode 吞吐从 77 tps 恢复到 95 tps,MTP 接受率恢复约 30% 提升。

## 动机

近期两个 commit 在 GLM-4.7-GPTQ-INT4-INT8MIX 上引入性能回归:e0327c9db(PR #31773)造成 decode 性能回退;654a71fc3(PR #32805)造成 MTP(Multi-Token Prediction)接受率回退。该模型主体为 int4/int8 混合量化,MTP 模块(第 92 层)被手动配置为未量化 BF16 以提高接受率,回归后内核选择出错,未量化 MTP 不再走 fused_triton kernels。

## 优化点

回退两个回归提交的相关改动,恢复正确的 kernel 选择:量化层继续走 Marlin kernels,未量化的 BF16 MTP 恢复使用 fused_triton kernels。可见 diff 中唯一改动在 flashinfer attention backend 的 build():把 `common_attn_metadata.seq_lens.cpu()`(async 模式下昂贵的 GPU→CPU 同步拷贝)改为直接复用已有的 `common_attn_metadata.seq_lens_cpu`。注意:pack 提供的 diff 中看不到对 #31773/#32805 的完整回退内容,仅此一处 hunk,实际合并内容以 PR 页面为准。

## 关键改动

```text
vllm/v1/attention/backends/flashinfer.py: seq_lens_cpu 改为复用 common_attn_metadata.seq_lens_cpu,避免 needs_seq_lens_cpu 时的 .cpu() 同步拷贝
(对 #31773 / #32805 的回退改动在可见 diff 中未展示)
```

## 性能声明

测试条件:4x RTX PRO 6000 BLACKWELL,模型 GLM-4.7-GPTQ-INT4-INT8MIX(未量化 BF16 MTP)。v0.13.0(无回归)+ quantized MTP 为 90 tps;基线 v0.13.0 + BF16 MTP 为 95 tps;引入 e0327c9db 后 86 tps(-9.5%);引入 654a71fc3 后 77 tps(-19.0%);revert 后恢复 95 tps。MTP 接受率恢复约 30% 提升。作者注明:数据在未使用优化 Triton 配置文件的情况下测得,后续还有额外提升空间。

## 验证与回归风险

PR 内验证点:内核选择经检查(量化层 Marlin、未量化 MTP fused_triton)。正文给出 Test Plan 但未列 CI 测试清单。回归面:flashinfer backend 中 seq_lens_cpu 的获取路径(受 use_dcp / cascade / 非 trtllm decode 条件控制),异步调度模式下该 CPU 拷贝被绕过,需确认调用方此时确实已有现成的 seq_lens_cpu。

## 关联

被回退:PR #31773(commit e0327c9db)、PR #32805(commit 654a71fc3)。
