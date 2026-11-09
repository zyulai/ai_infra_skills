# PR #33997: Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @mmangkad |
| 合并 | 2026-08-12 |
| 分类 | Attention / Kernel (tier: major) |
| Label | documentation, high priority, dependencies, run-ci, jit-kernel, release-highlight |
| Diff | diffstat 缺失(pack 记录 +None/-None across None files);pack 保留 diff 仅覆盖 flashinfer 侧文件,35 files dropped |
| 链接 | https://github.com/sgl-project/sglang/pull/33997 |

## 一句话结论

依赖升级到 FlashInfer 0.6.17 并移除 Kimi K3 workaround,带来 CuTe DSL MLA decode 的 DCP 支持与 split_kv/查询布局改进。

## 动机

PR 正文极简,仅一句"升级 FlashInfer 至 0.6.17 并移除 Kimi K3 workaround"。升级的具体动机(修了什么 bug、解锁什么特性)PR 未注明;K3 workaround 移除所涉及的 sglang 侧文件也未保留在 pack diff 中。

## 优化点

从保留的 flashinfer 侧 diff 看,0.6.17 的 CuTe DSL MLA decode 主要变化:新增静态 DCP(context parallel)解码支持(`enable_dcp`/`cp_world`/`cp_rank`/`causal_seqlens_kv_global`),DCP 强制走 monolithic 实现且必须 `return_lse=True`;split_kv 依 `max_seq_len` 归一化为非空 fixed-size K chunk 数,避免 grid 与 reducer 携带空的最终 split;查询 (token, head) 行扁平化进 128 宽 MMA M tile,替代原来的 `fold_sq` 折叠路径,heads 约束从固定 128 放宽到 1-128、q_len 放宽为任意正数;新增 `reducer_d_tiles`(1/2/4)在 row grid 不足以占满一个 wave 时提高 reducer 侧并行度;nonpersistent grid.y 超过 CUDA 65535 上限时显式报错。dispatcher 对 DCP 与 modular-only 特性(sinks)的组合做了严格校验,不兼容即抛 ValueError,不做静默回退。

## 关键改动

```text
flashinfer/cute_dsl/attention/mla_dispatch.py: dispatcher 层 DCP kwargs 校验与实现选择(DCP 强制 monolithic)
flashinfer/cute_dsl/attention/monolithic/mla_decode.py: split_kv 按 max_seq_len 归一化、reducer_d_tiles 选择、DCP 参数接线
flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py: (token,head) 扁平化 M tile 取代 fold_sq,heads 1-128
```

sglang 侧移除 Kimi K3 workaround 的改动未出现在保留 diff 中,具体内容 PR 未注明。

## 性能声明

PR 正文未提供 benchmark 数字,PR 未注明测试条件。

## 验证与回归风险

依赖升级影响所有 FlashInfer 后端用户(CuTe DSL MLA、trtllm-gen 等)。保留 diff 显示新内核对输入约束收紧:query 需满足 stride 布局(否则强制 contiguous)、out/lse 必须 contiguous(非紧凑视图提前报错);workspace 公式改为 `B * 128 * q_tiles * split_kv * (kv_lora_rank + 1) * 4`。PR 页面 CI 状态显示 Base/Extra 测试未全绿(`no_entry_sign` / `x`)。

## 关联

同系列 FlashInfer 升级:#28354(v0.6.16 CuTe DSL NVFP4 MoE)、#22851(v0.6.10 DSA topk backend)、#35120(v0.6.18 CuTe DSL)。
