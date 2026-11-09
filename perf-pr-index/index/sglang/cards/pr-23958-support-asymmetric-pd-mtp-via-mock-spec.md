# PR #23958: support asymmetric pd-mtp via mock spec hidden

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @hnyls2002 |
| 合并 | 2026-04-28 |
| 分类 | Speculative Decoding (tier: major) |
| Label | (none) |
| Diff | diffstat 缺失(pack 未提供 +/− 行数与文件数) |
| 链接 | https://github.com/sgl-project/sglang/pull/23958 |

## 一句话结论

统一 PD 分离 metadata 缓冲尺寸,打通 prefill 无 spec、decode 跑 EAGLE/MTP 的非对称配置。

## 动机

PD 分离架构下,`MetadataBuffers` 的 hidden_states 缓冲此前按本地 `--speculative-algorithm` 条件分配:decode 侧开 EAGLE 时按 `spec_hidden_size` 分配,未开 spec 的一侧只分配 16 字节的 RDMA padding。当 prefill 与 decode 配置非对称(如 prefill 纯 TP 不跑 spec、decode 跑 EAGLE MTP)时,两侧 wire 布局尺寸不匹配,传输直接不可用——PD ferry 此前隐含假设两侧对称配置 spec。这导致非对称部署(为吞吐优化 prefill、为低延迟优化 decode)无法享受 MTP 加速。

## 优化点

把 prefill 与 decode 两侧的 `MetadataBuffers` 一律按 `model_config.spec_hidden_size` / `model_config.dtype` 分配,保证 wire 布局两侧始终对齐。prefill 侧没有 spec 模块时,发送零初始化的 hidden buffer;decode 侧把它当作第一步 draft 的 mock conditioning 消费。已验证 speculative decoding 的输出 token 保持正确——只有第一次 decode 迭代的 accept length 受影响。代价是当两侧都不跑 spec 时,该缓冲会被分配但不使用(约几 MB);相比备选方案(为非对称配置定制 wire 协议尺寸),这是一个更小的代价。测试确认无需新增 server arg。

## 关键改动

```text
python/sglang/srt/managers/scheduler.py: init_disaggregation 中 prefill/decode 两侧
  MetadataBuffers 去掉按 spec_algorithm.is_eagle()/is_standalone() 的条件分支,统一
  hidden_size=model_config.spec_hidden_size、hidden_states_dtype=model_config.dtype
test/registered/8-gpu-models/test_dsv4_pd_disagg_nixl.py: 改为非对称配置——prefill
  纯 TP(移除全部 speculative-* 参数),decode TP+DP+EAGLE MTP(num_draft_tokens=4)
```

```python
# scheduler.py(两侧统一后的分配)
self.disagg_metadata_buffers = MetadataBuffers(
    buffer_size,
    hidden_size=model_config.spec_hidden_size,
    hidden_states_dtype=model_config.dtype,
    custom_mem_pool=...,
)
```

## 性能声明

PR 正文给出的唯一数字:verify 保持输出 token 正确,仅第一次 decode 迭代 accept length 受影响,长生成下 throughput 损失约 "< 1%"。本 PR 是打通性修复,无加速倍数声明;正文未注明正式 benchmark 的模型/硬件/workload 条件(测试上下文为 DSv4 Flash PD-disagg + NIXL 传输后端)。

## 验证与回归风险

依赖已注册的 `test/registered/8-gpu-models/test_dsv4_pd_disagg_nixl.py`(8-GPU、DSv4 Flash、NIXL 后端),该测试由对称 spec 配置改为非对称配置以覆盖新路径。回归风险:所有 PD 分离路径的 metadata 缓冲都变大——即使两侧均无 spec 也按 spec_hidden_size 分配,付出约几 MB 的额外缓冲;非对称路径下第一步 draft 的接受率依赖零初始化 buffer 的 mock 语义,若 spec_hidden_size 未来与 wire 协议再有偏差会重新破坏对齐。

## 关联

Follow-up to #23918(正文注明)。同区域(PD 分离 + spec)其余 PR 编号 pack 未提供,无。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
