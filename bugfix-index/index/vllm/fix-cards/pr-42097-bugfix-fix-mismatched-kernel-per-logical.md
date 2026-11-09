# PR #42097: [Bugfix] Fix mismatched kernel-per-logical blocks in NIXL HMA transfer

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#42097](https://github.com/vllm-project/vllm/pull/42097) |
| 作者 | ZhanqiuHu |
| 合并日期 | 2026-05-12 |
| labels | bug, kv-connector, ready, v1 |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Attention / Kernel |
| bug 类型 | correctness |

## 一句话结论
NIXL HMA 传输用本地而非远端 physical_per_logical 展开 FA 块,异构 TP + Mamba 混合模型静默精度损坏;改用远端比例并增加块数对齐与握手校验。

## 问题现象
正文未描述具体报错或日志;症状是异构 TP(heterogeneous TP)+ Mamba 混合模型下 KV 缓存传输出现**静默精度损坏**(silent accuracy corruption)——不崩溃但结果错误。diff 测试中的具体参数(照抄):Qwen3.5-0.8B 4P2D(kernel_block_size=64)时 prefill TP=4 → physical_per_logical=6,decode TP=2 → =10;Nemotron-3-Nano-30B-A3B 4p1d(P_TP=4, D_TP=1)时 remote=34、local=66,FA logical block 5 应映射到 kernel [170..203]。

## 根因
PR 正文披露:`_logical_to_remote_kernel_block_ids` 在把 FA 逻辑块展开为远端 kernel 块时,使用了**本地**的 `physical_per_logical` arange 而不是**远端**的,导致 kernel 块索引错误,造成异构 TP + Mamba 混合模型的静默精度损坏。此外,`_read_blocks` 中原有的部分前缀命中裁剪只考虑非 Mamba 场景(end-trim),而异构 TP 下逻辑块取整会让两侧 kernel 块数不一致(如 640 prompt tokens:远端 1 逻辑块×10=10 个 kernel 块,本地 2×6=12 个),需要按实际数据块数对齐。

## 修复方案
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` 三处改动:

1. `_logical_to_remote_kernel_block_ids`:FA 组展开改用**远端** `remote_physical_per_logical`(经 `BlockTable.map_to_kernel_blocks`),每个逻辑块 L 映射为远端 kernel 块 `[L*remote .. L*remote + remote-1]`;Mamba 组保持 1:1 直通。

```diff
-        local_arange = np.arange(local_ratio).reshape(1, -1)
+        remote_arange = np.arange(remote_physical_per_logical).reshape(1, -1)
```

2. 新增 `_apply_prefix_caching`,统一处理块裁剪:非 Mamba 模型保持原 end-trim(跳过已缓存前缀);Mamba 混合模型对 FA 组 front-trim 到 `min(local, remote)` 块数(用 `max(两侧 physical_per_logical)` 断言差异合理),SSM 组断言 1:1;`_read_blocks` 改为调用它。
3. `_validate_remote_agent_handshake`:Mamba 混合模型 + 两侧 `physical_blocks_per_logical` 不一致 + 开启 prefix caching 时直接抛 `RuntimeError`,提示用 `--no-enable-prefix-caching`(因 prefix caching 在逻辑块粒度工作,失配时前缀/后缀配对会静默错位,见新增的失败演示测试)。

测试:扩充 `tests/v1/kv_connector/unit/test_nixl_connector_hma.py`,新增远端展开、块数对齐、前缀缓存失配演示等用例(diff 注明 Qwen3.5-0.8B 数值经 verify_conv_split.py 对照 issue #13 验证)。

## 影响与回归风险
- 影响窗口:正文未给出具体版本/提交范围;受影响的是使用 NIXL KV connector + HMA、Mamba 混合模型或异构 TP 的部署。
- 回归面:改动限于 NIXL connector worker 的块 ID 映射与裁剪路径——远端展开索引、前缀缓存裁剪逻辑重排(Mamba 混合从隐式 end-trim 改为 front-trim 到最小值)、新增握手强校验(此前可静默跑错的配置现在会启动失败并给出明确提示)。同构 TP 与非 Mamba 模型的行为按测试用例保持不变。

## 关联
正文未列出

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
