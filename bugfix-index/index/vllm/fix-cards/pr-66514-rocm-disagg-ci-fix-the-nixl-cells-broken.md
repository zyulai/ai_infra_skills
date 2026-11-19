# PR #66514: [ROCm][Disagg][CI] Fix the NIXL cells broken by UCX 1.23's ROCm DMABUF default

| 字段 | 值 |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 编号 | [#66514](https://github.com/vllm-project/vllm/pull/66514) |
| 作者 | avininjamay8 |
| 合并日期 | 2026-11-18 |
| labels | rocm, ready, ci/build, kv-connector, nixl |
| 热度 | 👍 0 · 💬 5 |
| 分类 | Other |
| bug 类型 | regression |

## 一句话结论
CI 镜像里 UCX 1.23 默认对 ROCm 内存启用 DMABUF 注册,导致 AMD disagg 的 DeepSeek-V3 NIXL 单元(1P1D TP8、1P1D EP8、2P2D EP16,pull 与 push)全部起不来——TP8 启动 OOM、EP8/EP16 报 HIP error 209;通过 `UCX_ROCM_COPY_DMABUF=no` 等一组 CI 环境与脚本修正恢复所有单元,GSM8K 精度回到 0.94 左右。

## 问题现象
修复前(该镜像 UCX 1.23.0),DeepSeek-V3 NIXL 各单元均拿不到分数:1P1D TP8 pull/push 在启动时 out of memory;1P1D EP8 pull/push 与 EP16(3 节点)pull 在启动时 HIP error 209。另有一个隐蔽问题:`VLLM_NIXL_SIDE_CHANNEL_HOST` 默认 localhost,decode 端会与自身握手,导致每次 KV 加载都静默重算。修复后全部单元跑通且无失败传输:TP8 pull 0.9431 / push 0.9454,EP8 pull 0.9469 / push 0.9462,EP16 pull 0.9409 / push 0.9424(GSM8K 5-shot exact_match)。KV 只走 RDMA 并均匀分布到 8 张网卡;TP8 传输时间从每 137 MB 31-38 ms 降到 7 ms。

## 根因
正文披露根因:UCX 1.23 开始默认对 ROCm 内存启用 DMABUF 注册(`UCX_ROCM_COPY_DMABUF`),在该 CI 环境下 TP8 由此耗尽内存,EP8/EP16 触发 HIP error 209。次要问题:`cluster.sh` 原 `UCX_TLS` 含 `tcp`、`UCX_NET_DEVICES` 只有 `ionic_0:1,lo`,KV 传输可能回退到 TCP;`VLLM_NIXL_SIDE_CHANNEL_HOST` 未设置时默认 localhost,decode 与自身握手、每次加载静默重算;EP16 的精度评测在并发 1 时跑超 105 分钟限制;Kimi-K3 2P2D EP16 在 0.75 的显存利用率下 prefill KV cache 算出负值,且没有序列上限时 prefill 启动检查失败(`num_cache_lines >= batch`)。

## 修复方案
四处配置/脚本修改(均为 CI 基础设施,无 vLLM 运行时代码改动):

1. `.buildkite/amd-disagg/cluster.sh`:新增 `export UCX_ROCM_COPY_DMABUF="${UCX_ROCM_COPY_DMABUF:-no}"`;`UCX_TLS` 去掉 `tcp`(改为 `rc_verbs,ud_verbs,self,rocm`);`UCX_NET_DEVICES` 列出全部 ionic 网卡,不再含 `lo`。
2. `vllm_disagg.sh`:`build_server_cmd` 中对 Nixl 连接器导出 `VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-${HOST_IP}}"`(节点自身 IP)。
3. `pipeline-disagg.yaml`:EP8 与 2P2D EP16 精度评测设 `ACCURACY_NUM_CONCURRENT=32`;TP8 保持默认 1。
4. `models.yaml`:Kimi-K3 EP prefill 改 `--gpu-memory-utilization 0.82 --max-num-seqs 128 --max-num-batched-tokens 8192`,decode 改 `0.80 --max-num-seqs 128 --max-num-batched-tokens 512`,使 2P2D EP16 装进内存;gsm8k 0.9636(1319 questions)。

## 影响与回归风险
- 范围:只影响 `.buildkite/amd-disagg/` 的 CI 脚本与流水线,不影响库代码与用户路径;风险主要是 CI 行为变化。
- 环境耦合:修复依赖 UCX 1.23 这一镜像版本的行为;若未来镜像回退/升级 UCX,`UCX_ROCM_COPY_DMABUF=no` 的显式覆盖保持稳定(变量保留外部可覆盖语义)。
- 评测口径变化:EP 单元精度并发从 1 提到 32、TP8 维持 1,不同单元间并发不再完全一致,纵向对比历史分数时需注意。
- 性能余量:EP8 传输保持 93-102 ms 每 144 MB,TP8 提速明显(31-38 ms → 7 ms 每 137 MB)。

## 关联
- Follow-up to #62007(正文声明)。
- 涉及文件:`.buildkite/amd-disagg/{cluster.sh,vllm_disagg.sh,pipeline-disagg.yaml,models.yaml}`。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
