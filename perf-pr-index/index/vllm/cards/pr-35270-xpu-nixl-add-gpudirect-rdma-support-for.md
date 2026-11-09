# PR #35270: [XPU][NIXL] Add GPUDirect RDMA support for XPU

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @zhenwei-intel |
| 合并 | 2026-03-03 |
| 分类 | Overlap / Scheduler(major) |
| Label | ready, ci/build, kv-connector |
| Diff | pack 未提供 diffstat(+/- 数与文件数缺失);diff 内容可用(3 个文件) |
| 链接 | https://github.com/vllm-project/vllm/pull/35270 |

## 一句话结论

NIXL connector 为 XPU 启用 GPUDirect RDMA(KV buffer 走 VRAM),PD 分离下 2P1D 相对 Non-PD 吞吐提升 1.65×。

## 动机

此前 NIXL connector 的 XPU 平台只允许 `cpu` 一种 KV buffer device,KV 传输落在 host 内存,PD(disaggregated prefill/decode)场景下跨节点 KV 迁移无法利用 RDMA 直接读显存。本 PR 让 XPU 与 CUDA 一样以 VRAM 类型注册到 NIXL,走 GPUDirect RDMA。

## 优化点

三处改动:(1) `nixl_connector.py` 中 platform→memory-type 映射把 `"xpu"` 从仅 `("cpu",)` 扩为 `("cpu", "xpu")`;kv_buffer_device 为 `cuda` 或 `xpu` 时 NIXL memory type 取 `"VRAM"`(CPU 取 DRAM)。(2) XPU platform 的 `check_and_update_config` 强制设置 `UCX_MEMTYPE_CACHE=n`,规避 UCX 内部 memory type cache 把 GPU 内存误判为 host 内存导致的 invalid memory access(UCX FAQ 建议项)。(3) `Dockerfile.xpu` 改为从源码构建 UCX(固定 commit,`--with-ze=yes`)+ NIXL 0.7.0,并预装 rdma-core/libibverbs 等 RDMA 依赖,替代原先仅源码装 NIXL 的流程。

## 关键改动

```text
vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py: xpu 允许 XPU buffer,映射为 VRAM
vllm/platforms/xpu.py: 强制 UCX_MEMTYPE_CACHE=n
docker/Dockerfile.xpu: 源码构建 UCX(--with-ze)+ NIXL,RDMA 依赖
```

## 性能声明

Llama3.3-70B int4(AWQ)模型 + fp8 KV cache,8xB60,ISL=1500 / OSL=150,在 SLO(TTFT<5s、ITL<100ms)约束下:2P1D PD 分离 request throughput 1.06,Non-PD 为 0.64,即 1.65× improvement。注意这是 PD 分离整体收益对比,PR 未给出 GPUDirect RDMA 相对 host 内存路径的 A/B 数字。

## 验证与回归风险

测试计划为 PD 集成命令(两个 prefill + 一个 decode 实例 + `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` 代理,`UCX_TLS=ib,rc,ze_copy`),PR 未附自动化测试结果。限制:UCX 必须包含 openucx/ucx#11187 的修复;在 openucx/ucx#11180 合并前需手动设置 `UCX_NET_DEVICES` 以获得更好性能;ze-ipc 暂不支持(#11218)。回归风险:`UCX_MEMTYPE_CACHE=n` 是全局环境变量改动,影响 XPU 上所有 UCX 使用方;kv_buffer_device 行为变化仅限显式选择 xpu buffer 的部署。

## 关联

同区域为 NIXL KV connector 与 PD 分离传输;pack 正文未列出具体关联 PR 编号(引用的是上游 UCX issue/PR)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
