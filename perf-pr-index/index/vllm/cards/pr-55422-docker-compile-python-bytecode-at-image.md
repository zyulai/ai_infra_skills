# PR #55422: [Docker] Compile Python bytecode at image build time

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @matteso1 |
| 合并 | 2026-10-01 |
| 分类 | Model-specific Fast Path(tier: major) |
| Label | ci/build, cpu, intel-gpu, kv-connector, ready, rocm |
| Diff | diffstat 缺失(pack 未提供 +add/-del 与文件数;共 8 行 ENV + 1 空行) |
| 链接 | https://github.com/vllm-project/vllm/pull/55422 |

## 一句话结论

8 个 Dockerfile 设置 `UV_COMPILE_BYTECODE=1`,把启动时的 Python 字节码编译移到镜像构建期,冷启动到首个正确响应中位省 15.36 s。

## 动机

全新容器每次启动都要现场编译数千个 Python 文件:vLLM 0.28 环境实测一次 uncached serving CLI import 触发 5,118 次编译调用(约 2.16 百万行源码,裸 CPython 编译中位 6.3033 s),这部分 CPU 工作在每次容器启动时重复发生;uv 已有现成选项可在镜像构建期完成该工作。

## 优化点

在 CUDA/CPU/ppc64le/rock/rocm/rocm_gfx1250/s390x/xpu 八个 Dockerfile 加 `ENV UV_COMPILE_BYTECODE=1`,由 uv 在安装依赖时编译字节码并保留源码;字节码缓存架构无关(cpython-312 cache tag 与 magic,macOS ARM64 生成的缓存在 Linux x86-64 验证加载成功)。已知 caveat:不承诺可复现镜像 digest(CPython 序列化字节依赖编译顺序,已复现 upstream issue);QEMU-user 下的编译挂起属构建环境问题,本 PR 的 native 验证不覆盖。

## 关键改动

```text
docker/Dockerfile: ENV UV_COMPILE_BYTECODE=1
docker/Dockerfile.cpu / Dockerfile.ppc64le / Dockerfile.rock / Dockerfile.rocm / Dockerfile.rocm_gfx1250 / Dockerfile.s390x / Dockerfile.xpu: 同一 ENV 行
```

## 性能声明

匹配构建对照(CUDA sm_89、Python 3.12.3、Torch 2.13.0+cu130、uv 0.12.10;RTX 4090 Laptop、4-CPU 配额,Qwen3-0.6B 本地离线,loopback registry 拉取到首个正确 HTTP 响应中位):control(gzip)155.7265 s vs bytecode 144.5137 s,中位配对节省 15.3606 s(n=3,3 对全部改善);bytecode 镜像压缩层 +153,083,004 bytes(+1.7985%);candidate import 编译 0 个源文件 vs control 5,257 个。resident-page 归因实验(vLLM 0.28.0):首个正确响应中位 57.0394 → 47.6911 s,省 10.1300 s(n=3)。import 根(n=3 配对,非可加):`import vllm` 6.7075 → 2.1068 s(省 4.6007 s);`import vllm.entrypoints.cli.serve` 12.2659 → 5.0198 s(省 7.2461 s)。

## 验证与回归风险

CI 91034(H100 fault-tolerance 通过;structured-output 失败与 main 同因;AMD 13576 超时与 wheel 构建顺序相关);每次激活的 5 项 next-token 检查与 4 请求 burst 输出一致。风险:镜像体积 +1.8%;损坏的 marshalled body 可能在无 source 回退时失败、timestamp cache 对同 size/mtime 源的校验局限(Python 文档已述);uv 可能只编译重装的 vLLM 发行版,ENV 行不能证明所有镜像变体缓存全覆盖;仅单主机中位数,未覆盖慢链路/多平台/其他加速器镜像。

## 关联

Related to #48193;同批 #51360(TP1 snapshot,其 H200 验收即使用本 PR 的 bytecode CI 镜像)。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
