# PR #24515: LPLB: linear-programming load balancer for MoE expert parallelism

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @feliang-git |
| 合并 | 2026-06-16 |
| 分类 | MoE(major) |
| Label | dependencies, deepseek, run-ci, jit-kernel, release-highlight |
| Diff | pack 未提供 diffstat(diff 正文可用,另有 13 个文件因过长未收录) |
| 链接 | https://github.com/sgl-project/sglang/pull/24515 |

## 一句话结论

新增按层解 LP 的 MoE 专家分发算法 LPLB,路由倾斜的 GSM8K 负载吞吐 +5.40%,ShareGPT TTFT p50 -9.2%。

## 动机

EP 下冗余专家副本(`--ep-num-redundant-experts`)的 token 分发此前只有 `static`/`dynamic` 等启发式选择,路由倾斜负载(GSM8K 类)会让部分副本过载。LPLB 把每层的 token 分发建模为线性规划,在副本间均衡路由。默认行为不变,opt-in 经 `--ep-dispatch-algorithm=lp`(与 static/dynamic/fake 并列)。

## 优化点

每个 MoE 层求解一次 LP,输出逻辑专家到物理副本的 dispatch 概率与映射,再由分发 kernel 按逆 CDF 采样选副本。计算栈为 4 个 JIT 编译的 CUDA kernel(`python/sglang/jit_kernel/csrc/lplb/`):`ipm.cuh` 是单 block 内点法(IPM)LP 求解器,用 cuBLASDx 做 GEMM、手写 block Cholesky 做 POSV,从而不需要 `-rdc=true` 设备链接;`lp_prep`/`lp_post`/`dispatch_probability` 分别把 8/5/7 个 torch op 融合成单次 launch(如 dispatch_probability 把 gather、行和判零回退均匀分布、cumsum 逆 CDF 采样、log2phy 映射折叠进一个 kernel,并附逐算法一致的 torch 参照实现供测试)。依赖仅新增 Math-DX(`nvidia-mathdx` PyPI 包 / `MATHDX_HOME` / 随附下载脚本)。失败语义严格:无 CUDA、SM < 9.0 或缺 Math-DX 时 server 启动即报错,不静默回退到 torch IPM 或 dynamic 分发。默认(不带 LP flag)代码路径为惰性 import,与上游逐字节一致。附带一个无关的 multi-node worker 节点 `tokenizer_manager is None` 启动守卫修复。

## 关键改动

```text
python/sglang/jit_kernel/csrc/lplb/ipm.cuh: 单 block IPM LP 求解器(cuBLASDx + block Cholesky POSV)
python/sglang/jit_kernel/csrc/lplb/{lp_prep,lp_post,dispatch_probability}.cuh: torch op 融合 kernel
python/sglang/jit_kernel/lplb/cublasdx_solver.py + cuda_solver.py: JIT 包装与 LPLBSolver
(其余 13 个文件未在 pack diff 中:server_args/ep-dispatch 接入、eplb 测试等)
```

## 性能声明

测试条件:2x H20 节点,DeepSeek-R1,EP=16 + DP-attention,DeepEP backend(normal),`--max-concurrency 2048`,4096 prompt 计时运行 x 3 reps,顺序 counterbalanced,每次 rep 之间 kill-and-relaunch 以控制 JIT cache 与 NIC 发热漂移;LP 侧需 `--init-expert-location` 提供 expert_stats.pt(dynamic 模式 + expert distribution recorder 的 1000-prompt 统计)。结果(dynamic → lp):GSM8K 吞吐 19,715 → 20,779(+5.40%),TTFT p50 -5.99%;MMLU 吞吐 22,949 → 22,825(-0.54%),TTFT p50 -2.58%;ShareGPT 吞吐 2,114 → 2,104(-0.48%),TTFT p50 -9.18%。PR 说明:GSM8K 是 LPLB 针对的路由倾斜负载;MMLU/ShareGPT 吞吐为零效应(不回归);ShareGPT 的 -9.2% 是中位用户 TTFT 改善;所有指标 n=3、p>0.05,但 GSM8K 上 5/5 方向一致。

## 验证与回归风险

多 rank smoke(2x H20:server 启动、`/server_info` 报告 lp、单/8 并发补全正常);3 数据集 x 6 reps(18 次计时运行)无错误;Codex review P0 清零;分布式测试 `test/registered/eplb/test_lplb_distributed.py`(torchrun 2 进程)在 PR 合并时 CI 尚未触发(checklist 未勾选)。风险面:范围严格限定 DeepSeek-v2 前向——其他 MoE 路径(Qwen3、GLM4、SDAR-MoE 等)不参与 EP all-reduce,LP 开启时会死锁,故架构检查在 `LPLBSolver.__init__` 拦截;LP 还要求 `--enable-dp-attention` + `--moe-a2a-backend=deepep` + Hopper(SM>=9.0),由 `check_lplb_server_args` 启动校验;JIT 路径在 DP-attention all-reduce 下需 `NVSHMEM_*` 环境变量避开 legacy 默认流。

## 关联

同区域:EP 冗余专家与 EPLB 系列(dispatch algorithm 的 static/dynamic 选项为基线);依赖 DeepEP a2a backend。pack 未列出其他具体 PR 编号。
