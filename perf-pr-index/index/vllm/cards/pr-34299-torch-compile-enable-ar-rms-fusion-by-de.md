# PR #34299: [torch.compile] Enable AR+rms fusion by default available for `-O2`

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @ProExpertProg |
| 合并 | 2026-02-11 |
| 分类 | CUDA Graph / Compiler (tier: major) |
| Label | performance, ready, torch.compile |
| Diff | diffstat 缺失(pack 未提供 +None/-None across None);可见 diff 含 vllm/config/compilation.py 与 vllm/config/vllm.py 两个文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/34299 |

## 一句话结论

把 allreduce+rmsnorm fusion(AR+rms)设为 -O2/-O3 编译档默认开启(TP>1、Hopper+、有 flashinfer),llama3-70B-FP8 TP=2 B200 上收益 5-22%,启动时间仅小幅增加。

## 动机

AR+RMS fusion(把 TP all-reduce 与后续 RMSNorm 融合,引 #24252)与"为第二个 compile range 编译"此前只能手动开;实测它们只边际增加 warm/cold 启动编译时间,却带来 5-22% 收益,值得进入默认优化档。同 PR 顺带把 eliminate_noops 默认置 True 并从各档位字典与 compute_hash 中移除。

## 优化点

vllm/config/vllm.py 新增 enable_allreduce_rms_fusion(cfg) 判定:TP>1、CUDA 平台、device capability >= 90(Hopper+)、已安装 flashinfer。OPTIMIZATION_LEVEL_02 与 OPTIMIZATION_LEVEL_03 的 pass_config.fuse_allreduce_rms 由固定 False 改为该判定函数;O0/O1 档保持关闭。compilation.py 中 eliminate_noops 默认值 None -> True,并从 compute_hash 字段列表与 O0-O3 的 pass_config 字典中删除(不再影响配置哈希)。机制上是纯默认值/配置层改动,融合 pass 本体(AllReduceFusionPass)已在库中,由 pass_manager 在 CUDA 平台注册。

## 关键改动

```text
vllm/config/vllm.py: enable_allreduce_rms_fusion(TP>1 + CUDA + cap>=90 + flashinfer);-O2/-O3 fuse_allreduce_rms 默认启用
vllm/config/compilation.py: eliminate_noops 默认 True,并移出 compute_hash 与各档位字典
```

## 性能声明

Bench:llama3-70B-FP8,TP=2,B200,5-22% improvement(TTFT/TPOT 对比图在截图中,具体数字未以文字给出;workload 参数未注明)。Startup(RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8,TP=4):cold 94.18s -> 108.63s,warm 68.90s -> 76.07s(即开启 fusion 后编译时间增加约 14-10s)。Eval(lm-eval gsm8k,5-shot,limit 100,num_concurrent=50):Llama-3.1-70B-FP8 flexible 0.89/strict 0.86;Llama-3.3-70B-NVFP4 0.91/0.86;Llama-4-Scout-17B-16E-NVFP4 0.90/0.88。qwen3-235B TP=4 B200 一栏标注 TODO 未测。

## 验证与回归风险

Test Plan 覆盖 startup/bench/eval/CI,CI "Looks good"。回归面:-O2/-O3 的 TP>1 CUDA Hopper+ 且装 flashinfer 的部署默认行为改变(多一个 fusion pass 与第二个 compile range,启动编译时间增加约 10-15s);eliminate_noops 默认开启影响所有 compile 用户的图变换(且不再进配置哈希,旧缓存兼容性需注意);TP=1、非 Hopper、无 flashinfer 时不生效。

## 关联

前序:PR #24252(AR+RMS fusion 本体,收益数字来源)。同区域:AllReduceFusionPass(torch.compile pass 体系,与 PR #61448 的 ROCm fusion pass 对应)。
