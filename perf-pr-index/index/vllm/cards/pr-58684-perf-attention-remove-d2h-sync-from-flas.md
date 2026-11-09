# PR #58684: [Perf][Attention] Remove D2H sync from FlashInfer SM90 sparse MLA plan under async scheduling

| | |
| --- | --- |
| 仓库 | vllm-project/vllm |
| 作者 | @NickLucche |
| 合并 | 2026-09-27 |
| 分类 | Speculative Decoding (tier: major) |
| Label | nvidia |
| Diff | diffstat 缺失(pack 未记录 additions/deletions/files);diff 含 `flashinfer_mla_sparse_sm90.py` 及其测试文件 |
| 链接 | https://github.com/vllm-project/vllm/pull/58684 |

## 一句话结论

去掉 FlashInfer SM90 sparse MLA 每次 metadata build 的阻塞 D2H 同步,GLM-5.3-Flash async TPOT c=1 从 11.09 降到 6.86 ms(-38%)。

## 动机

async scheduling 下 `FLASHINFER_MLA_SPARSE_SM90`(Hopper,NoPE 模型如 GLM-5.3-Flash 的默认 sparse 后端)在每次 metadata build 都做阻塞 D2H 拷贝:FlashInfer 在 host 上把每行 kv_len 烘进 plan,而 async spec decode 下 `seq_lens_cpu_upper_bound` 是乐观值,所以 builder 不得不把 `positions`/`seq_lens` 拷回 CPU 取精确长度;开 MTP 后每个 draft step 都发生一次。后果是 async 调度反而比 `--no-async-scheduling` 慢(GLM-5.3-Flash,c=1:11.1 vs 7.7 ms TPOT)。torch profiler 显示 draft-step metadata build 中每约 5 步 13 次阻塞同步、共约 79 ms,位于 `_kv_lens_host`。

## 优化点

三步方案。其一,CPU 上用 host 上界加余量做 plan:`seq_lens_cpu_upper_bound` pad 32 token(`_PLAN_SLACK`)并 clamp 到 `max_valid`(index_topk + index_kpool - 1),全程不碰 GPU。其二,设备端 clamp:新 Triton kernel `_clamp_work_kv_end` 直接读 FlashInfer 的 int workspace 中 plan 出的 work item 列表(MAPlanInfo 布局常量解析,长度断言校验),用与 convert kernel 相同公式从 `seq_lens`/`query_start_loc`/`req_id_per_token` 算每行精确 valid count,把每个 work item 的 `kv_end` clamp 到它;FA3 kernel 不会越过 `kv_end` 读取,余量和 index buffer 的 -1 尾部永远不会被碰。其三,plan 复用:context 每步只增长几个 token,slack 足以吸收,新上界仍能塞进旧 plan 时跳过 `wrapper.plan()` 只跑 clamp——这也是 no-async 也变快的原因。自适应验证(host query split 与 device 不一致)时上界退化为 batch 最大值;无 host 上界时仍走精确(同步)路径。代码内 TODO 标注:待 FlashInfer 支持上界 plan 或 GPU 侧 plan API(上游 issue flashinfer#5559)后可删除。

## 关键改动

```text
vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:
  _clamp_work_kv_end Triton kernel、_SM90State.plan 上界+复用+clamp、
  _kv_lens_host 返回 (rows, lens, exact)、_PLAN_INFO 布局常量
tests/v1/attention/test_flashinfer_mla_sparse_sm90.py: plan() 新签名,
  新增上界 plan 对比精确 plan、plan 跨步复用两个 GPU 测试
```

关键片段(clamp kernel 核心):

```python
ctx = seq_len - query_end + row + 1
valid = tl.where(ctx <= TOPK, ctx, TOPK + ctx % KPOOL)
end = tl.load(saved_kv_end + works, mask=mask, other=0)
tl.store(int_ws + kv_end_off + works,
         tl.maximum(tl.minimum(end, valid), start), mask=mask)
```

## 性能声明

E2E:zai-org/GLM-5.3-Flash,H100 TP8,fp8 KV cache,MTP 5 speculative tokens,max-model-len 32768,`vllm bench serve` random 1024/128 ignore-eos,各点至少 32 prompts、跑两轮取均值。Mean TTFT / median TTFT / mean TPOT (ms):c=1 主线 async 136.6/138.8/11.09 → 本 PR async 128.1/126.9/6.86;c=8 21.87 → 17.36;c=32 45.04 → 37.04。async 下 TPOT 降 38%/21%/18%(c=1/8/32),且 async 在所有并发下都反超 no-async(如 c=1 no-async 7.47 ms)。基准为随机 token,几乎无 draft acceptance,TPOT 接近投机解码最坏情形。精度:GSM8K(1319 题 5-shot)0.913 → 0.921(async)、0.911 → 0.917(no-async),MTP mean acceptance length 约 4.2 不变。所有 e2e(含基线)都本地叠加了 #55222 的 fp8 plan-dtype 修复。

## 验证与回归风险

单元测试 18 passed(H100,含两个对真实 FlashInfer 的 GPU 测试:上界 plan + clamp 与精确 plan 输出一致、plan 随 context 增长复用且不重 plan);torch profiler 确认 draft-step metadata build 无阻塞同步残留。风险:直接读取 FlashInfer 内部 `_plan_info` 与 int workspace 的 MLAPlanInfo 布局(有断言保护),FlashInfer 升级改布局会破坏;仅影响 `FLASHINFER_MLA_SPARSE_SM90`;与 #58260(>16k 行 plan/run 分块)都动 `plan()`/`build()`,后落地者需 rebase。

## 关联

相关:#55222(fp8 plan dtype 修复,本 PR 依赖)、#58260(plan/run 分块)。运行时去同步化方向同 #23569(全重叠模型执行);同分类投机解码:#63801、#22872。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
