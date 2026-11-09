# PR #29420: [AMD][DSV4] Remove per-batch D2H syncs in MTP to avoid bubbles between 2 batches

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @amd-danli103 |
| 合并 | 2026-06-30 |
| 分类 | Speculative Decoding(tier: major) |
| Label | amd, deepseek, run-ci |
| Diff | diffstat 缺失(素材未统计行数);diff 只含 deepseek_v4_backend_hip_radix.py 一个文件 |
| 链接 | https://github.com/sgl-project/sglang/pull/29420 |

## 一句话结论

消除 HIP 上 DSV4 MTP target-verify 元数据构建中每 batch 的 `seq_lens.tolist()` D2H 同步,decode 步时省 8%-13%。

## 动机

DSV4 在 HIP 后端开启 MTP(EAGLE)时,连续 batch 之间出现周期性 2-xx ms 空隙:调度与执行无法重叠。profiling 定位到 target-verify 元数据构建里存在每 batch 的 device->host 同步,阻塞 host 发射线程,使 batch N 的执行与 batch N+1 的调度串行化。原 PR 还计划改 `_attach_unified_kv_prefill_meta` 中的 `torch.repeat_interleave` D2H 回读,后为避免与 #29202 重叠而移出;最终本 PR 只保留 `init_forward_metadata_target_verify` 的改动(正文明示:收益来自两处修改,其中 Modification-1 是主因)。

## 优化点

`init_forward_metadata_target_verify` 原来每个 batch 都对 GPU 张量执行 `seq_lens.tolist()`,强制 D2H 同步。改为新增可选参数 `seq_lens_cpu`:由调用方把已维护好的 CPU 镜像 `seq_lens_cpu` 穿透到 eager 与 cuda-graph-replay(`init_forward_metadata_out_graph`)两个入口;仅当传入 None 时才回退到 `.tolist()`。CPU 值与 GPU 张量一致,行为保持不变;ROCm 专用(CUDA 后端本就避开了这两处同步)。graph replay 路径中直接传 `seq_lens_cpu.tolist()`,注释说明此处无 D2H。

## 关键改动

```text
python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py: init_forward_metadata_target_verify 增加 seq_lens_cpu 参数;eager 与 out_graph 两个调用点传入 CPU 镜像
```

```python
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
-        seq_lens_cpu = seq_lens.tolist()
+        if seq_lens_cpu is None:
+            seq_lens_cpu = seq_lens.tolist()
         return self.init_forward_metadata_target_verify_old(...)
```

## 性能声明

MI355X x8,TP8 + DP8 Attn,DeepSeek-V4-Pro,EAGLE(steps3 / draft4 / topk1),客户代表性 decode-heavy workload(约 50k 输入 + MTP,输出 32 token,3 次重复)。decode 步时/TPOT(越低越好;吞吐由步时推导 1/(1-Δ),未单独实测):

| Batch(local/DP) | Input | BEFORE(ms) | AFTER(ms) | 节省 | 加速 |
| --- | --- | --- | --- | --- | --- |
| 16 | 51.2k | 52.96 | 46.44 | -6.5 ms | -12.3%(约 +14% tput) |
| 16 | 54.1k | 53.29 | 46.31 | -7.0 ms | -13.1%(约 +15% tput) |
| 32 | 51.2k | 68.90 | 62.95 | -6.0 ms | -8.6%(约 +9.4% tput) |
| 32 | 54.1k | 68.30 | 62.77 | -5.5 ms | -8.1%(约 +8.8% tput) |

trace 显示移除 MTP 中的 D2H 同步后 batch 间空泡消失。注意:正文明示上表收益来自 Modification-1 与 Modification-2 两处,而 Modification-1(`repeat_interleave` 的 output_size 预先给值)已移至 #29202,本 diff 仅含 Modification-2。

## 验证与回归风险

精度:GSM8K 0.947,与修改前 0.945 相当。无新增单测(Checklist 各项均未勾选);CI 状态显示 PR Test (Base) pending、Extra 失败(素材所示)。影响面限于 HIP 的 DSV4 radix attention backend 的 target-verify 与 prefill cuda-graph 路径;`seq_lens_cpu` 参数为可选、默认 None 时行为不变,对 CUDA 后端无影响。风险在于传入的 CPU 镜像与 GPU 张量必须保持一致,依赖调用方正确维护 `seq_lens_cpu`。

## 关联

关联:#29202(承接本 PR 移出的 `_attach_unified_kv_prefill_meta` repeat_interleave D2H 同步修复,为收益主因)。同分类其他 PR 编号素材未注明。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
