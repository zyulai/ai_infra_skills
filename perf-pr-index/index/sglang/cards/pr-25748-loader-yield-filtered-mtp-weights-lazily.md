# PR #25748: loader: yield filtered MTP weights lazily to avoid OOM hang on multi-layer EAGLE

| | |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 作者 | @JustinTong0323 |
| 合并 | 2026-05-20 |
| 分类 | Speculative Decoding |
| Label | run-ci |
| Diff | diffstat 缺失(pack 未提供 + / - 数字);单文件 loader.py,约 +8/-6 行 |
| 链接 | https://github.com/sgl-project/sglang/pull/25748 |

## 一句话结论

把 MTP 权重过滤改为惰性 generator,修复多层 EAGLE 加载 ~963 GiB fp8 checkpoint 时 CPU OOM 导致的启动挂死,就绪时间从无限挂起降至约 2.5 分钟。

## 动机

`DefaultModelLoader._filter_mtp_weights` 在返回前把整个过滤后的 checkpoint 物化成 `tuple`,使上游 `buffered_multi_thread_safetensors_weights_iterator` 的滑动窗口(`buffer_size = max_workers + 1`)失去内存上界。多层 EAGLE 下大 MoE checkpoint(如 MiMo-V2.5-Pro,约 963 GiB fp8)每个 TP rank 会实例化 `speculative_num_steps` 个 draft runner(各有 `draft_model_idx`),每个 runner 独立重扫全量 checkpoint;eager tuple 使每个 runner 都把整份前缀重命名的张量驻留 CPU——TP=8 加 3 个 draft runner 时主机进入 page reclaim。表现为:scheduler 线程全部卡在 `futex_do_wait`、`/proc/<pid>/io read_bytes = 0` 无读增长、日志停在 "Found local HF snapshot ... skipping download." 之后再无进展;virtiofs 与本地 NVMe 现象相同(排除存储因素)。

## 优化点

改动极小但击中要害:`_filter_mtp_weights` 的返回类型从 `Tuple[Tuple[str, torch.Tensor], ...]` 改为 `Generator[Tuple[str, torch.Tensor], None, None]`,逐条 `yield (prefix + new_name, tensor)` 替代 append 后 `return tuple(filtered_weights)`。调用方本来就是迭代使用,兄弟路径 `_get_weights_iterator` 也已返回 generator,行为等价;改为惰性后,上游有界缓冲滑动窗口重新生效,CPU 内存被真正限制在 window 大小,多层 EAGLE 的每个 draft runner 不再各自物化全量过滤结果。

## 关键改动

```text
python/sglang/srt/model_loader/loader.py: _filter_mtp_weights 从 eager tuple 改为惰性 yield
```

```python
def _filter_mtp_weights(cls, weights_iterator, prefix, draft_model_idx
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
    for name, tensor in weights_iterator:
        match = cls._MTP_PATTERN.match(name)
        if match is not None:
            ...
        else:
            new_name = name
        yield (prefix + new_name, tensor)
```

## 性能声明

8×B200,MiMo-V2.5-Pro(~963 GiB fp8),TP=8,`enable_multithread_load=True`(默认),EAGLE multi-layer(num_steps=3、topk=1、draft_tokens=4),flashinfer_trtllm + fa4:

- 修改前:server ready 无限挂起(全部 scheduler 卡在 futex_do_wait);修改后:约 2.5 min 就绪。
- 修改后质量指标(此前因挂起无法测):GSM8K(200 条,parallel=4)score 0.97;avg accept_rate 0.755(307 batches);avg accept_length 3.27(max 4);gen throughput ~700 tok/s。
- 与旧的 `enable_multithread_load=false` workaround 对比:accept_rate 0.753→0.755、GSM8K 0.975→0.97(噪声内),server 就绪快约 **30%**,原因是多线程 loader 真正生效。

## 验证与回归风险

验证依赖上述手工测试与 CI(PR Test 通过)。改动为标准 Python 语义替换,行为等价性依赖"调用方只迭代一次"这一前提——若有调用方需要重复迭代或取长度则破坏;筛选/重命名逻辑本身未变,MTP 权重匹配(`_MTP_PATTERN`)与 `model.mtp.layers.0.` 前缀映射保持不变。

## 关联

同区域:多层 EAGLE / MTP 加载相关 PR #3582、#6606、#24663;与 #25455(MiMo-V2 flash 适配)同属 MiMo 生态。

---
> 本卡由 PR 正文与 diff 自动素材 + LLM 撰写,性能数字未独立核实。
