# PR #26831: Fix multi-tokenizer batch request output routing (health stuck at 503)

| 字段 | 值 |
| --- | --- |
| 仓库 | sgl-project/sglang |
| 编号 | [#26831](https://github.com/sgl-project/sglang/pull/26831) |
| 作者 | whybeyoung |
| 合并日期 | 2026-05-31 |
| labels | (none) |
| 热度 | 👍 1 · 💬 2 |
| 分类 | Frontend / Serving |
| bug 类型 | hang |

## 一句话结论
multi-http-worker 模式下 batch 请求先建子对象再绑定 http_worker_ipc,导致子对象缓存了 None、输出被 router 丢弃,PD prefill warmup 收不到响应,/health 永远 503;调整两步调用顺序修复。

## 问题现象
在 multi-http-worker 模式(`--tokenizer-worker-num > 1`)下,发送 batch 请求后:tokenize 后的 batch item 携带 `http_worker_ipc=None`,scheduler 构建的每个 Req 也是 None,流式返回的 `BatchStrOutput` 中 `http_worker_ipcs=[None, ...]`;`MultiTokenizerRouter` 把这些输出全部丢弃(日志「IPC name is None ... skipping」),于是 PD prefill warmup 的 POST 永远等不到响应,`server_status` 一直停在 Starting,`/health` 永远返回 503。单 tokenizer 模式与非 batch 请求不受影响。

## 根因
正文披露:`generate_request` 里先调 `_init_req_state(obj)` 再调 `_attach_multi_http_worker_info(obj)`。对 batch 请求,`_init_req_state` 会访问 `obj[i]`,而 `GenerateReqInput.__getitem__` 会构建并缓存(CACHE 到 `_sub_obj_cache`)按下标拆出的子对象,子对象复制的是当时父对象的 `http_worker_ipc`;由于 attach 在其后才执行,父对象的 `http_worker_ipc` 尚为 None,所有被缓存的子对象都固化了 `http_worker_ipc=None`,随后 attach 只给父对象打了标记,子对象不再更新。

## 修复方案
只调整 `python/sglang/srt/managers/tokenizer_manager.py` `generate_request` 中的两步顺序:把 `_attach_multi_http_worker_info(obj)` 移到 `_init_req_state(obj, request)` 之前,保证父对象 `http_worker_ipc` 在子对象构建并缓存前就已设置:

```diff
+        if self.server_args.tokenizer_worker_num > 1:
+            self._attach_multi_http_worker_info(obj)
         self._init_req_state(obj, request)
         if self.server_args.language_only:
             self._handle_epd_disaggregation_encode_request(obj)
-        if self.server_args.tokenizer_worker_num > 1:
-            self._attach_multi_http_worker_info(obj)
```

## 影响与回归风险
- 影响窗口:正文未说明具体版本/提交范围;从机制看,凡引入 `_sub_obj_cache` 缓存语义与该调用顺序组合的版本均受影响。
- 回归面:改动仅为一条语句的位置交换,只作用于 `tokenizer_worker_num > 1` 时 `generate_request` 的初始化顺序;attach 先行不依赖 `_init_req_state` 的产物(按 diff 所示两段代码互不引用,由 diff 推断),单 tokenizer 模式与非 batch 请求路径无行为变化。若 `_attach_multi_http_worker_info` 未来开始依赖 `_init_req_state` 初始化的字段,该顺序需重新审视。

## 关联
正文仅 CC @hnyls2002,未列出 issue/PR 链接。

---
> 修复卡由 LLM 基于 PR 正文与 diff 生成,未经人工核实;以原帖与实际代码为准。
