# Kimi K3 本地来源验证记录

验证日期：2026-11-05 至 2026-11-06。当前用户指定来源：`/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3`。此前来源为 `/mnt/tidal-nj01/dataset/youtai/models/Kimi-K3`；下文保留其历史验收，本轮新增证据见文末。

本文记录实际源码与 CPU 验证，不能代替完整权重加载、模型生成或 GPU optimizer 验收。原模型目录保持只读；动态模块及 tokenizer 缓存使用测试临时目录。实施边界见 [CONTRACT.md](CONTRACT.md)。

## 模型与运行环境

- 架构：`KimiK3ForConditionalGeneration`，`model_type=kimi_k3`，文本配置为 KimiLinear。
- 文本配置：93 层、hidden size 7168、896 个 routed experts、top-16、2 个 shared experts、词表 163840。
- tokenizer 与 processor 使用该目录的 Python XTML encoder；SGLang reasoning/tool parser 注册名均为 `kimi_k3`。
- 本机测试解释器：`/opt/sglang/bin/python`；SGLang 0.5.21、Transformers 5.16.0、PyTorch 2.13.0+cu130。GPU 运行兼容性尚未验收。
- 本地 SGLang checkout：`e00930c5489053f26d86b179cee0d087f846acbb`；实际源码文件指纹也需随运行环境记录，版本号本身不足以证明 parser 相同。
- 模型 index 指向 96 个存在的分片，声明总大小 `1,560,860,324,864` 字节。只读分片头确认专家参数采用 `weight_packed` / `weight_scale` 的 U8 压缩存储；没有读取全量张量或计算全量权重哈希。
- 现有 `/root/Megatron-LM/megatron/bridge/models/kimi/kimi_k3_bridge.py` 已有 MXFP4 解包/反量化及 BF16 专家参数映射，Relax 的 `KimiK3VLBridge` 继承该路径。不因量化存储就要求另一个 BF16 目录；实际分布式导入仍未验收。CPU bridge 测试通过 `PYTHONPATH=/root/Megatron-LM` 使用现有源码，没有安装新依赖。

## 已发现问题与修复边界

1. 原通用图片加载会提前按非 K3 规则缩放。K3 编译路径改为保留原始图像尺寸和模式，由原生 processor 处理；不支持的显式通用 resize 覆盖会报错。
2. K3 工具结果需要上一 assistant 的工具 ID、名称和顺序。编译器现支持显式历史上下文，使用原生规则同步重排消息与图片；若会重排已提交前缀，则拒绝增量追加。经用户明确批准，Service 已按 compiler 能力开关传入已提交历史；非 K3 编译器保持原调用约定。
3. 真实模板的非 thinking 最终回答可在现有 parser 路径泄漏 `<|end_of_msg|>`；截断的非 thinking 回答可被误判为 reasoning。Service 现从响应所属 subtree 读取实际 thinking 设置，并只在解析视图去掉尾部结构终止 token；原始生成 token、logprob、usage 和调用 ID 的来源保持不变。相关回归已通过。

## 旧模型来源的历史验证证据

- 修复前合并回归：**247 passed / 5 failed / 0 skipped**（45.90 秒）。5 项失败复现终止标记泄漏及非 thinking 截断回答的 reasoning 误分类。
- 获批 Service 补丁应用后的合并回归：**252 passed / 0 failed / 0 skipped**（47.01 秒）；原有 5 项失败全部修复，没有 skip/xfail。
- 补齐原生 Service 工具历史接线回归和阶段 3 后的最终合并回归：**293 passed / 0 failed / 0 skipped**（47.00 秒）。
- 阶段 3 独立验收：**40 passed / 0 failed / 0 skipped**（15.39 秒），其中 31 项上下文/选择/完整性检查、9 项原生提交与显式导出检查。
- `tests/cpu/models/test_kimi_k3_hf.py`：4 passed、0 skipped（42.77 秒），覆盖真实 tokenizer/processor/worker、受限规模 CPU 视觉反向传播、配置到视觉 bridge 的映射、KDA 参数的 FP32 保留。视觉测试缩小架构并使用测试语言模块；没有加载完整 checkpoint 权重。
- 真实 parser 文件现为 **10 passed / 0 failed**；新增用例经过原生 `_terminal_response_locked`，要求从实际 subtree 取得 thinking 设置，检查字面终止标记保留，并要求 Forest 中原始 token、logprob 与 usage 不被解析处理改写。

重跑基础环境：

```bash
export CUDA_VISIBLE_DEVICES=''
export KIMI_K3_SOURCE_DIR=/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/Megatron-LM:/mnt/tidal-nj01/dataset/youtai/Kimik3_harness/Relax
/opt/sglang/bin/python -m pytest \
  tests/cpu/agentic/test_kimi_k3_compiler.py \
  tests/cpu/agentic/test_kimi_k3_parser.py \
  tests/cpu/utils/data/test_kimi_k3.py \
  tests/cpu/models/test_kimi_k3_hf.py -q -rs
```

合并回归另包含 `tests/cpu/agentic/test_protocol_consistency.py`、`tests/cpu/agentic/test_protocol_errors.py`、`tests/cpu/test_agentic_rollout.py` 和 `tests/cpu/examples/agentspark_agentic`。上述回归使用真实 checkpoint 模板生成的受控续写和实际解析器，不是完整模型权重的生成验收。

完整 Service 修复已在用户明确批准后应用并执行，通过上述 CPU 回归。修改只涉及历史上下文调用点、K3 解析视图的终止 token 和有效 thinking 设置传递；没有新增依赖或修改服务生命周期。阶段 3 的本地 Forest、媒体与原生显式导出验收已通过。完整 V21 reward/token credit、真实 AgentSpark、GPU 权重加载/更新仍未验收，`training_ready=false`。

## 阶段 3 原生接线与导出证据

- 新增 Service 历史回归经真实 `_append_observation`、`_terminal_response_locked` 与 Forest，连续提交仅携带 opaque `tool_call_id` 的图像结果；未在测试中手传 compiler history。验证工具名称/序号解析、RGBA 保留、每图一个 backend slot 和训练侧 4/6 个媒体 token 展开。即使工具参数从 `1e2` 规范为 `100`，原始生成 token/logprob 仍保留。
- 原生导出测试使用真实 K3 tokenizer/processor，以及固定模板生成的受控 backend tokens/logprobs。执行 `_normalized_chat_request → _chat → _append_observation → _apply_generate_result → _terminal_response_locked → SessionOutput.from_records → _build_session_export_transport`，不绕过提交过程拼装假 Forest。
- 顺序追加、重置和分支产生 5 个明确选择的物理导出，均保留同一个逻辑 `Sample.index` / `group_index`。训练 token、response spans、loss mask 和 logprob 来自实际 Forest；输出 metadata 不替代数据集 `train_metadata`。
- 图片新增、删除、重复、换位和分支隔离均检查保留的 PNG 尺寸/像素、processor grid 与 pixel values、backend slots 和训练 token 展开。另检查 JSONL 记录解析，以及未提交 messages/tools/template、无生成响应的状态、错误 defaults/调用身份和伪 native 对象均被拒绝。
- 精确覆盖同 Group、同规范上下文、不同 Session 的采样混用：A 审计不能验证或导出 B 的 token；选择说明中出现 Session credential 也会拒绝。

这些测试证明本地协议、编码、提交和导出行为，不能证明真实 AgentSpark、远端图片 URL 内容不可变、完整 K3 权重生成或 V21 训练效果。离线分析和原生验证不会发布 `RELAX_OUTPUT_JSON`；`training_ready` 仍为 `false`。

最终完整 CPU 命令：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/root/Megatron-LM:/mnt/tidal-nj01/dataset/youtai/Kimik3_harness/Relax \
KIMI_K3_SOURCE_DIR=/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3 \
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/sglang/bin/python -m pytest \
  tests/cpu/agentic/test_kimi_k3_compiler.py \
  tests/cpu/agentic/test_kimi_k3_parser.py \
  tests/cpu/utils/data/test_kimi_k3.py \
  tests/cpu/models/test_kimi_k3_hf.py \
  tests/cpu/agentic/test_protocol_consistency.py \
  tests/cpu/agentic/test_protocol_errors.py \
  tests/cpu/test_agentic_rollout.py \
  tests/cpu/examples/agentspark_agentic -q --disable-warnings --tb=short
```

本轮静态检查通过：16 个相关 Python 文件的 AST、按仓库配置的 isort、冲突标记与 `git diff --check`。SFT 图片展开提取的机器比较仍通过，允许的差异仅为 `len(images)` 参数化。完整 `pre-commit run --all-files` 再次尝试后仍因本地缺 hook 缓存、离线禁止 HTTPS fetch 而未完成；没有安装依赖，不能将完整 pre-commit 计为通过。

## 旧模型来源文件 SHA256

以下仅覆盖编码、配置和 tokenizer 文件，非完整权重指纹。

```text
49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676  encoding_k3.py
f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944  tokenization_kimi.py
ec9f7e86d2ab0eee07a8e7e7c037046e77ac3c25a710ad1298ec13be3b585b54  kimi_k3_processor.py
d122b30bfd3a51a6f05d4bfcfda1e657827322b1353f7caefeebc2835d7736b5  kimi_k3_vision_processing.py
78403540328f9847d6b7ebc5c44eb2e6a752863de0afb7d0710728bb161dc60d  media_utils.py
735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae  configuration_kimi_k3.py
9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213  config.json
5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e  tokenizer_config.json
4be333605990c53a816e586dee9d5dd545afb7a59947c17f8f7ef26b4782668e  preprocessor_config.json
c6648c25e9705af7fba8847e243840d21b5cc63ddeb6297f750a7ddbb6a02836  generation_config.json
b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103  tiktoken.model
```

## 当前模型来源与数据适配（2026-11-06）

当前模型目录的 14 个编码/配置/词表/index 文件与旧目录比较，只有 `encoding_k3.py` 不同，SHA256 为 `b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333`。新实现会解析并重新序列化工具参数 JSON（例如 `1e2` 变为 `100.0`），并清除纯空白 reasoning。回归按当前来源的原生 token IDs 和数字语义验证，不再把旧模板的字面拼写当作契约；真实生成 token/logprob 仍须原样保留。96 个 index 引用分片可达；没有加载或计算全量权重哈希。

本轮在 `data_utils.py` 修复 K3 长度预检：message list 走原生结构编码和 processor 图片展开，rendered string 沿现有 rollout 编码契约；异常、处理进程失败、无 processor 的图片和非法首条输入不再绕过检查。非 K3 路径保持原行为。转换结果的顶层 `messages` 只承载当前问题；完整 Replay/系统/工具/图片上下文要在实际回调时由 Service 根据训练 token 计数检查，不能拿数据集当前问题的长度证明完整请求一定容纳得下。

数据来源为 `/mnt/tidalfs-wx-idc01/dataset/pai/mmx/runs/formal-v21-40g-onestep-20261022-147edbae/data/prepared.jsonl`。任务配置从该 run 的 `stage/config.json` 及冻结 `think_online` profile 核实：no-Loader、源工具版本、每轮最多 30 张图片、`pinned_compat` 选择及源上下文预算。离线转换调用显式选定的 Miles 输入编译器，运行时不依赖 Miles；源 metadata、rubric 和资源快照保留，编译后的资源别名/上下文和媒体计划另外记录。

新增挂载的历史/快照图片 30,408 个本地路径均完成真实读取、SHA256 和完整解码；当前附件另外 2,208 个路径同样通过，共 32,616 个。此数字覆盖先前 review 的深层图片与当前附件清单；额外资源索引引用的验证单独记录。运行 adapter 的容器/节点必须具备对应绝对路径挂载；运行时重新读取并校验清单，不能仅依赖核验机的可达性。原始字节固定后才通过本次调用的临时图片路由或 data URI 使用。

私有媒体计划按消息角色、图片出现位置、tool ID/name/arguments 哈希绑定原始内容；模型回调恢复实际图片后，保留 raw/forwarded 两份身份。原生集成测试已贯通 `MediaPlan.restore → _chat → committed Forest → native export`，并检查 RGBA/RGB 像素、尺寸、backend 媒体槽与训练 token 展开、图片删除后的分支、原始生成 IDs/logprobs；raw marker 上下文不能冒充带图的已提交状态。工具参数哈希遵循 Miles 源数据协议，与 K3 原生数字规范化的上下文身份各有用途，不能混换。

本轮最终合并 CPU 回归为 **351 passed / 0 failed / 4 skipped**（56.28 秒）。4 个跳过项全部为未提供本地 Qwen3-VL-4B/Qwen3-4B 来源的既有对照测试；K3 和真实 Miles 输入编译器测试均已运行。命令在前述完整 CPU 命令中额外设置 `MILES_SOURCE_DIR=/mnt/tidal-nj01/dataset/youtai/Kimik3_harness/miles`，并增加 `tests/cpu/utils/data/test_check_sample_length_multimodal.py`、`tests/cpu/utils/data/test_data_utils.py`。日志：`/tmp/relax-agentspark-adaptation-20261106/final-cpu-regression.txt`。

资源索引另发现 4,091 个不在原视觉清单中的本地图片路径，已全部读取、SHA256 与完整解码通过；总清单现为 **36,707 个唯一本地路径**。新增根目录包括 `.../dots_rl/yufeng/rl_datas/vlm_datas/role_play_0915`，另有图片位于已授权的 `.../dots_rl/xianqing/runs/.../inputs/images`。最终清单为 `/tmp/relax-agentspark-adaptation-20261106/all-media-manifest-v2.jsonl`，SHA256：`d1e883d009e645a89b97c536231889c4fb09e9c9e8238ffced1f34f958d3605c`。

资源索引里的远端 URL 是 Replay/工具的资源引用，不等于当前模型输入中的图片。转换器按冻结来源的域名列表验证并保留这些引用，没有联网下载或声称它们的内容已验证；显式历史/当前/快照视觉块需要固定本地字节，缺少本地身份的远端视觉块单独拒绝。此区别已通过真实源行及回归检查。

另外使用新模型来源对原始数据全部 **21,288 条当前 query** 做联合编码验证：0 编码异常，长度 89–9,109 tokens，P99 为 257；超过 8,192 tokens 的有 2 条。13 个最短/最长/阈值边界样本在上限等于实际长度时接受、少 1 token 时拒绝。原始数据 2,747,188,907 字节，SHA256 为 `f3feb6c5ff1e32f701871811b9c156074efb2e7c5f6b2ba8b8819796ecb64e8c`，与此前 review 一致。该实验使用原生默认 thinking/effort，不含 tools、系统提示、Replay 或图片，不能当作完整 AgentSpark 上下文长度验收。报告：`/tmp/relax-agentspark-adaptation-20261106/k3-real-data-query-validation.json`。

静态检查：23 个相关 Python 文件 AST、仓库 isort 配置、尾部空白、冲突标记及 `git diff --check` 通过；既有 SFT 图片展开的完整正文机器比较仍通过。完整 pre-commit 已再次尝试，因本地缺 hook 缓存且离线禁止 HTTPS fetch 未完成；没有安装新依赖，也没有将此项标为通过。

完整流式转换结果：**21,288 行输入 → 21,258 行接受、30 行明确拒绝**。退出码 `3` 表示存在拒绝记录，不是全部输入通过。分类为：22 行空历史用户消息、3 行工具参数存在相同值重复键、4 行工具参数存在冲突值重复键、1 行显式历史图片仍为 HTTPS 且无已验证本地副本。没有剩余的本地图片缺失、清单不全或目录授权错误。源输入未改动；拒绝清单保留 source_line、原行哈希、query-ID 哈希和固定原因码，不将缺失内容补成猜测文本，也不采用有冲突的 last-wins 参数。

最终数据产物均为 mode 0600，位于 `/tmp/relax-agentspark-adaptation-20261106/`：`prepared-relax-v2.jsonl`、`rejections-v2.jsonl`、`preparation-report-v2.json`。首轮中断产物已明确改名为 `attempt-1-interrupted.*`，不得当作完整数据。源 profile 指纹为 `9f6bdda84d199311a628653035769e8e6894dcb4b33b238b7dfd35d1d3881fd2`，源 compiler 与本次 converter 合并指纹为 `a656b2d6ccd7c1682a7b62dcfb178ec94dfe9446686bb656d9605f8822f6440b`。完整逐文件指纹在 preparation report 中。

独立逐行复核 **通过，0 错误**：21,258 接受行与 30 拒绝行组成完整且无重复的输入分区；源字段/metadata、原行哈希、编译身份、媒体清单、envelope 均一致，129 条超过 1900 UTF-8 字节的问题完整保留。最终 prepared SHA256 为 `f0773d5083d9f5c2d104571d2b316ebd78f2d360644ab34fb56cef33583d5737`。详见 `/tmp/relax-agentspark-adaptation-20261106/validation-v2.json`。

**本数据集需要显式设置回调体积预算。** 18 行的初始图片 data URI 下限已超过默认 16 MiB；最大 34,013,989 字节（32.438 MiB，源第 16,774 行），因此 32 MiB 也不足。已生成 `/tmp/relax-agentspark-adaptation-20261106/agentspark-config-overrides.json`，包含 `max_callback_bytes: 67108864`（64 MiB）及上述三个明确挂载根目录。它是合并到后续真实 AgentSpark config 的覆盖片段，不是具有端点/模型/认证的可运行配置。64 MiB 是本数据集联调起点，不是完整轨迹的充分上限证明；仍需测量真实文本、工具、历史和后续图片的总请求体。适配器对实际原始/恢复后请求都执行上限检查，不为满足上限缩图或丢图。

再以真实行 2、4、231、16,059 覆盖历史用户图片、当前附件、多资源索引图片和工具历史图片，构造实际 `MediaPlan` 重读 **35 个资产、7,847,513 字节**；SHA256、MIME、尺寸与 manifest 全部匹配。这不是模拟 AgentSpark callback；报告为 `/tmp/relax-agentspark-adaptation-20261106/runtime-media-validation-v2.json`。

本轮完成离线数据入口、媒体身份/回填、K3 长度筛选与真实来源回归；原数据与模型目录未改动。仍保持 `training_ready=false`。30 条拒绝数据、真实 AgentSpark 的完整上下文/动态图片/可达性、V21 reward/token credit、完整权重加载及 GPU optimizer/恢复验收，均未被本轮通过项替代。
