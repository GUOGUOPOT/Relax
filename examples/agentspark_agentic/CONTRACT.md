# AgentSpark → Relax / Kimi K3 实施契约

状态：范围 B 已选定；阶段 0–2 已完成初始离线实现。用户现已提供模型来源，授权测试完成后继续阶段推进。必要的 Service/parser 修复已获批应用，阶段 3 的上下文、媒体和原生显式导出离线验收也已完成；旧模型来源的 CPU 合并回归已通过；新模型路径和本次全量数据适配证据统一见 MODEL_VALIDATION.md。本文**不是冻结的训练 plan，也不表示可以启动训练**。

## 范围与来源

范围 B 的最终目标是在 Relax 原生 agentic 路径中保留完整 V21-GRPO 语义：AgentSpark 业务流程、轨迹与图片语义、V21 奖励及 CTR 局部 token credit。阶段 0–2 交付接口契约、K3 编码适配和 AgentSpark adapter；完整 V21 token credit、训练损失、真实模型与 GPU 验收属于后续阶段。

| 来源 | 固定提交 | 用途 |
| --- | --- | --- |
| Miles | `b201928bab03b4a70d70a6adbaf4fb43d9c755aa` | AgentSpark Replay/Send、任务与奖励、历史/图片/R3/SC 行为的参考来源 |
| Relax | `68a89f05acf47767db902157283ca3db04c816ff` | 原生 agentic Runtime、SessionForest、训练导出与模型接口的实施基线 |

这些提交标识代码来源；本次工作区改动及后续依赖、模板、模型和镜像身份须分别记录，不能由提交号推断运行环境已验证。

| 阶段 | 本次交付 | 验收边界 |
| --- | --- | --- |
| 0：契约 | 本文；责任、数据、协议、未知项和验收范围 | 可以据此审阅阶段 1、2；不生成平台资源选择或启动 plan |
| 1：K3 编码 | 消息编码、模板及 reasoning/tool-call 协议的离线适配 | fixture 与指定真实来源验证分别记录；后续真实来源发现的问题见 MODEL_VALIDATION.md，不能将旧 fixture 通过等同于完整 K3 通过 |
| 2：AgentSpark adapter | 接入 Relax 原生 agentic，完成输入、Replay/Send、模型回调、终态输出与清理 | 本地可控 fixture 验证协议和失败路径；不访问真实 AgentSpark、RM、CTR 或平台 |
| 3：上下文与原生导出 | 离线上下文分析、明确选择、调用身份绑定；以原生 Service/Forest 验证提交状态、图片归属和逻辑采样身份 | 原生提交/导出检查通过；最新模型与数据验证见 MODEL_VALIDATION.md，不开放训练输出 |
| 后续 | 真实业务轨迹、V21 奖励与 token credit、训练损失、完整 K3 权重、R3/SC、GPU 更新与恢复 | 逐项定义证据后实施；离线成功不能代替这些验收 |

最小改动限于所需适配代码、相关 CPU 验证和文档。沿用 Relax 的 Runtime、SessionForest、rollout、训练、权重同步和 checkpoint；不迁入 Miles 的训练调度栈，不新增依赖或推测实际服务器。

## 组件职责

- **AgentSpark** 执行 agent loop 和业务工具，管理自身对话、Replay 和技能/工具加载。
- **adapter** 将数据送入 AgentSpark，将其模型调用指向本次 Relax Session，并保留协议证据。本次阶段 2 仅产生独立审计产物，`training_ready=false`；完整范围 B 验收前不写 `RELAX_OUTPUT_JSON`，CLI 即便采集成功也以非零码退出，阻止 Relax 在缺少输出时自动执行隐式导出。适配层不自行计算或重建训练 token、logprob 或模型权重。
- **Relax** 提供模型请求路由、K3 生成、SessionForest 已提交状态、训练样本导出及训练生命周期。
- **DotsReward / RM / CTR** 提供评分证据；完整 V21 的质量门控、逻辑采样归一化、局部 token credit 和辅助损失仍须独立迁移。模型 Session 不承担奖励评分。

## adapter 输入与输出

运行入口遵循 [Relax agentic 集成契约](../../skills/agentic-rollout/references/integration-contract.md)，由 `--use-agentic-rollout` 选择原生路径。该路径不能依靠 `--custom-generate-function-path` 接管循环。

| 输入 | 约定 |
| --- | --- |
| `RELAX_INPUT_JSON` | 读取本 Session 的 messages、metadata 及逻辑身份；实际字段以 Relax 的输入协议为准 |
| `RELAX_BASE_URL` | 本次 Session 的模型 API 基地址；保留 Runtime 提供的完整路由前缀 |
| `RELAX_SESSION_ID` | Session 路由身份及 Bearer credential；不得写入日志、产物或固定配置 |
| `RELAX_OUTPUT_JSON` | 后续完整范围 B 训练导出的原子写入位置；阶段 2 禁止写入，审计产物必须使用不同路径 |
| `RELAX_SESSION_IO_DIR` | Runtime 拥有的临时目录；仅使用当前 Session 的文件 |
| `RELAX_ROLLOUT_MODE` / `RELAX_GROUP_ID` | 保留 train/eval 与逻辑 Group 身份；不得用物理导出行数替代逻辑采样数 |
| 外部 AgentSpark 配置 | 地址、模型标识、工具/技能名单、prompt、超时和认证均由本次明确输入提供；缺失必填项时给出错误，不借用 Miles 的历史服务地址 |

数据中的当前问题、历史、附件、资源 ID、rubric 和评分标签各保留原职责。源数据转换器应优先选择 `metadata.replay_history`，缺失时按来源契约使用 `history`，并生成明确的 `metadata.agentspark.replay_spec`；不得将多轮历史拼接成当前问题。运行时 adapter 仅接受已准备的 `metadata.agentspark` envelope，不从顶层 messages 或旁路 history 字段猜测业务输入。独立离线 `prepare.py` 现显式调用指定 Miles checkout 的纯输入编译器，将原始 prepared JSONL 转为该 envelope；源任务 prompt/tools/预算由冻结配置提供并记录指纹。长文本沿源规则将 UTF-8 后缀放入 TEXT、前缀放入 user prompt；源问题和评分 metadata 完整保留。运行时仍拒绝 TEXT 总量超过 1900 UTF-8 字节的未准备输入。使用方式见 README.md。

输出分为任务终态与训练导出：

- 仅在确认真实终态后记录成功；错误、取消和远端执行结果未知必须可区分。
- 显式导出使用规范 Chat 形状的完整 `messages`、实际 `tools`、`chat_template_kwargs`、唯一 `name` 和所需 metadata；多个导出使用 JSONL，不能写成 JSON 数组。
- 所有导出必须指向 Relax SessionForest 的已提交状态。不能伪造空轨迹、成功 reward、训练 token 或 logprob 以通过下游。
- 阶段 0–2 不实现完整 V21 reward。若后续采用标量 reward 做独立连通性基线，应明确命名，不能声称与范围 B 的完整 V21 目标等价。

## Replay、Send 与模型回调

1. 为本次逻辑采样建立独立的 AgentSpark 身份，保留必要的用户绑定与资源 ID 映射；不复用其他任务的 session。
2. 有历史时先执行 Replay：`replay_type=restore_only` 位于 `replay_spec` 内。Replay 只恢复历史，随后将当前问题经普通 Send 发送。
3. Send 显式传入本次 `prompt_spec`、`tools`、必要的 `tool_versions`、`capabilities` 和 `skills_name_list`；缺省规则不得偷偷切换到平台默认工具集合。
4. `llm_config` 指向 Relax 的 Chat Completions 接口。必须验证客户端最终拼接的 URL 保留 `RELAX_BASE_URL` 前缀，并用 `RELAX_SESSION_ID` 配置 Bearer 认证；不沿用 Miles 的 session 路径或固定 `api_key=local`。
5. 每个模型请求携带完整、实际可见历史。K3 的 reasoning/tool-call parser、模板及 thinking 参数由 K3 的已核实契约决定，不复制 Qwen 的 parser 名或 thinking 档位。
6. 消费 AgentSpark SSE 到 EOF；`agent.stream.end` 后仍可能出现资源索引 snapshot。必须区分模型 API 的 JSON/Buffered SSE 与外层 AgentSpark SSE。
7. **不自动重发 Send。** 网络中断、超时或收到不完整事件不能证明业务工具没有执行；记录执行结果未知，避免重复副作用。协议没有证明幂等时，不凭 request ID 假设可以安全重试。

模型请求的超时必须覆盖 Relax prelaunch/partial/fully-async 可能保留请求的时间；它与 `--agent-timeout` 的 agent 活跃时间预算分别配置。阶段 0–2 不自行选择并发槽位或外部服务容量。

当前 Relax Chat 归一化消费 `messages`、`tools`、`chat_template_kwargs`、`max_completion_tokens`/`max_tokens`、`stop`、`seed` 和 `logprobs`。顶层 `temperature`、`top_p`、`tool_choice`、`response_format` 不在这一请求转换路径生效；不能把审计中出现这些字段当成参数已执行的证据。采样默认来自 Relax Session 配置，模板身份必须合并 compiler defaults 与请求覆盖。后续真实联调需核对 AgentSpark 是否依赖这些未消费参数；本次不扩展 Service 的参数接口。

取消时关闭当前连接并执行已实现的本地清理；只调用已核实的远端取消/清理接口。若 AgentSpark 尚无可验证取消协议，则记录远端执行状态未知，不能声称已停止工具或已释放远端资源。

## 历史、工具、图片与 SessionForest

`messages`、`tools` 和模板参数共同构成模型可见上下文。保留 assistant reasoning、tool-call IDs、参数和结果；空工具结果须使用一致的非空规范表示，并在后续请求和导出中复用同一表示。

Miles 会为某些明确匹配的历史工具调用恢复遗漏 reasoning，并在历史或 tools 改写时建立新 segment。Relax 接入必须显式决定对应策略：正常追加、预期分支、压缩/重置分别可观察。不能把不确定的旧 reasoning 插回当前输入，不能把所有变化强行当线性追加。

- 只在线性历史已经证明时使用隐式导出；压缩、删改历史、工具集合变化或多分支需要显式导出。
- 一个逻辑 Session 导出多个 contexts 时，保持同一个逻辑 `Sample.index`；不得把多个导出当成额外 GRPO 兄弟。
- 多 context 训练要求显式 JSONL、custom advantage 和动态 batching；advantage 必须与所选 context 对齐。Eval 的聚合另行验收，不能直接按物理行数计算逻辑任务成功率。
- 图片必须绑定到当前上下文中实际出现的位置和内容；移动、删除、重复图片不能从旧消息位置自动补齐。保留 Replay 的资源映射，验证 processor 展开长度和模型实际图片输入。
- 当前 Relax agentic 的新增训练媒体边界见原生契约；不能因 AgentSpark 能返回某种媒体便宣称 Relax 可训练该媒体。

多 context 的具体导出与分数规则以 [agentic training contract](../../skills/agentic-rollout/references/agentic-training-contract.md) 为准。

## 阶段 1 实现边界

K3 首轮和增量 observation 使用 checkpoint 原生 `apply_chat_template(tokenize=True)`，保留结构 token 与文本字面量的区别；不重编码已生成的 response tokens。训练侧与 SGLang 侧分别编码、验证和裁剪校准前缀。图片在 SGLang 输入中每图保留一个原始媒体槽，训练侧按 processor 的 grid 展开。非 K3 保留原路径，SFT 复用同一图片展开函数。

当前 K3 processor 在 CPU 线程执行器中运行，不使用通用的 rendered-text 进程池。后续验证配置应显式设 `--mm-processor-pool-size 0`；尚未提供 K3 进程池的隔离或吞吐验收。指定来源的 tokenizer、processor 和 reasoning/tool-call parser 已通过 CPU 回归，GPU 路径仍待验收；可通过 `KIMI_K3_SOURCE_DIR` 启用本地来源测试。

## 阶段 2 实现边界

实现和命令见 [AgentSpark adapter README](README.md)。协议层接收明确准备包，完成 GetSession 或 Replay→Send；无私有媒体计划时模型回调原样转发；有媒体计划时校验本地资产字节，将授权历史 marker 和图片引用恢复为 data URI，再转发到 Relax。审计分别保留原始请求与实际转发请求，阶段 3 使用实际转发上下文校验已提交 Forest。外层 SSE 读至 EOF，保留尾部业务 snapshot；错误响应正文不进入审计。所有回调和本地资源关闭后，才根据最终采集记录判断线性候选。候选只证明观察到的消息追加关系，尚未证明 SessionForest 提交身份。

审计文件保留 source 输入、评分元数据、group/mode/model、回调证据和业务事件，使用独立私有路径原子创建，不覆盖已有文件。CLI 采集成功返回 `3`，失败审计返回 `1`，无法产生审计返回 `2`；任何路径均不发布训练输出或伪造奖励。`training_ready` 始终为 `false`。这意味着阶段 2 可以用于协议取证，但不能直接挂到正式训练任务当作完成的范围 B harness。

未知远端取消协议仍是未知项。只关闭本地连接不等于停止远端工具；可由部署方注入已核实的清理函数，其完成也不自动证明资源已释放。发生 Replay/Send 中断时保留未知状态，不自动重发。

## 阶段 3 实现边界

`contexts.py` 在独立离线分析中消费完整的阶段 2 审计记录，复用 Relax 原生 Chat 规范化、消息检查和状态摘要规则。调用方必须明确提供 compiler 的模板默认值；请求覆盖与默认值合并后才构成候选身份。每次响应形成一个候选，并记录追加、分支或重置、tools/template 变化及已识别的 reasoning 删改。原始回调保留，不将不确定的旧 reasoning 补回请求。

- 没有选择清单时只分析，不选择任何上下文。选择项必须包含唯一名称、回调序号和选择意图；相同规范状态的重复生成因 token 归属不明确而拒绝选择。
- 阶段 2 中只有明确的纯拓扑失败允许继续分析；所有回调、一次 Send、SSE 终态/EOF、本地清理证据仍必须完整。并发记录、传输失败或缺失回调不会因此被放行。
- 图片按消息位置、内容块位置和出现次序记录。引用摘要不等于图片内容摘要，重复引用不自动去重；真实字节、processor 特征与 Forest 媒体归属由原生测试另行验证。
- `validate_native_exports` 只能接收调用方持有的原生 Shard/Session 和调用身份。它绑定阶段 2 审计到同一次 Session 调用，校验实际 compiler defaults，并调用原生 `SessionOutput` 与导出构造方法；不从审计重建 Forest，不编造 token 或 logprob，也不提供新的远端验证 API。
- 一个 Session 选择多个 context 时，保持相同逻辑 `Sample.index` / `group_index`；输出 metadata 与数据集 `train_metadata` 各保留其原用途。多 context 后续训练仍需 custom advantage、动态 batching 和完整 V21 token credit；Eval 默认只允许选一个代表上下文。

本阶段仍输出独立私有审计，CLI 成功返回 `3`，始终 `training_ready=false`，不写 `RELAX_OUTPUT_JSON`。受控后端测试只验证状态和导出协议；完整模型生成、真实工具行为、奖励与 optimizer step 仍需后续验收。

## 范围 B 的后续算法工作

完整 V21 包含 response-aligned `followup_token_masks` / quality masks、局部 scalar 与 flags、逻辑 rollout 分母、质量门控、辅助 policy loss 和局部 reference KL。Relax 当前 custom advantage 的标量/逐 assistant-turn 返回值不能直接替代这些任意 token 级目标。

后续需分别交付：

- DotsReward/RM/CTR 的输入输出与版本身份；回答质量、硬规则和 CTR 证据完整传递。
- K3 原始生成 token 上的可见追问 mask，排除 reasoning、工具参数和非训练 token。Miles 的 `<tool_call>` / `</think>` 规则属于 Qwen 假设，不能直接用于 K3。
- token 字段的传输、packing、CP 分片、逻辑分母和辅助 loss/KL；以固定张量验证 loss 与梯度，而非只验证返回字段存在。
- 多 context 下 reward 只按逻辑采样计一次，保留所选上下文与训练 token 的对应关系。
- R3 的 K3 层数/top-k/专家身份、展开 token 坐标和暂停恢复；SC 的逐 token 候选 ID/logprob、候选宽度和重要性加权。R3/SC 需独立确认与验收，均不属于阶段 0–2 的完成承诺，也不改变范围 B 已选定的 V21-GRPO 目标。

## 未定输入与验收状态

用户当前指定真实 K3 来源 `/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3`，以及 `formal-v21-40g-onestep-20261022-147edbae/data/prepared.jsonl`。当前实际验证和已暴露缺口统一记录在 [MODEL_VALIDATION.md](MODEL_VALIDATION.md)；旧的无来源测试结果保留为阶段 0–2 历史记录。其余外部服务和运行资源仍未定，不填入历史服务器或猜测参数。

| 项目 | 当前状态 |
| --- | --- |
| K3 checkpoint、tokenizer、chat template、processor 和精确模型版本 | 路径已指定；源码指纹与 CPU 编码/bridge 验证见 MODEL_VALIDATION.md；Service 接线和 parser 修复已应用，完整权重/GPU 未验收 |
| AgentSpark 实例、认证、服务版本、远端取消协议与可用槽位 | 未定；没有访问真实服务 |
| RM / CTR 的地址、模型、版本和归属 | 未定；没有评分服务验收 |
| 业务任务 profile、工具/技能名单及对应数据版本 | 已从指定数据 run 的 stage/config.json 与冻结 profile 核实为 think_online、no-Loader；工具、版本及预算沿源配置，身份与转换证据见 MODEL_VALIDATION.md |
| 平台、区域、workspace、队列、卡型/卡数、镜像、挂载及 Ray 地址 | 未定；当前申请资源为 0 |
| 训练量、输出/checkpoint 目录、恢复起点和保留策略 | 未定；没有可启动 plan |
| 阶段 0 契约 | 已记录；须与阶段 1、2 实际接口一同复核 |
| 阶段 1 编码与真实来源验收 | 初始 fixture 验收已完成；现指定来源已启用，Service/parser 修复后的 CPU 回归通过，证据见 MODEL_VALIDATION.md |
| 阶段 2 adapter 离线验收 | 已实现；74 项本地协议/回调/生命周期测试通过；本地伪服务通过不等于平台连通 |
| 阶段 3 上下文/媒体/原生显式导出 | 40 项新增检查通过；包含真实 K3 编码与受控 backend 的原生提交/导出，尚无真实业务生成或训练更新 |
| 完整 V21、真实多轮工具/图片、GPU optimizer、保存/恢复 | 未实施/未验收；不计入本次阶段 0–2 完成 |

阶段 1、2 验收至少记录：实际命令与结果、字段保留/非法输入拒绝、URL/认证、Replay→Send 顺序、SSE 尾部事件、Send 不重发、失败/取消与资源清理。提交前按仓库规则完成适用检查；依赖缺失或 GPU/外部服务未测须标明原因，不计作通过。

## 本地验证命令

以下保留阶段 0–2 的历史验证记录；旧来源的 Service 修复和阶段 3 合并结果为 **293 passed / 0 failed / 0 skipped**；当前指定来源与数据的最新结果见 [MODEL_VALIDATION.md](MODEL_VALIDATION.md)。

在 Relax 仓库根目录执行；不加载模型权重、不申请 GPU：

```bash
CUDA_VISIBLE_DEVICES='' /opt/sglang/bin/python -m pytest \
  tests/cpu/agentic/test_kimi_k3_compiler.py \
  tests/cpu/utils/data/test_kimi_k3.py \
  tests/cpu/agentic/test_protocol_consistency.py \
  tests/cpu/agentic/test_protocol_errors.py \
  tests/cpu/test_agentic_rollout.py -q
```

阶段 1 结果：150 passed / 5 skipped。`KIMI_K3_SOURCE_DIR` 未设置时，真实来源项明确跳过；提供来源后需要重跑并单独审阅，跳过不计入真实模型验收。既有 SFT 图像展开正文还与基线作机器比较，仅有 `len(images)` 到 `image_count` 的参数化；原诊断字符串保持不变。

阶段 2 单独回归：

```bash
CUDA_VISIBLE_DEVICES='' /opt/sglang/bin/python -m pytest \
  tests/cpu/examples/agentspark_agentic -q
```

最终合并验证在上述阶段 1 命令中追加 `tests/cpu/examples/agentspark_agentic`，并设置 `PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`：**224 passed / 5 skipped**（21.81 秒）。5 个真实来源项仍未验收。测试使用 CPU fixture、HTTP mock 和本机临时回调服务器，没有访问真实 AgentSpark/RM/CTR、下载模型或提交 GPU 训练。

静态检查已完成：Python 语法/编译、按仓库配置的导入排序、冲突标记及 `git diff --check`。完整 `pre-commit run --all-files` 已尝试，但本地缺少 hook 缓存，离线初始化被 `transport 'https' not allowed` 阻止；不能将完整 pre-commit 计为通过，也未为此安装依赖。当前保留仓库的导入后双空行规则；本地 Black 与该规则存在空行格式差异。

## 源码核对入口

- Miles `examples/async_dots_rl/components/dotsbridge/clients/agentspark_client.py`：Replay/Send 与 SSE transport。
- Miles `examples/async_dots_rl/components/dotsbridge/adapters/agentspark_environment_adapter.py`、`agentspark_replay_adapter.py`：输入、长文本与资源历史。
- Miles `miles/agentspark_session/turn_policy.py`、`reasoning_history.py`、`segment_images.py`：历史、分段与图片语义；这些实现依赖 Miles Session，迁移行为而非复制整套 Runtime。
- Miles `miles_plugins/algorithms/v21/{extension,transport,advantages,loss,token_mask,reward}.py`：完整 V21 数据与损失语义。
- Relax `relax/agentic/session/service.py`、`state.py`、`pipeline/reward.py`、`runner/ipc.py`：目标端协议、已提交上下文、分数及进程输入输出。

## 128 B300 启动前检查（阶段 C）

用户已确认目标为16节点×8卡、全部接受数据的完整RL。阶段C当前结论是 **NEEDS_CHANGES / UNVERIFIED，Ready to launch: no**：生产训练输出与冻结V21 reward/token credit已完成源码接线和CPU闭环；服务可达性、回调路由、外部slots、完整权重/optimizer及恢复仍无目标集群证据。当前预检脚本、服务配置候选来源与可运行命令统一见 [CLUSTER_VALIDATION.md](CLUSTER_VALIDATION.md)；128卡参考并行方案记录在非执行的 `training-plan-128b300.template.json`。本轮新增文件/硬件预检不改变既有审计CLI退出3和不发布训练输出的约束。

## 生产训练接线（本轮阶段B实现）

新增独立production入口及原生Sample converter，完整实现logical group优势、K3 response通道追问mask、共享祖先所有权、质量门控、辅助PPO与局部reference KL。采用冻结formal run的配置与policy源文件，不以当前Miles同名policy代替。多上下文优势由converter处理，避免metadata-only custom-advantage钩子覆盖原生response spans；动态batch与logical分母保持原生契约。实现和验证细节、128卡命令见 [TRAINING.md](TRAINING.md)。

本轮核心改动仅为custom-loss额外字段传输和可选loss组合接口；Service/Controller/参数解析不需新增修改。CPU数值/梯度对照与实际分布式验收分别记录，Ready to launch仍需外部环境证据。
