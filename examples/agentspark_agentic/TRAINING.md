# Kimi K3 / AgentSpark V21 生产训练适配

生产训练输出、冻结 V21 奖励和 token credit 已接通，并完成本地 CPU 闭环验证。真实 AgentSpark / RM、完整权重、128 B300 更新与保存恢复仍需集群验收。审计入口 `client.py` 保持退出3；训练使用独立的 `production.py`。

## 本轮实现与来源

- 生产入口恢复已观察到的工具调用思考内容，固定 K3 thinking 与0.8采样温度，记录原始请求和实际转发请求；在完整 SSE 成功、评分成功后原子发布显式 SessionOutput。错误、取消、尾部 SSE 错误或评分身份不符均不发布训练输出。
- 连续轨迹、分支和上下文重置按真实 committed Forest 导出。所有上下文保留同一逻辑 `Sample.index`；共享祖先的生成只拥有一次 loss mask，观察 token 始终为0。
- 奖励复用用户 `formal-v21-40g-onestep-20261022-147edbae` run 的冻结 Miles Python 源码，以隔离 CPU worker 调用；策略环境、源码闭包、RM prompt文件和配置均带身份校验。RM返回必须满足请求身份、策略版本、prompt fingerprint和judge模型；失败不转成有效零分。
- 每个 Session 只请求一次真实RM。最后活动段提供评分轨迹；非anchor上下文在RM成功后携带显式稀疏0及同一评分证据。全部逻辑样本组成组后只执行一次冻结 V21 归一化，再分发到物理上下文。
- K3追问 credit只落在原始生成token中 `response` 通道的可见措辞，排除think、工具参数、观察和不拥有的祖先。没有重编码替换训练token。冻结XML规则标记文字，再映射回实际token，支持非规范UTF-8分词。
- 训练端传递双mask、CTR/clarification优势及逻辑分母，执行局部质量门控、辅助PPO和局部reference KL。核心只新增可选custom-loss字段通道及loss组合点；未新增依赖，也未改本轮参数解析、Controller或Service逻辑。

冻结配方：CTR辅助loss权重0.15，clarification权重0.1，主KL0.003，局部KL分别0.01/0.02；长度soft/hard限制、tail penalty及per-token cap均为0。保持每题4次采样、温度0.8、GRPO、TIS clip [0,2]、PPO clip [0.8,1.28]、logical Session mean。不要直接采用当前Miles插件的默认0.25/40/64。

生产多上下文优势在真实 Sample converter中处理，未使用仅支持scalar/每轮scalar的 `agentic-custom-advantage-path`；后者会覆盖本适配需要的原生response spans。这是该配方的显式扩展。原生动态token batching保留。

## 运行时图片与失败诊断

训练脚本通过 `AGENTSPARK_ALLOW_DYNAMIC_IMAGES=1` 启用工具执行后新出现的图片；默认关闭，避免已有作业在升级适配器后改变准入策略。只接收有界的内嵌 `data:image`，校验原始字节与图片解码；未知 HTTP、文件 URL 和本地路径仍被拒绝。初始数据图片仍按原清单验证。图片历史、重复出现次数和每段预算沿用冻结 `pinned_compat` 的含义；超出预算的位置明确保留省略标记。候选状态仅在模型响应完整成功后提交。

新段以适配器实际观察到的历史或工具定义变化为依据，从原始回调重新执行图片准入；没有移植 Miles 的 token 层 TITO 分段器。最终转发的上下文仍由原生 K3 Forest 校验并导出。

设置 `AGENTSPARK_DIAGNOSTICS_DIR` 后，失败 Session 会留下唯一的安全摘要文件（0600，专属目录0700），记录可关联的身份摘要、阶段、固定错误码及允许的 HTTP 状态。它不保存原始提示词、图片或 Session 凭据，也不会把失败变成可训练的零分。完整训练审计仍由 Session IO 生命周期管理。

Judge 原始返回需在奖励服务自身开启 `DOTSREWARD_RM_TRACE_ENABLE=1` 和 `DOTSREWARD_RM_TRACE_DIR`。原生 trace 包含完整提示词及响应，应放在权限受限的目录；它不会修改冻结奖励策略或 schema 校验。修改训练环境不能改变已运行的奖励进程，需要在该服务下次启动时生效。

上述环境设置及 `AGENTIC_CONCURRENCY` 都在下次提交时读取。每组仍为4次采样：并发2对应最多8个驻留 Session；外部服务容量及两个组的首请求屏障、清理和吞吐需要实际运行验证。

## 生成冻结契约与训练配置

从Relax目录、训练镜像的Python执行。`RUN_ROOT`是用户给定数据run目录；`CONFIG_DIR`须是所有Ray worker可见的新配置目录。不能假设当前机器的 `/tmp` 在其他节点共享。

```bash
export RELAX_ROOT=/absolute/shared/path/to/Relax
export RUN_ROOT=/mnt/tidalfs-wx-idc01/dataset/pai/mmx/runs/formal-v21-40g-onestep-20261022-147edbae
export CONFIG_DIR=/absolute/shared/path/to/new-k3-training-config
cd "$RELAX_ROOT"
mkdir -p "$CONFIG_DIR"

python -m examples.agentspark_agentic.v21_reward pin \
  --frozen-source "$RUN_ROOT/miles" \
  --run-config "$RUN_ROOT/stage/config.json" \
  --reward-source "$RUN_ROOT/miles/examples/async_dots_rl/components/dotsreward" \
  --output "$CONFIG_DIR/v21-reward-contract.json"

python -m examples.agentspark_agentic.training_inputs configure \
  --contract "$CONFIG_DIR/v21-reward-contract.json" \
  --overrides-output "$CONFIG_DIR/v21-training-overrides.json" \
  --compiler-defaults-output "$CONFIG_DIR/k3-compiler-defaults.json"
```

这些操作只读冻结来源；输出拒绝覆盖，权限0600。冻结run本次推导出的RM prompt fingerprint是 `32f52e20b1101c81`（源算法本来就是16位hex）。训练RM必须提供匹配的真实评分，不是把此值补进不兼容响应。

当前本机已生成上述三个文件，位于 `/tmp/relax-k3-128b300/`。若在集群改动源代码/配置路径，按实际共享路径重新pin，而不是手改hash。

## 服务与输入

使用 [CLUSTER_VALIDATION.md](CLUSTER_VALIDATION.md) 的节点/模型/图片预检。训练数据使用已转换的 `prepared-relax-v2.jsonl`，21,258条，SHA256为 `f0773d5083d9f5c2d104571d2b316ebd78f2d360644ab34fb56cef33583d5737`。原始21,288条中30条仍在显式拒绝清单；本配方不会静默修复它们。

本机并发训练连接候选为 `/tmp/relax-k3-128b300/agentspark-config.training-candidate.json`，已复用历史服务配置并设置动态端口；服务和节点路由尚未验收。AgentSpark配置由该候选填写：模型标签Kimi-K3、每次8192tokens、64MiB回调起始预算、已验证媒体根目录。并发生产设置 `callback_bind_host="0.0.0.0"`、`callback_bind_port=0`，`callback_public_base_url`中的端口也设为0。实际端口在监听成功后确定，不预先抢占。

原生SessionShard用SPREAD分布到worker，并启动节点本地Agent进程。`callback-host-map.json`须覆盖所有可能执行Agent的节点，以实际 `socket.gethostname()` 为key、AgentSpark能访问的IP/域名为value，例如：

```json
{"worker-hostname-01":"worker01.reachable.example","worker-hostname-02":"worker02.reachable.example"}
```

这不是Ray Dashboard地址。动态端口须可由AgentSpark回连；如果网络只允许固定网关，需要部署实际代理路由后验证，不能仅填写域名宣称可达。服务并发容量至少容纳 `AGENTIC_CONCURRENCY × 4` 个同时驻留Session；该值需根据真实AgentSpark容量填写，不能从128卡推算。

## 128 B300 预览与提交

脚本：[run-kimi-k3-agentspark-v21-128b300.sh](../../scripts/training/agentic/run-kimi-k3-agentspark-v21-128b300.sh)。Actor/Rollout共用128张物理卡；训练TP4/PP4/CP2/DP4/EP32，推理8个两节点16卡引擎。完整K3 HF权重经native bridge加载并推送给dummy初始化的SGLang引擎。

```bash
export MODEL_DIR=/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3
export PROMPT_SET=/absolute/shared/path/to/prepared-relax-v2.jsonl
export EXPECTED_PREPARED_SHA256=f0773d5083d9f5c2d104571d2b316ebd78f2d360644ab34fb56cef33583d5737
export AGENTSPARK_CONFIG="$CONFIG_DIR/agentspark-config.json"
export AGENTSPARK_CALLBACK_HOST_MAP="$CONFIG_DIR/callback-host-map.json"
export AGENTSPARK_V21_CONTRACT="$CONFIG_DIR/v21-reward-contract.json"
export V21_TRAINING_OVERRIDES="$CONFIG_DIR/v21-training-overrides.json"
export K3_COMPILER_DEFAULTS_JSON="$CONFIG_DIR/k3-compiler-defaults.json"
export AGENTSPARK_REWARD_URL=http://VERIFIED_REWARD_PROXY:PORT
export AGENTIC_CONCURRENCY=VERIFIED_RESIDENT_GROUP_COUNT
export SAVE_DIR=/absolute/shared/path/to/new-checkpoints
export RAY_DASHBOARD_ADDRESS=http://VERIFIED_RAY_HEAD:8265

# 填好以上值后，只打印命令，不访问Ray/服务或申请GPU。
DRY_RUN=1 bash "$RELAX_ROOT/scripts/training/agentic/run-kimi-k3-agentspark-v21-128b300.sh"

# 真实提交：先执行离线文件/契约检查，再提交当前已运行集群。
DRY_RUN=0 bash "$RELAX_ROOT/scripts/training/agentic/run-kimi-k3-agentspark-v21-128b300.sh"
```

默认18题×4采样=72逻辑样本/更新，21,258÷18=1,181次rollout覆盖一轮；`EPOCHS`可增加轮数。脚本拒绝不能整除样本总数的batch，避免原生数据源从下一轮补尾批。原始数据拒绝、运行失败和真实上下文超限均不能算作成功训练覆盖。脚本保留优化器保存，未启用`no-save-optim`；恢复是否完全保持训练/数据进度仍需真实验证。

脚本强制 `RELAX_SKIP_CLUSTER_CLEANUP=1`，不停止其他任务或移除已有placement group。启动脚本依赖已有、完整的训练镜像；不安装包或重建CUDA。除命令dry-run和CPU验证外，本轮没有运行真实提交。

## 已验证与仍需验收

已验证：指定真实K3 tokenizer/processor的原生导出；冻结V21请求/响应/组策略；loss和梯度与冻结源函数对照；静态/动态CP的thd、bshd及媒体padding切分；TransferQueue编码/重排；4个逻辑Session各3个原生上下文，经converter到loss反传的CPU闭环；完整准备数据重新读入并核对SHA。

CPU闭环使用受控模型输出和受控RM响应，不加载完整权重。还需真实集群证明：AgentSpark/RM/CTR及回调可达、全组工具/图像交互、16节点128卡布局与NCCL、完整权重/optimizer内存、首步有限loss/梯度与参数变化、8个推理引擎权重更新、保存后新任务恢复。未支持/未验证的组合会明确拒绝，包括token全局归一化、额外OPD/custom PG reducer等。

完整pre-commit需能够取得仓库hook；当前离线环境缺缓存，不能把本地格式/语法检查算作全部hook通过。运行命令与最终回归结果保存在本次 `/tmp/relax-k3-128b300/` 产物目录。

本轮最终回归：综合487 passed / 1测试fixture重复参数失败 / 4 skipped；修复fixture后定向4 passed，覆盖失败项及启动检查。合计488个不同用例通过，无未解决失败；4个skip均为缺少本地Qwen对照模型。全量runtime gate 21,258/21,258通过，5,106条空时间保持unknown。完整证据与源码/配置SHA见 `/tmp/relax-k3-128b300/v21-delivery-manifest.json`。
