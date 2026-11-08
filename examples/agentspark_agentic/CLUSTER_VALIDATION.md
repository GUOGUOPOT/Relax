# 16 × 8 B300：当前可执行的验证与完整 RL 前提

生产训练输出、冻结 V21 奖励和 token credit 已完成代码适配，并通过本地 CPU 闭环；实际服务和128卡训练仍待验收。完整配方、配置生成和提交步骤见 [TRAINING.md](TRAINING.md)。审计CLI继续退出3，正式训练使用独立production入口。

本目录提供：

- `cluster_preflight.py`：检查现有集群、文件身份、GPU 分配及可选 NCCL。不会创建 Ray 集群、加载完整权重、启动训练、停止其他任务或删除已有 placement group。
- `agentspark-config.template.json`：需要填写的服务连接模板。配置含义见 [README](README.md#explicit-configuration-and-standalone-cli)。
- `training-plan-128b300.template.json`：有源码依据的并行方案与未完成项，**不是启动命令或可直接传给 train.py 的配置**。

## AgentSpark 配置从哪里来

Miles 已有相关配置，可从当前 checkout 的 `.claude/skills/rai-train-v02/assets/recipes/qwen38-flash-next.json` 和数据 run 的 `stage/config.json` 核对 `agent.base_url`、`source`、`agent`、`max_tokens`。不要把整个 runtime_env、历史凭证或 Qwen 模型别名复制到新任务。服务地址存在于配置，不证明当前存活、跨集群可达、K3 别名受支持或外部并发充足。

用户给定 run 的非敏感连接字段已整理为私有候选 `/tmp/relax-k3-128b300/agentspark-config.candidate.json`，来源记录在同目录 `agentspark-config.provenance.json`。它复用历史服务地址、`eden`、`diandian` 和每轮 8192 token 上限，覆盖为本次已验证图片根目录及 64 MiB 起始回调预算；模型标签设为 `Kimi-K3`，回调监听端口和对外回调地址仍是占位符。原生 Relax 按 Session 路由，模型标签不负责选择权重；AgentSpark 对该标签的接受情况仍需联调。8192 是历史配置值，完整 K3 上下文预算尚未验收。

网络上有三段连接：adapter 请求 AgentSpark；AgentSpark 请求 adapter 的模型回调/图片地址；adapter 请求本次 Relax Session。`base_url` 指第一段的 AgentSpark 服务；`callback_public_base_url` 指第二段的入口，不能填 Ray Dashboard 或 `0.0.0.0`。`RELAX_BASE_URL` 和 Session Bearer 由 Relax 在运行时注入，不应手工写入配置。

## 先把验证产物放到共享路径

上一轮生成的 `/tmp/relax-agentspark-adaptation-20261106/` 只保证当前机器存在，不能假设所有节点共享 `/tmp`。请将下列文件复制到所有 16 节点可读的同一绝对路径；不要覆盖已有数据：

- `prepared-relax-v2.jsonl`：21,258 条接受数据；完整 SHA256 为 `f0773d5083d9f5c2d104571d2b316ebd78f2d360644ab34fb56cef33583d5737`。
- `all-media-manifest-v2.jsonl`：36,707 个已验证本地图片路径；SHA256 为 `d1e883d009e645a89b97c536231889c4fb09e9c9e8238ffced1f34f958d3605c`。
- `validation-v2.json`、`preparation-report-v2.json`、`rejections-v2.jsonl`：保留数据身份、转换策略及30条拒绝记录。

图片本体不在 JSONL 内，各节点仍需挂载清单中的原始绝对路径。当前模型 `encoding_k3.py` SHA256 为 `b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333`。详见 [本地验证记录](MODEL_VALIDATION.md)。

## 运行预检

从已经具备相同 Ray/Python 环境、可以连接现有集群的驱动节点运行。`--ray-address` 使用 GCS 或 Ray Client 地址，例如 `HEAD_IP:6379`；它不是 `http://HEAD_IP:8265` 的 Jobs/Dashboard 地址。脚本要求显式地址，不自动选择集群。

下面路径需要替换为实际共享目录。`MODEL_DIR` 是 **Kimi-K3 目录本身**；`DATA_DIR` 是上述转换产物所在目录；`SAVE_DIR` 是报告目录。报告文件必须不存在。

```bash
export RELAX_ROOT=/absolute/path/to/Relax
export MODEL_DIR=/mnt/tidalfs-wx-idc01/dataset/pai/MODEL_CACHE/Kimi-K3
export DATA_DIR=/absolute/shared/path/to/validated-k3-data
export SAVE_DIR=/absolute/shared/path/to/new-test-reports
export RAY_ADDRESS=HEAD_IP:6379

PREFLIGHT_ARGS=(
  --ray-address "$RAY_ADDRESS"
  --model-dir "$MODEL_DIR"
  --prepared-data "$DATA_DIR/prepared-relax-v2.jsonl"
  --media-manifest "$DATA_DIR/all-media-manifest-v2.jsonl"
  --media-root /mnt/tidalfs-jdbj01/dataset/dots_rl/xianqing
  --media-root /mnt/tidalfs-jdbj01/dataset/dots_rl/yufeng/rl_datas/vlm_datas/role_play_0915
  --media-root /mnt/tidalfs-wx-idc01/dataset/pai/mmx/runs/formal-v21-40g-onestep-20261022-147edbae
  --nodes 16
  --gpus-per-node 8
  --expected-shards 96
  --timeout-s 600
  --expected-prepared-sha256 f0773d5083d9f5c2d104571d2b316ebd78f2d360644ab34fb56cef33583d5737
  --expected-media-manifest-sha256 d1e883d009e645a89b97c536231889c4fb09e9c9e8238ffced1f34f958d3605c
  --expected-model-encoder-sha256 b9cb7ae100fed34b9337f80dacee5abbf7e261fe9b74bc0e76366701d46f5333
)

# 只展示检查计划，不连接 Ray、不读完整数据、不占 GPU。
python "$RELAX_ROOT/examples/agentspark_agentic/cluster_preflight.py" \
  "${PREFLIGHT_ARGS[@]}" \
  --report "$SAVE_DIR/preflight-preview.json" \
  --dry-run

# 检查16个节点和128张卡，同时执行数值校验的NCCL all-reduce/all-gather。
python "$RELAX_ROOT/examples/agentspark_agentic/cluster_preflight.py" \
  "${PREFLIGHT_ARGS[@]}" \
  --report "$SAVE_DIR/preflight-128-b300.json" \
  --driver-version \
  --nccl
```

正式预检会申请 128 个本任务 GPU actor；有 GPU 被占用时可能等待并超时，不会抢占或停止现有任务。各节点文件检查先通过，才会申请 GPU。只清理本次创建的 actor 和通信组。

省略 `--nccl` 时不做分布式通信测试，报告中该项为 `UNVERIFIED`。增加 `--local` 只做当前机器文件检查，不导入 Ray、不使用 GPU；成功时仍返回 `3`，表示集群未验证。`--dry-run` 返回 `0` 仅表示计划生成成功，状态为 `NOT_RUN`。其他失败返回非零；报告中的 `training_ready` 始终为 `false`。

默认验证范围是：driver 完整计算 prepared SHA256；每节点比较 prepared 大小与首尾块指纹、关键模型源码指纹、96个 index 引用权重分片的存在性/大小、完整媒体清单哈希，并从每个媒体根目录抽取代表图片检查真实字节。**没有逐节点计算全部模型权重哈希，也没有逐节点读取全部36,707张图片。** 可加 `--node-data-sha256` 在每节点完整计算 prepared SHA256，会显著增加共享存储读取量。检查范围会写入报告。

## 真实集群仍需验收什么

当前源码参考拓扑为训练 TP4 / PP4 / CP2 / EP32 / ETP1、PP层数21+24+24+24，推理8个两节点16卡引擎。Actor 与 Rollout 共用128张物理卡；它们不是256张卡。参考来源是 OpenR1-MM 任务，不能继承其数据字段、reward或 `<think>/<answer>` 提示当作 AgentSpark 配方。

已接通生产原生训练输出、冻结 V21 奖励及 token credit；还需完成真实 AgentSpark 全组多轮、完整权重加载、optimizer 更新及保存恢复验收。AgentSpark 外部 slots 必须按 resident Groups × 每组采样数核实，不能从128张GPU推算。多Session回调的监听端口/代理路由也需实际配置。

生产配方默认18题×4次采样、每轮1,181次rollout，完整覆盖21,258条接受数据；拒绝不可整除的batch，避免原生数据源跨epoch补尾。源数据30条拒绝及运行时失败不能计入成功覆盖。详见 [TRAINING.md](TRAINING.md)。

## 本轮脚本验收

新增预检与既有配置/transport 的 CPU 回归 **67 passed，0 skipped**（2.64 秒）；AST、仓库 isort 配置、JSON模板和 dry-run 无执行副作用检查通过。使用用户给定模型、上一轮转换数据和37k媒体清单运行 `--local`，三个预期SHA一致、96个引用分片非空且可达、三个媒体根的代表文件哈希一致，无文件错误。该模式正确报告 `UNVERIFIED` / `training_ready=false` 并退出3，没有连接集群、加载权重或执行GPU运算。

报告与测试日志位于 `/tmp/relax-k3-128b300/`：`preflight-local.json`、`preflight-dry-run.json`、`cpu-tests-final.txt`。完整 pre-commit 已尝试，仍因本地缺hook缓存、离线禁止HTTPS fetch而未完成，未安装依赖。没有实际128卡/NCCL/完整RL验收证据。
