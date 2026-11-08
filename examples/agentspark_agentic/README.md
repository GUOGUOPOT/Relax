# AgentSpark adapter for native Relax agentic rollout

This example includes **Stage 2 protocol auditing, Stage 3 context/native-export validation, and a separate production agent command** for the [integration contract](CONTRACT.md). The audit and context CLIs remain read-only with respect to `RELAX_OUTPUT_JSON`; their reports retain `training_ready: false`. The production command below publishes explicit contexts only after a complete real AgentSpark invocation and verified V21 scoring. Deployment acceptance remains separate from CPU tests.

A successful standalone audit CLI run exits **3**, not 0. Relax can implicitly export a context when an agent exits successfully without an output file; the nonzero exit deliberately prevents that behavior. Do not configure the audit CLI as a production `--agent-command`; use the separate production entrypoint below. Use the library API and local fixtures for protocol validation; model/template, media/context identity, V21 reward/token credit, and deployment acceptance remain separate gates.

## Production agent command

The production entrypoint is `python -m examples.agentspark_agentic.production`. It uses the same prepared input and adapter configuration. Supply separate files for exact K3 compiler defaults (including a boolean `thinking`) and the frozen V21 reward contract:

```bash
python -m examples.agentspark_agentic.production \
  --config "$AGENTSPARK_CONFIG" \
  --compiler-defaults "$K3_COMPILER_DEFAULTS_JSON" \
  --reward-contract "$V21_REWARD_CONTRACT" \
  --reward-base-url "$V21_REWARD_BASE_URL" \
  --sampling-temperature 0.8 \
  --callback-host-map "$AGENTSPARK_CALLBACK_HOST_MAP"
```

Run this command through native `--use-agentic-rollout --agent-command`; the runtime supplies every `RELAX_*` variable. The default audit path is `RELAX_SESSION_IO_DIR/agentspark-audit.json` and is removed with that runtime directory. To retain evidence, provide a distinct private path for each invocation through `--audit-output`; never reuse one global output filename.

SessionShard actors use Ray's `SPREAD` placement and a node-local launcher. A shared callback hostname must therefore not be assumed. `--callback-host-map` reads a JSON object such as `{"worker-hostname": "agentspark-reachable.example.invalid"}` and selects the exact `socket.gethostname()` entry. Missing entries fail before invoking AgentSpark. Values contain only the explicit routable host, without a scheme or port; the configured callback URL retains its scheme, prefix and port. With both bind port and advertised port set to `0`, each Session advertises its actual allocated listener port. All worker routes and those dynamic ports must be reachable from AgentSpark; no IP, proxy or firewall configuration is inferred. A fixed callback URL can be used without a map only for a deployment that explicitly supplies the correct host to every process.

Production forwarding pins the complete configured template kwargs, including K3 `thinking`, and rejects unknown template controls. Valid upstream false/default thinking aliases cannot override the selected training mode. Sampling requests record the configured temperature (frozen default `0.8`) and unfiltered `top_p=1, top_k=-1`; the native Session actually obtains sampling values from the launch arguments, so those arguments must match. Raw AgentSpark requests and actual forwarded requests remain separate audit evidence. Reasoning and recognized omitted narration are restored only inside the complete fixed prefix validated against the frozen formal Miles reconciliation policy. Ordered unique tool-call semantics permit call-ID rewrites, and appended tool-result IDs are mapped back to the stored IDs. A changed/moved prefix, changed tools, or ambiguous tool correlation preserves the incoming history for a new segment; it never borrows reasoning from another branch. The exact frozen template helper SHA-256 is recorded in callback history evidence. The standalone audit path does not enable these production transformations.

The adapter selects each consecutive segment's terminal canonical context, including abandoned branches and resets. `--append-roles` defaults to `tool`, as in the source Miles policy; set it explicitly if the frozen recipe permits other observation roles. Each successful callback response must have one unique native canonical state identity. Repeated indistinguishable generations fail closed. `metadata.agentspark_training` records every segment's `owned_resp_state_hashes` and the invocation-wide `all_resp_state_hashes`; they partition the generations exactly once. The production converter must validate these identities against native `agentic_trace.turns` and mask shared ancestors before computing token denominators. The output contains no reconstructed token IDs or logprobs. Native Relax performs the final committed-Forest check after process success.

Scoring runs once on the final active segment's real callback trajectory, with full terminal history and complete Send events including trailing snapshots. All exports share the verified reward evidence; the terminal anchor carries the actual base reward and earlier segments carry the source policy's explicit sparse `0.0`. These zeros are emitted only after successful scoring. Errors, cancellation, incomplete callbacks, ambiguous ownership, invalid rewards, or an existing output file cause nonzero exit without publishing training output. On success, the adapter creates private JSONL atomically and exits `0`. Eval emits only the terminal representative. Built-in physical-row passrate is unsuitable for counting these multi-context logical Sessions.

This entrypoint must be paired with the V21 sample converter, token-credit policy and loss configuration from the reviewed training recipe. Running the command alone does not prove correct group normalization, remote capacity/cleanup, a live model/tool loop, or an optimizer step. CPU tests use explicit controlled backend/RM fixtures and do not establish those deployment results.

## Prepared input contract

The adapter reads the native Relax input JSON and requires an explicit `metadata.agentspark` envelope. The source dataset converter must produce the prepared business inputs, prompt overrides, resource mappings and Replay history. The adapter does not infer them from the native top-level `messages`, flatten conversations, truncate text, or create placeholder history.

A minimal input is:

```json
{
  "messages": [],
  "metadata": {
    "rubric": "source scoring metadata is preserved in the private audit",
    "agentspark": {
      "schema_version": 1,
      "inputs": [{"type": "TEXT", "info": {"text": "Prepared business input"}}],
      "tools": ["Search"],
      "prompt_spec": {
        "system_prompt": "An explicit source-provided system prompt",
        "user_prompt": "An explicit source-provided user prompt"
      }
    }
  }
}
```

Allowed prepared inputs are `TEXT`, `NOTE`, `POI`, `IMAGE`, and `IMAGES`. Images require an `info` **list**, in original order, with one of `image_url` or `file_id` per image and optional positive integer `width`/`height`. Wire shape validation does not download images. The optional private media path below reads, validates and pins local image bytes; it never fetches a remote source. All TEXT inputs together must fit **1900 UTF-8 bytes**, the conservative Miles preparation limit. Longer source text must be split by the source converter, retaining its prefix in the explicit prompt; this adapter rejects it instead of risking upstream truncation.

Tools must be nonempty, unique names without surrounding whitespace. Optional fields are `bound_user_id`, `tool_versions`, `capabilities`, `skills_name_list`, and `replay_spec`. For `capabilities.skill_loader == "1"`, an explicit nonempty skills list requires those names in a `<AvailableSkills>...</AvailableSkills>` system-prompt catalog. This validates the prepared envelope only; it does not establish semantic equivalence of Loader and no-Loader training tasks.

Replay requires the complete prepared messages and all three resource maps:

```json
{
  "messages": [{"role": "user", "content": "Prepared prior context"}],
  "attributes": {
    "ark.idindex.mapping": {},
    "ark.idindex.image_urls": {},
    "ark.idindex.reverse_mapping": {}
  }
}
```

Place this object in `metadata.agentspark.replay_spec`. Empty maps above illustrate the schema, not a substitute for actual source resource mappings. The adapter sends `replay_type: restore_only`, requires an explicit `link.up.ack`, exact integer `code: 0`, and a nonempty consistent session ID. Unexpected/error frames or EOF without an ACK fail. Without Replay, it uses `GET /diandian/session`. Send uses exactly one POST, consumes the SSE stream through EOF after `agent.stream.end`, and retains trailing snapshots. Missing end, late error, transport failure or malformed Replay fail without resending tool actions.

## Explicit configuration and standalone CLI

Start from [agentspark-config.template.json](agentspark-config.template.json). **It is a non-runnable template:** replace every angle-bracket placeholder and every `/ABSOLUTE/MOUNTED/...` path. Replace the quoted integer placeholders with JSON integers, not numeric strings. There are no saved endpoints or credentials. Its field names match the strict `AdapterConfig` schema; additional fields are rejected.

| Field | Required deployment information |
| --- | --- |
| `base_url` | AgentSpark **service** HTTP(S) base URL, including any gateway prefix. This is where the adapter sends session lookup, Replay and Send. Do not append `/diandian/session` or `/diandian/chat/sse-send-protocol`; the adapter adds them. |
| `agent`, `source` | The deployed AgentSpark agent name. The implemented protocol currently supports only `source: "eden"`; confirm that the service supports its session/Replay/Send wire contract. |
| `model` | The served model alias carried in AgentSpark's model-callback configuration. It is not a local checkpoint path and does not select Relax's backend weights; those belong to the separate Relax/SGLang configuration. |
| `max_tokens` | An explicit positive integer per-turn generation limit, compatible with the model's context budget. This does not set the full agent-loop limit. |
| `callback_bind_host`, `callback_bind_port` | The address and integer port on which this adapter process listens. Each simultaneously resident invocation needs a usable listener; a fixed shared port cannot be assumed to work for multiple processes. Port `0` is supported for local fixture discovery, not evidence of remotely reachable routing. |
| `callback_public_base_url` | The HTTP(S) callback base that **AgentSpark can reach**, including any routed prefix, without `/v1/chat/completions`. The adapter appends that path and serves temporary image routes below the same base. This can differ from the local bind address when a verified proxy forwards to it. |
| `media_allowed_roots` | Explicit absolute directories mounted on every node that can own an adapter invocation. Use the actual deployment paths for the three authorized image roots; the template contains placeholders only. Runtime checks resolve symlinks and revalidate bytes against each row's private manifest. |

The network path is adapter → AgentSpark service, AgentSpark → callback/image routes, and callback → the native `RELAX_BASE_URL`. Verify DNS, routes, scheme/port restrictions, proxy prefixes, request/body limits and idle timeouts on these paths. A loopback fixture does not establish remote reachability. The listener itself is plain HTTP; advertising HTTPS requires an explicitly configured TLS-terminating proxy. URLs cannot embed credentials, query strings or fragments, and redirects are not followed.

The template sets `max_callback_bytes` to **67,108,864 (64 MiB)** as an explicit acceptance starting point. The validated dataset has initial image payloads exceeding both 16 MiB and 32 MiB before text/tool/history overhead. This setting is not a proven upper bound for complete or later callbacks: measure actual forwarded bodies and configure every proxy consistently. Images are never silently resized or removed to meet it. See [MODEL_VALIDATION.md](MODEL_VALIDATION.md) for current measured evidence.

`connect_timeout_s` covers connection/pool acquisition. `read_timeout_s: null` deliberately leaves response reads and writes unbounded in the current HTTP clients, so a first model request can remain open at a Group barrier. Verify AgentSpark's own model-client deadline and all intermediary idle deadlines separately. `cleanup_timeout_s` bounds local callback draining and an injected cleanup hook; it is not the remote task deadline, a first-request-barrier timeout, or proof that remote execution was cancelled.

Authentication has two distinct boundaries. The model callback requires `Authorization: Bearer <RELAX_SESSION_ID>`; the adapter obtains this credential from the native invocation environment and puts it in AgentSpark's `llm_config`, never in this JSON file or the audit. Temporary image GETs use random per-invocation capability paths because the image consumer may not send that bearer header. By contrast, the stock standalone client currently adds **no AgentSpark-service Authorization header**, and `AdapterConfig` has no service API-key, cookie or client-certificate fields. A deployment requiring such authentication needs its exact authentication contract and an explicit adapter change; do not invent config fields or place credentials in URLs. The library accepts caller-owned HTTP clients, but the CLI does not configure arbitrary authentication from this template.

After filling a private copy, validate only its schema locally, without contacting a service:

```bash
python - /absolute/path/filled-agentspark-config.json <<'PY_CONFIG'
import sys
from pathlib import Path
from examples.agentspark_agentic.media import strict_json_loads
from examples.agentspark_agentic.protocol import AdapterConfig

AdapterConfig.from_payload(strict_json_loads(Path(sys.argv[1]).read_bytes()))
sys.stdout.write("Config schema valid; deployment and training readiness remain unverified.\n")
PY_CONFIG
```

The unchanged template intentionally fails validation. A filled configuration passing this check proves only field/URL/value shape; it does not check mounted roots, remote capacity, live parsers or network reachability. Even a successful live protocol audit still exits **3**, keeps `training_ready: false`, and writes no `RELAX_OUTPUT_JSON`. Do not attach it to a full-dataset training launch: use the separate [production training recipe](TRAINING.md), which implements V21 reward/token credit and native context selection/export. Remote cleanup/capacity and real model/tool-loop acceptance still require deployment evidence. Do not substitute a constant reward or change the exit code to manufacture training success.

The normal Relax invocation environment supplies `RELAX_BASE_URL`, `RELAX_SESSION_ID`, `RELAX_GROUP_ID`, `RELAX_ROLLOUT_MODE` (`train` or `eval`), `RELAX_SESSION_IO_DIR`, `RELAX_INPUT_JSON`, and `RELAX_OUTPUT_JSON`. `RELAX_SESSION_ID` is a bearer credential, not a public business identifier. Virtual user/request IDs are hashes of the session credential and invocation directory. A new Session or invocation directory produces a different identity; manually reusing both values reuses that identity and must not be treated as a safe retry.

```bash
python -m examples.agentspark_agentic.client \
  --config /absolute/path/agentspark-config.json \
  --audit-output /absolute/path/new-private-audit.json
```

This command contacts the configured services. For **offline verification**, run the tests below instead. No production endpoint, model weight, GPU, or credential is needed for them.

The audit destination must be new and must differ from `RELAX_OUTPUT_JSON` after resolving symlinks. The writer uses a private mode-0600 temporary file and atomic creation without replacement. Existing audit or training outputs are never overwritten. Exit 3 means protocol evidence was collected but training remains unavailable; exit 1 means a completed audit reports failure; exit 2 means validation/publication/lifecycle setup could not produce an audit. Error output contains fixed codes, not raw remote exception details.

## Callback evidence and lifecycle

The Chat callback accepts only the explicit advertised `/v1/chat/completions` route and the exact Relax bearer credential. Without a private media plan it forwards the original request body bytes. With a plan it restores only authorized image occurrences and tool-history stubs, then serializes the actual forwarded request; reasoning, tools and sampling fields are otherwise retained. Response bytes remain unchanged. Records contain `raw_request` and its byte hash, actual forwarded `request`, `request_sha256`, `forwarded_request_canonical_sha256`, and canonical assistant/response evidence. Stage 3 normalizes **actual forwarded** requests; a media audit without this identity, or with a mismatched canonical hash, is rejected. It understands Relax's complete JSON replies and its **Buffered SSE** form (one complete terminal choice, then `[DONE]`). It does not reconstruct a partially streamed assistant. SSE-only tool-call indexes are removed from the candidate message because they are absent from Relax's committed canonical message; the response hash still describes the original bytes.

A candidate is produced only after the callback server and handlers close, all started requests were captured, and the full observed conversation is strictly append-only with unchanged tools/template kwargs. Concurrent calls, edited histories, forks, incomplete streams, or unrecorded requests fail. This proves only observed payload shape. It does **not** verify SessionForest committed-context hashes, merged template identity, token IDs/logprobs, media resolution, or training credit. There is no implicit fallback export for complex histories.

The private audit preserves the original source input and scoring metadata, group/mode/model identifiers, callback evidence, and business SSE payloads including trailing snapshots. These files contain business/model content; manage them as private artifacts. Credential checks reject known sensitive field names, JSON-encoded credential fields in tool arguments/observations, and any occurrence of the Relax session credential. Headers and raw `llm_config` are never persisted. Remote error bodies are always omitted because arbitrary error text can expose credentials or paths. Non-error SSE event labels and payloads are retained after credential checks. A credential-bearing business event is omitted, represented by its hash and failure marker, and the remaining stream is still drained. These checks cannot identify every unrelated secret embedded in arbitrary prose; they do not make private business content safe for public logging.

HTTP connections and the callback server are owned by one invocation and closed on completion, failure or cancellation. Injected HTTP clients remain owned by their caller. SIGINT/SIGTERM set a stop event, cancel local transport work, run bounded cleanup, and produce a failed audit. Direct cancellation of the library coroutine performs cleanup and re-raises `CancelledError`; that caller must handle its own audit/lifecycle policy.

There is **no verified AgentSpark cancellation API** in the inspected source. Closing a local stream does not prove remote tool execution stopped. `run_audit(..., cleanup_hook=...)` accepts an explicit trusted async operation supplied by a deployment owner, with remote session/request/user IDs and a reason; its completion remains `remote_state_unverified`. No guessed cancel URL is sent. Missing, failed or timed-out cleanup is recorded as unknown. A failed Replay after POST also records remote state as unknown, even if no Send occurred.

The group-first-request barrier still requires remote capacity for every admitted sample in a Relax group. A remote concurrency limit below the admitted group width can deadlock first callbacks. This example does not bypass that barrier, tune dynamic batching/custom advantage, or claim capacity validation. Read timeout defaults to unbounded to avoid confusing barrier wait with model failure; an outer cancellation/deployment deadline and a verified remote cleanup contract are required before deployment.

## Private local media plan

`metadata.agentspark_media` is a local control-plane object, separate from the AgentSpark wire envelope:

```json
{
  "schema_version": 1,
  "assets": [{
    "ref": "agentspark-asset:SHA256_OF_ORIGINAL_BYTES",
    "path": "/absolute/allowed/image.png",
    "sha256": "SHA256_OF_ORIGINAL_BYTES",
    "mime_type": "image/png",
    "byte_length": 1234,
    "width": 640,
    "height": 480
  }],
  "replay_bindings": [{
    "marker": "[[MILES_REPLAY_IMAGE:0123456789abcdef0123456789abcdef:0000]]",
    "source": "agentspark-asset:SHA256_OF_ORIGINAL_BYTES",
    "role": "user"
  }],
  "seed_refs": []
}
```

Use `AssetStore(allowed_roots).freeze(path)` and `.build(replay_bindings, seed_refs)` to construct this object from real images; the illustrative hashes above are not valid manifests. Seed references preserve current-image order and duplicates. Replay markers use a single namespace and consecutive indexes. Tool images additionally require the exact source `tool_history` object (`tool_call_id`, `name`, `arguments_sha256`, `content`). Its argument hash follows Miles' source JSON-object hashing, including float representation; it is separate from Relax's native context canonicalization. Duplicate JSON keys are rejected.

The runtime configuration must independently supply `media_allowed_roots`, a list of authorized absolute mounted directories. Paths and symlinks are resolved before checking these roots. Every referenced source is read, decoded and SHA256-checked against its manifest, then its original bytes are frozen in memory for the invocation. A changed, missing, invalid, out-of-root or remote source fails. The optional offline verified catalog accelerates dataset conversion only: it is caller-provided evidence, requires every selected source to be present, and does not bypass runtime revalidation. `AssetStore.new_scope()` shares the validated catalog while keeping each row's asset manifest separate.

Once the callback server is listening, exact internal asset references in prepared input/resource-map values become URLs under the **configured, real** callback public base and a random per-invocation content-addressed route. This route serves only frozen manifest bytes, has no filesystem/URL fetch API, and closes with the callback server. Private bindings and filesystem paths are never added to Replay or Send. Unsupported wire positions containing an unresolved internal reference fail. The model receives data URIs encoding the same frozen bytes, so its input no longer depends on the temporary HTTP route.

On the first callback every historical marker must occur exactly once, inside the registered user/tool content, with the registered role. The adapter records the full message hash and an identity that excludes registered image occurrences. Later callbacks can move/repeat/remove image occurrences within that message identity, reorder complete history messages, or remove an entire historical branch. Changed non-image text, role or tool identity cannot acquire an old marker's image authority. Assistant text and tool arguments never expand markers. A compressed tool stub is restored only for its unique matching assistant call ID/name/argument hash and original message identity. Deleted image occurrences are not silently restored. Undeclared dynamic remote/local image sources and AgentSpark URL rewrites are rejected pending a verified source-resolution contract.

The default `max_callback_bytes` is **16 MiB for each raw/forwarded request and response**. Base64 increases image bytes by roughly one third, so a source image below the per-image 16 MiB limit can exceed the callback-body limit after restoration. Configure a sufficient explicit callback budget for the intended context; the adapter returns `callback_forwarded_request_too_large` rather than resizing or discarding images. At most 30 image occurrences are admitted per restored request, consistent with the source recipe's bound; excess occurrences fail explicitly.

Loopback tests verify byte identity and routing, not AgentSpark deployment acceptance. Real AgentSpark support for the selected URL scheme/port, callback reachability and its handling of current image URLs still require live acceptance. No deployment, model generation, reward or token-credit readiness is implied.

## Offline source conversion

The source converter uses the explicitly selected Miles checkout's preparation helpers and the selected run's config/profile. It does not import Miles at training runtime or invent missing prompt/tool defaults:

```bash
python -m examples.agentspark_agentic.prepare \
  --input /absolute/path/prepared.jsonl \
  --output /absolute/path/new-relax-input.jsonl \
  --source-config /absolute/path/run/stage/config.json \
  --miles-source /absolute/path/miles \
  --profile-dir /absolute/path/run/miles/examples/async_dots_rl/.../think_online \
  --allowed-root /absolute/mounted/image/root \
  --verified-media-manifest /absolute/path/all-media-manifest.jsonl \
  --rejections /absolute/path/new-rejections.jsonl \
  --report /absolute/path/new-conversion-report.json
```

Repeat `--allowed-root` for additional authorized mounts. Omit the verified manifest to validate local bytes directly. The output keeps `messages` as a list, retains source metadata and labels, and adds explicit `agentspark`/`agentspark_media` metadata; use Relax's `--input-key messages` without applying a dataset chat template. Review the report and explicit rejected rows; conversion does not discard unsupported rows silently. Exit 3 indicates explicit rejections, and `training_ready` remains false. No remote image is downloaded; no AgentSpark service, model weights or GPU is used by this conversion.

Image **references in the resource registry** and **pixels required by explicit model-visible image parts** have different contracts. The converter preserves Replay resource URL mappings using the source recipe's explicit `agent.image_allowed_hosts`; the Miles helper can upgrade an allowed historical HTTP image URL to HTTPS. Note covers and avatars are not automatically downloaded or added as visual inputs. Keeping these remote references does not verify their bytes, availability, or later tool behavior. Local registry image URLs still require authorized, verified assets so the runtime can publish them without exposing filesystem paths.

Current image attachments and explicit history/snapshot image parts require local verified pixels and retain each occurrence. A remote URL in one of these positions is rejected as `unresolved_remote_media` during offline conversion; another nearby local image cannot substitute for it. The manifest covers only the explicitly verified local assets. Empty user history and duplicate tool-argument JSON keys are also reported as rejected source rows, with line number, source-row hash, query-ID hash and fixed failure code; same-value and conflicting duplicate keys have separate codes and neither is silently repaired.

## 128 B300 cluster preflight

For the 16-node, eight-GPU-per-node target, use [CLUSTER_VALIDATION.md](CLUSTER_VALIDATION.md) and `cluster_preflight.py`. The script checks an explicitly selected existing cluster and optionally exercises basic NCCL collectives. Its local/dry-run modes do not connect to Ray or use GPUs. Every report retains `training_ready: false`; a successful hardware/file probe does not establish live V21 training acceptance. The source-backed topology and remaining work are recorded in [training-plan-128b300.template.json](training-plan-128b300.template.json), which is a non-executable plan.

## CPU validation

`test_production.py` covers source-style reasoning restoration, pinned controls, explicit host maps, complete local HTTP callback/SSE loops, real native K3 context exports with a controlled backend, scoring identity/failure/cancellation, and atomic non-overwriting publication. These tests exercise no remote AgentSpark/RM service or GPU.

From the Relax repository root:

```bash
CUDA_VISIBLE_DEVICES="" /opt/sglang/bin/python -m pytest \
  tests/cpu/examples/agentspark_agentic -q
```

The foundational Stage 2 tests use HTTP mocks and short-lived loopback aiohttp servers to exercise the actual byte proxy, JSON/Buffered SSE tool history, Replay/GetSession paths, late snapshots/errors, one-shot Send failure, cleanup/cancellation, credential boundaries, and safe audit publication. These protocol fixtures do not prove model correctness, platform reachability, V21 reward equivalence, or RL training acceptance.

Stage 3 native-export tests additionally use the selected local K3 tokenizer/processor. Set `KIMI_K3_SOURCE_DIR` explicitly to enable them; a missing source produces a skip, not validation. Current source identity, commands, and acceptance evidence are recorded in [MODEL_VALIDATION.md](MODEL_VALIDATION.md).

```bash
KIMI_K3_SOURCE_DIR=/absolute/path/to/Kimi-K3 \
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/sglang/bin/python -m pytest \
  tests/cpu/examples/agentspark_agentic/test_contexts.py \
  tests/cpu/examples/agentspark_agentic/test_native_exports.py -q
```

## Stage 3: offline contexts and native export validation

`contexts.py` analyzes a completed Stage 2 capture without contacting services. It applies Relax's own Chat canonicalization, including system-message merging, tool-argument normalization and image content shapes. The original audit is retained unchanged, including any Stage 2 failure status. Compiler `apply_chat_template_kwargs` defaults must be supplied explicitly; request overrides are merged onto them for every context identity. The `model` field remains a configured/request label. Neither that label nor caller-supplied defaults verifies deployed weights or model identity.

Only complete transport and callback evidence is accepted: exactly one Send, completed remote stream with end and EOF and no error, successful local close, successful callback records, exact contiguous request indexes, and no concurrency. Stage 2 failures may be analyzed only for `callback_tools_or_template_changed`, `callback_history_not_linear`, or `callback_observation_not_linear`. The combined incomplete/concurrent failure is rejected. Unknown remote cleanup remains a pending gate.

Analysis preserves observed append, fork and reset structure, changed tools/template parameters, and reasoning deletion. It does not restore deleted reasoning, move old media back into new histories, or choose branches for training. Each image occurrence keeps its callback/message/part coordinates, including duplicate appearances. `reference_sha256` hashes the locator string, **not downloaded image bytes**; `content_verified` remains false. Image content validation requires separate processor/backend evidence.

Provide a JSON object containing the exact compiler defaults (use `{}` only when that is the actual configuration). An optional selection JSON explicitly names each selected response and its intent:

```json
[
  {"name": "main", "record_index": 2, "intent": "Keep the final main-agent context"},
  {"name": "branch", "record_index": 4, "intent": "Keep this intentional branch"}
]
```

Without selection, the analysis records candidates and selects nothing. Names must be unique. Canonically identical response states are reported as aliases; selecting an ambiguous repeated generation is rejected because payloads cannot establish which generation tokens the native Forest retained. Eval currently allows one representative selection until a separate aggregation policy is reviewed. More than one selected context requires reviewed advantage handling and dynamic batching. The production recipe handles generation ownership and V21 token credit in its native-Sample converter; this standalone analysis does not apply that policy.

```bash
python -m examples.agentspark_agentic.contexts \
  --audit /absolute/path/stage2-audit.json \
  --compiler-defaults /absolute/path/compiler-defaults.json \
  --selection /absolute/path/selection.json \
  --audit-output /absolute/path/new-stage3-analysis.json \
  --relax-output /absolute/path/reserved-training-output.json
```

This command is fully offline. It writes a new private analysis artifact, exits **3**, and never writes the reserved training output. `records` contains canonical named candidates, not authorized training exports. The report keeps `training_ready=false`, `committed_state_validation=not_performed`, and `media_content_validation=not_performed`. There is no token/logprob/reward reconstruction from captured text. Top-level request options retained as evidence are not a promise that Relax applied them; only fields consumed by the native API participate in its behavior.

For an in-process integration test, `validate_native_exports(report, shard=..., session=..., invocation_id=...)` accepts actual native SessionShard/SessionRecord objects with an existing committed Forest. It binds the original audit to `sha256(session_id + NUL + invocation_id)`, checks native group/mode and compiler defaults, parses candidates through `SessionOutput.from_records`, then calls the real native export builder. The invocation ID is the caller-owned `RELAX_SESSION_IO_DIR` value; no Session credential is stored in the analysis. The function does not construct or replay a Forest from the audit and exposes no remote endpoint.

Native validation returns the original `SessionExportTransport` for caller inspection without publishing it or changing the report's training gate. All physical exports retain the Forest's logical `Sample.index` and `group_index`. Adapter provenance is confined to `metadata.agentspark_context`; native `agentic_trace`, tools/template metadata and dataset `train_metadata` remain owned by Relax. The adapter does not add a reward or custom advantage value.

The Stage 3 CPU tests exercise actual native request normalization, observation/response commits and explicit export against controlled backend responses. They check branch selection, media lineage, identity rejection and original backend token/logprob retention. Fixed backend responses are test data, not real model generations. Real K3 compiler/media checks, where enabled by `KIMI_K3_SOURCE_DIR`, validate encoding and state transport only; no inference weights, optimizer or remote AgentSpark task runs in this stage.

The complete source-pinned V21 objective, 128 B300 full-data launcher and configuration commands are documented in [TRAINING.md](TRAINING.md).
