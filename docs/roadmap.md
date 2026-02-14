# Roadmap

Tracking progress from first compile to public launch.

---

## Current State

**Project compiles cleanly** (26 warnings, 0 errors). All core abstractions have real implementations backed by Rig framework integration, direct HTTP LLM calls, and database queries. The full message-in → LLM → response-out pipeline is wired end-to-end. The system starts messaging adapters, routes inbound messages to agent channels via binding resolution, runs real Rig agent loops for channels/branches/workers, and routes outbound responses back through the messaging layer.

### What's Fully Implemented

- **Project structure** — all modules declared, module root pattern (`src/memory.rs` not `mod.rs`)
- **Error hierarchy** — thiserror domain enums wrapped by top-level `Error` with `#[from]`
- **Config** — hierarchical TOML with `Config`, `AgentConfig`, `ResolvedAgentConfig`, `Binding`, `MessagingConfig`
- **Multi-agent** — per-agent database isolation, `Agent` struct bundles all dependencies
- **Database connections** — SQLite + LanceDB + redb per-agent, migrations for all tables
- **LLM** — `SpacebotModel` implements Rig's `CompletionModel`, routes through `LlmManager` via HTTP
- **Model routing** — `RoutingConfig` with process-type defaults, task overrides, fallback chains
- **Memory** — full stack: types, SQLite store (CRUD + graph), LanceDB (embeddings + vector + FTS), fastembed, hybrid search (RRF fusion)
- **Identity** — `Identity` struct loads SOUL.md/IDENTITY.md/USER.md, `Prompts` with fallback chain
- **Agent loops** — all three process types run real Rig loops:
  - **Channel** — per-turn tool registration, status injection, `max_turns(5)`
  - **Branch** — history fork, `max_turns(10)`, memory tools, result injection
  - **Worker** — fresh history, per-worker ToolServer, `max_turns(50)`, interactive mode
- **Compactor** — **FULLY IMPLEMENTED** — tiered thresholds (80%/85%/95%), LLM summarization, pre-compaction archiving, emergency truncation
- **Cortex** — signal buffering, bulletin generation on 60min interval (complete); consolidation stubbed
- **StatusBlock** — event-driven updates, renders to context string
- **Hooks** — `SpacebotHook` with tool call/result events, leak detection; `CortexHook` for observation
- **Messaging** — `Messaging` trait with RPITIT + companion, `MessagingManager` with fan-in/routing
- **Discord adapter** — full serenity implementation (message handling, streaming via edit, typing indicators)
- **Tools** — 16 tools implement Rig's `Tool` trait with real logic (reply, branch, spawn_worker, route, cancel, skip, react, memory_save, memory_recall, set_status, shell, file, exec, browser, cron, web_search)
- **Conversation persistence** — **FULLY IMPLEMENTED** — `ConversationLogger` with fire-and-forget SQLite writes, compaction archiving
- **Cron** — **FULLY IMPLEMENTED** — scheduler with timers, active hours, circuit breaker (3 failures → disable), creates real channels
- **Message routing** — full event loop with binding resolution, channel lifecycle, outbound routing

### What's Stubbed or Missing

- **Cortex consolidation** — `run_consolidation()` just logs, not implemented
- **Memory maintenance** — decay + prune implemented, merge stubbed (`merge_similar_memories()` is a no-op)
- **RouteTool follow-ups** — validates worker exists and returns success, but never delivers the message — `input_tx` dropped immediately after worker spawn
- **Streaming** — `SpacebotModel.stream()` returns error, not implemented
- **Telegram adapter** — empty struct, no `Messaging` impl
- **Webhook adapter** — empty struct, no `Messaging` impl
- **Secrets store** — empty struct, no redb integration
- **Settings store** — empty struct, no redb integration
- **CortexHook** — all hook methods are trace-only passthrough, no observation logic
- **Cortex observe()** — only extracts 3 event types, hardcodes `memory_type: "unknown"` and `importance: 0.5`

### Known Runtime Bugs (Must Fix Before MVP)

1. **Will panic** — `search.rs:273` — `partial_cmp().unwrap()` on f64 sort, NaN would crash
2. **Will panic** — `embedding.rs:57` — `EmbeddingModel::default()` uses `expect()` in a `Default` impl callable outside init
3. **Arrow type mismatch** — `lance.rs` casts `_distance`/`_score` as `Float64Type` but LanceDB returns `Float32`, will panic on any memory search
4. **Arrow version mismatch** — `arrow = "54"` vs `arrow-array`/`arrow-schema` at `"57.3.0"` in Cargo.toml
5. **RouteTool no-op** — returns `routed: true` to the LLM without actually sending the message
6. **Silent error swallows** — `memory_save.rs:208` drops association errors, `memory_recall.rs:138` drops access recording errors, `discord.rs:446` can silently lose inbound messages
7. **memory_recall** — `memory_type` filter arg accepted but never applied to search
8. **cron tool** — not wired into `add_channel_tools()` factory

---

## Development Timeline

Chronological development history based on git commits:

### 2026-02-11 — Foundation & Core Architecture

- `5629b5f` — Initial project scaffold and repository setup
- `aae3219` — Project structure with module root pattern (no `mod.rs` files)
- `29780e4` — Documentation refinement and cortex prompt design
- `ad5f826` — Memory management with LanceDB integration, SQL migrations, `MemorySearch` struct
- `5b08674` — Agent documentation and routing configuration design
- `08faaa0` — Multi-agent config structure with `DefaultsConfig`, agent resolution, identity handling
- `cb86a0a` — LLM routing with `RoutingConfig`, rate limit management, Discord dependencies
- `a3d60de` — Messaging architecture with `MessagingManager`, Discord adapter foundation
- `a7c5dc7` — Discord messaging adapter full implementation with serenity
- `b75462e` — Discord setup documentation and tools architecture docs
- `a0a0c32` — Channel state management, tool server integration, routing improvements
- `990cbd1` — Discord channel filtering, config enhancements
- `76fa2a1` — Message history backfill, worker summaries
- `43e9cca` — Enhanced Channel structure with conversation context
- `98fd965` — **Conversation persistence** — `ConversationLogger` with SQLite storage

### 2026-02-12 — Workers, Compaction, Cron & Tools

- `52ae37d` — **Compaction** — `Compactor` with tiered thresholds, archiving, emergency truncation
- `11c15a4` — Discord threading support, attachment handling, `ReplyTool` enhancements
- `2b83c08` — **Browser tool** — chromiumoxide integration, element ref system, screenshots
- `64f2d1e` — Skills framework foundation
- `bcd57f2` — **Cron scheduler** — timer management, circuit breaker
- `a2ab16f` — **Cron tool** — CRUD operations, delivery targets, active hours
- `a9068c1` — Config hot-reloading with `arc-swap`, file watching via `notify`
- `6ee1d26` — Exit strategy improvements, cron shutdown cleanup
- `65427f4` — **Skip and react tools** — message suppression and emoji reactions
- `17930dd` — Tool server documentation refinements
- `f25ce96` — Channel tool registration documentation
- `26e42ec` — **Memory ingestion** — bulk import from text files, chunking, auto-save intervals

---

## ~~Phase 1: Migrations and LanceDB~~ Done

- [x] SQLite migrations for all tables (memories, associations, conversations, cron_jobs)
- [x] Inline DDL removed from `memory/store.rs`, `conversation/history.rs`, `cron/store.rs`
- [x] `memory/lance.rs` — LanceDB table with Arrow schema, embedding insert, vector search (cosine), FTS (Tantivy), index creation
- [x] Embedding generation wired into memory save flow
- [x] Vector + FTS results connected into hybrid search via `MemorySearch` struct
- [x] `MemorySearch` bundles store + lance + embedder

---

## ~~Phase 2: Wire Tools to Rig~~ Done

- [x] All 18 tools implement Rig's `Tool` trait
- [x] `AgentDeps.tool_server` uses `rig::tool::server::ToolServerHandle` directly
- [x] `PromptHook<M>` on `SpacebotHook` and `CortexHook`
- [x] `agent_id: AgentId` threaded through SpacebotHook, SetStatusTool, all ProcessEvent variants
- [x] `MemorySaveTool` — `channel_id` field wired into `Memory::with_channel_id()`
- [x] `ReplyTool` — replaced with `mpsc::Sender<OutboundResponse>` for ToolServer compatibility
- [x] `EmbeddingModel` — shares model via `Arc` instead of creating new instance per call

---

## ~~Phase 3: System Prompts, Identity, and Multi-Agent~~ Done

- [x] `prompts/` directory with all 5 prompt files (CHANNEL.md, BRANCH.md, WORKER.md, COMPACTOR.md, CORTEX.md)
- [x] `identity/files.rs` — `Identity` struct, `Prompts` struct with workspace-aware fallback loading
- [x] `conversation/context.rs` — `build_channel_context()`, `build_branch_context()`, `build_worker_context()`
- [x] `conversation/history.rs` — `ConversationLogger` with fire-and-forget persistence
- [x] Multi-agent config — hierarchical TOML with `AgentConfig`, `DefaultsConfig`, `ResolvedAgentConfig`, `Binding`, `MessagingConfig`
- [x] Per-agent database isolation
- [x] `Agent` struct bundles db + deps + identity + prompts
- [x] `main.rs` per-agent initialization loop
- [x] Prompt resolution fallback chain

---

## ~~Phase 4: The Channel (MVP Core)~~ Done

- [x] `RoutingConfig` — process-type defaults, task-type overrides, fallback chains
- [x] Fallback logic in `SpacebotModel` — retry with next model on 429/502/503/504
- [x] Rate limit tracking — deprioritize 429'd models for configurable cooldown
- [x] Shared ToolServer for channel/branch tools
- [x] Per-worker ToolServer factory
- [x] Real Rig agent loop with `AgentBuilder`
- [x] Status block injection — prepend rendered status to each prompt call
- [x] Identity injection — prepend identity context to system prompt
- [x] Discord adapter — full serenity implementation
- [x] `ChannelState` bundle — shared state for channel tools

---

## ~~Phase 5: Branches and Workers~~ Done

- [x] Branch: real Rig agent loop with `max_turns(10)`, shared ToolServer, history fork
- [x] Branch result injection — conclusion returned via `ProcessEvent::BranchResult`
- [x] Branch concurrency limit enforcement
- [x] Worker: real Rig agent loop with `max_turns(50)`, per-worker ToolServer, fresh history
- [x] Worker state machine with transition validation
- [x] Worker status reporting via set_status tool
- [x] Interactive worker follow-up loop on `input_rx`
- [x] `MaxTurnsError` / `PromptCancelled` handling with partial result extraction
- [x] `BranchTool` creates and spawns Branch processes
- [x] `SpawnWorkerTool` creates and spawns Worker processes
- [x] `CancelTool` aborts branches and removes workers

---

## ~~Phase 6: Messaging Routing~~ Done

- [x] Binding resolver — `Binding::matches()` and `resolve_agent_for_message()`
- [x] Message routing loop in `main.rs`
- [x] Channel lifecycle — create on first message, spawn `Channel::run()` as tokio task
- [x] Outbound response routing — per-channel task reads from `response_rx`
- [x] Event bus — `broadcast::Sender<ProcessEvent>` in `AgentDeps`
- [x] `MessagingManager` Arc-wrapped for shared outbound access
- [x] Graceful shutdown

---

## ~~Phase 7: Compaction and Persistence~~ Done

- [x] **Compactor** — tiered thresholds (80%/85%/95%), LLM-based summarization, emergency truncation
- [x] **Conversation persistence** — fire-and-forget SQLite writes via `ConversationLogger`
- [x] Load conversation history from DB on channel creation (resume across restarts)
- [x] Compaction summaries persisted and loaded into context

---

## Phase 8: Critical Bug Fixes

Fix runtime panics and functional gaps before launch.

- [ ] Fix `search.rs:273` — replace `partial_cmp().unwrap()` with `unwrap_or(Ordering::Equal)` or `total_cmp()`
- [ ] Fix `embedding.rs:57` — remove `Default` impl or make it fallible
- [ ] Fix Arrow type mismatch — `lance.rs` casts `_distance`/`_score` as `Float64Type` but LanceDB returns `Float32`
- [ ] Fix arrow version mismatch in `Cargo.toml` (align or drop `arrow` meta-crate)
- [ ] Fix RouteTool — store worker `input_tx` in `ChannelState` so `route` actually delivers messages
- [ ] Fix silent error swallows — add `tracing::warn` on association creation, access recording, and inbound message send failures
- [ ] Wire `memory_type` filter through to `SearchConfig` in `memory_recall`
- [ ] Wire `CronTool` into `add_channel_tools()` factory
- [ ] Add workspace path guards to `file` and `shell` tools
- [ ] Test: send a message via Discord, get a real LLM response back
- [ ] Test: trigger a branch (message that requires memory recall), verify branch result incorporation
- [ ] Test: trigger a worker (ask for a file operation), verify worker completion

---

## Phase 9: Cortex Consolidation

- [ ] Implement `run_consolidation()` — memory merging, decay management, graph optimization
- [ ] Cross-channel coherence — shared observations across an agent's conversations
- [ ] Memory graph traversal during recall — walk typed edges (Updates, Contradicts, CausedBy)

---

## Phase 10: Webhook Adapter

HTTP adapter for testing and programmatic access without Discord.

- [ ] Implement WebhookAdapter (axum) — POST endpoint for `InboundMessage`, response routing
- [ ] Optional sync mode (`"wait": true` blocks until agent responds)
- [ ] Test: `curl -X POST` a message, get a response back

---

## Phase 11: Hardening

- [ ] Tool nudging — inject "use your tools" in `SpacebotHook.on_completion_response()` when LLM responds with text in early iterations
- [ ] Expand leak detection — add patterns for AWS keys, Slack tokens, bearer tokens, DB connection strings
- [ ] Outbound HTTP leak scanning — block exfiltration via tool output before it reaches external services
- [ ] CortexHook — implement real observation logic (anomaly detection, consolidation triggers)
- [ ] Cortex observe() — extract signals from all ProcessEvent variants with real values
- [ ] Memory maintenance — implement `merge_similar_memories()` in `maintenance.rs`
- [ ] Secrets store — AES-256-GCM encrypted credentials in redb, `DecryptedSecret` wrapper type
- [ ] Settings store — redb key-value with env > DB > default resolution

---

## Phase 12: Ingestion Pipeline

Drop files in, agent picks them up.

- [ ] File watcher on `agents/{id}/ingest/` directory
- [ ] Chunking — split large files into context-window-safe segments
- [ ] Per-chunk worker using INGESTION.md prompt — recall existing, extract memories, build associations
- [ ] Cleanup — move processed files to `agents/{id}/ingest/done/`
- [ ] OpenClaw migration path — drop MEMORY.md, daily files, and skill configs into ingest folder

---

## ~~Phase 13: OpenClaw Skill Compatibility~~ Done

- [x] Parse OpenClaw skill format (SKILL.md + tool definitions)
- [x] Map skill tools to Spacebot worker tool dispatch
- [x] Skill directory watcher with hot reload
- [x] Instance-level and per-agent skill directories

---

## Phase 14: Telegram Adapter

- [ ] Implement TelegramAdapter via teloxide — long polling mode
- [ ] Message handling, typing indicators, reply threading
- [ ] Bindings support for `chat_id` routing

---

## Phase 15: Launch Prep

Everything needed to ship the video and go public.

- [ ] End-to-end testing — Discord conversation with branching, workers, memory recall, compaction
- [ ] Record demo footage — multi-user Discord interaction showing non-blocking behavior
- [ ] README rewrite — clear setup instructions, architecture overview, quick start for new users
- [ ] GitHub repo public — clean commit history, license, contributing guide
- [ ] Landing page — spacebot.sh with overview, setup instructions, link to hosted version
- [ ] Launch video — 60-90s Twitter video (see docs/launch-script.md)

---

## Phase 16: Hosted Version (spacebot.sh)

One-click hosted Spacebot for people who don't want to self-host.

- [ ] Containerization — Docker image with all dependencies (Rust binary + Chrome for browser tool)
- [ ] Container orchestration — per-user isolated containers with persistent volumes for agent data
- [ ] Onboarding flow — connect Discord/Telegram, set API keys (or use shared keys with usage billing), configure identity
- [ ] Billing — subscription model, usage-based pricing for LLM calls if using shared keys
- [ ] Dashboard — web UI for agent management, memory browsing, conversation history, cron config
- [ ] Monitoring — container health, per-user resource usage, error alerting

---

## Post-Launch

- **Streaming** — implement `SpacebotModel.stream()` with SSE parsing, wire through messaging adapters
- **Agent CLI** — `spacebot agents list/create/delete`, identity template bootstrapping
- **Cross-agent communication** — routing between agents, shared observations
- **Hot reload agent topology** — adding/removing agents without restart
- **Agent templates** — pre-built configurations for common use cases
- **JsonSchema-derived tool definitions** — replace hand-written JSON schemas in all 16 tools with the `JsonSchema` derive that's already on every Args struct
- **Spacedrive integration** — connect agents to terabytes of indexed, content-addressed file data across devices (see [docs/spacedrive.md](spacedrive.md))
