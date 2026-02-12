# OpenCode Workers

OpenCode integration for complex coding tasks. Instead of running a built-in Rig agent loop with shell/file/exec tools, Spacebot can spawn an [OpenCode](https://opencode.ai) subprocess as a worker backend. OpenCode has its own codebase exploration, context management, tool suite, and agentic loop -- all the things you'd want for multi-file coding tasks that a basic shell/file worker can't handle well.

## Why Not Just Use Built-in Workers

Built-in workers are general purpose. They get a system prompt, a task, and shell/file/exec tools. For simple tasks (run a command, edit a file, check a status) they're fine. For complex coding -- refactoring across multiple files, understanding a codebase, running tests and fixing failures iteratively -- they're limited. They don't build codebase context, don't have smart file search, and don't manage their own context window.

OpenCode is purpose-built for this. It reads AGENTS.md files, explores project structure, uses LSP data, manages its own context compaction, and has a sophisticated tool suite tuned for coding. Rather than replicating all of that in Spacebot, we run it as a subprocess and communicate over its HTTP API.

## Architecture

```
Channel
  → spawn_worker { task, worker_type: "opencode", directory: "/path/to/project" }
    → OpenCodeServerPool looks up or spawns a server for that directory
    → OpenCodeWorker creates a session, sends the task as a prompt
    → Subscribes to SSE event stream for real-time progress
    → Tool executions show up as status updates in the channel
    → Permissions/questions route back through the channel
    → On completion, result injected into channel history
```

### Per-Directory Persistent Servers

OpenCode runs as a persistent HTTP server per working directory (`opencode serve --port <port>`). The `OpenCodeServerPool` manages these:

- First task targeting `/path/to/project` spawns a new server
- Subsequent tasks reuse the existing server (new session, same process)
- Servers auto-restart on crash (up to 5 retries)
- Health check polling on startup (`GET /global/health`)
- Pool limit configurable (default 5 concurrent servers)
- All servers killed on Spacebot shutdown

This means the first OpenCode task for a directory has ~5-10s startup overhead. After that, new sessions start instantly.

### Communication Protocol

All communication is over localhost HTTP:

- **Session management** — `POST /session` (create), `POST /session/{id}/abort` (cancel)
- **Prompting** — `POST /session/{id}/prompt_async` (non-blocking, returns immediately)
- **Event stream** — `GET /event` (SSE, real-time updates on tool calls, completion, errors)
- **Permissions** — `POST /permission/{id}/reply` (respond to permission requests)
- **Questions** — `POST /question/{id}/reply` (respond to question tool prompts)

No OpenCode Rust SDK exists. Spacebot implements a thin HTTP client in `src/opencode/server.rs` covering only the endpoints we need.

### SSE Events We Care About

| Event | What It Means |
|-------|---------------|
| `message.part.updated` (tool) | OpenCode is executing a tool — maps to worker status update |
| `message.part.updated` (text) | OpenCode produced text output — captured as result |
| `session.idle` | Session finished — worker completes |
| `session.error` | Session failed — worker reports error |
| `session.status` (retry) | Rate limited, retrying — status update |
| `permission.asked` | OpenCode needs permission for an action |
| `question.asked` | OpenCode's question tool is asking the user something |

Everything else (LSP events, file watcher, TUI, PTY, MCP) is ignored.

## Configuration

```toml
[defaults.opencode]
enabled = true                          # whether OpenCode workers are available
path = "env:OPENCODE_PATH"             # binary path, falls back to "opencode" on PATH
max_servers = 5                         # max concurrent server processes
server_startup_timeout_secs = 30        # health check timeout
max_restart_retries = 5                 # restart attempts before giving up

[defaults.opencode.permissions]
edit = "allow"                          # auto-allow file edits
bash = "allow"                          # auto-allow shell commands
webfetch = "allow"                      # auto-allow web fetches
```

The permission settings are passed to OpenCode via `OPENCODE_CONFIG_CONTENT` env var. With all permissions set to `"allow"`, OpenCode runs headless without prompting for confirmation. If you want the channel to mediate permissions, set them to `"ask"` and the `WorkerPermission` events will surface.

OpenCode's LSP and formatter are disabled in headless mode — they're expensive and unnecessary when OpenCode is running as a backend worker.

## Usage

The `spawn_worker` tool gains two optional parameters when OpenCode is enabled:

```json
{
  "task": "Refactor the auth module to use JWT tokens instead of session cookies",
  "worker_type": "opencode",
  "directory": "/Users/me/projects/myapp",
  "interactive": true
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `worker_type` | No | `"builtin"` (default) or `"opencode"` |
| `directory` | When `worker_type` is `"opencode"` | Working directory for the OpenCode agent |
| `interactive` | No | Whether the worker accepts follow-up messages |

The channel LLM decides when to use OpenCode vs built-in workers based on task complexity. The tool definition dynamically includes the OpenCode parameters only when `opencode.enabled` is true.

## Lifecycle

### Fire-and-forget

```
1. Channel branches to think about user's request
2. Branch decides this is a coding task, spawns OpenCode worker
3. Worker gets OpenCode server for the directory (spawn or reuse)
4. Worker creates session, sends task prompt via async API
5. Worker subscribes to SSE events, processes them in a loop:
   - Tool starts: status update → "running: bash"
   - Tool completes: status update → "working"
   - Session error: worker fails, reports error
   - Session idle: worker completes, sends result to channel
6. Channel receives WorkerComplete event, responds to user
```

### Interactive

Same as above, but after the initial task completes the worker enters a follow-up loop:

```
7. Worker sends status: "waiting for follow-up"
8. User says "also update the tests"
9. Channel routes message to worker via input_tx
10. Worker sends new prompt to the same OpenCode session
11. Worker processes events until session goes idle again
12. Back to step 7 (until input channel closes)
```

The OpenCode session maintains its context across follow-ups — it remembers what it already did.

## Permissions and Questions

When OpenCode encounters an action that requires confirmation (file edit, shell command, etc.), it emits a `permission.asked` event. The worker:

1. Sends a `ProcessEvent::WorkerPermission` to the channel
2. Auto-approves the permission (when config permissions are `"allow"`)

If permissions are set to `"ask"` in config, the channel would need to present the permission to the user and route the response back. This path exists in the event types but the channel-side UI for it is not yet built.

Questions work similarly — OpenCode's question tool emits `question.asked`, the worker sends `ProcessEvent::WorkerQuestion` to the channel and auto-selects the first option.

## Module Structure

```
src/opencode.rs           — module root, re-exports
src/opencode/types.rs     — API request/response/event types (serde)
src/opencode/server.rs    — OpenCodeServer + OpenCodeServerPool
src/opencode/worker.rs    — OpenCodeWorker (session lifecycle, SSE loop)
```

### Key Types

```
OpenCodeServer         — single server process (spawn, health check, API methods)
OpenCodeServerPool     — per-directory server management (get_or_create, shutdown_all)
OpenCodeWorker         — drives an OpenCode session (create, prompt, events, complete)
OpenCodeConfig         — config struct (path, permissions, limits)
OpenCodePermissions    — permission settings (edit, bash, webfetch)
OpenCodeEnvConfig      — JSON config passed via OPENCODE_CONFIG_CONTENT env var
SseEvent               — tagged enum for SSE event parsing
Part                   — message content parts (text, tool, step-start, step-finish)
```

### ProcessEvent Additions

```rust
WorkerPermission {
    agent_id, worker_id, channel_id,
    permission_id, description, patterns,
}

WorkerQuestion {
    agent_id, worker_id, channel_id,
    question_id, questions: Vec<QuestionInfo>,
}
```

## What's Not Built Yet

- **Channel-side permission/question UI** — events are emitted but the channel doesn't present them to users yet. Permissions are auto-approved.
- **Interactive worker `input_tx` routing** — same pre-existing gap as built-in interactive workers. The `input_tx` is dropped at spawn time, so `route_to_worker` can't deliver messages. Both worker types need this fixed.
- **Model selection passthrough** — the worker can set a model, but currently doesn't read from routing config. OpenCode uses whatever model is configured in its own config or the one passed per-prompt.
- **OpenCode plugin injection** — kimaki injects a plugin into OpenCode for Discord-specific tools. Spacebot could do the same for memory integration or identity-aware behavior.
- **Remote OpenCode servers** — kimaki has plans for remote servers over tunnels. Spacebot's `OpenCodeServerPool` currently only manages local subprocesses.
