# Spacedrive Integration

Future integration between Spacebot and [Spacedrive](https://v2.spacedrive.com) -- an open source cross-platform file manager powered by a virtual distributed filesystem (VDFS) written in Rust. This is a long-term vision document, not a current feature.

## Why This Matters

Spacebot's agents are bottlenecked by what they know. Today, an agent's knowledge comes from three sources: conversation history, its memory graph, and whatever it finds via web search or tool execution. That covers what people say and what the internet knows. It doesn't cover what's on your machines.

Spacedrive indexes files across devices, clouds, and platforms. It treats files as content-addressed objects with rich metadata -- not paths on a disk. A photo on your laptop and the same photo on your NAS are recognized as one piece of content. Spacedrive can search millions of entries in under 100ms, run local AI analysis (face recognition, transcription, scene classification, document analysis), and sync metadata peer-to-peer without servers.

Connect the two and an agent stops being a conversational tool. It becomes something that understands your files, your devices, and your data -- across every machine you own.

## What Spacedrive Brings

Spacedrive v2 is a ground-up rewrite with a production-grade architecture:

- **Content identity** -- BLAKE3 hashing creates a unique fingerprint for every file. Identical content recognized regardless of name or location.
- **Cross-device index** -- unified view of files across macOS, Windows, Linux, iOS, Android, and cloud storage (S3, Google Drive, Dropbox).
- **Semantic search** -- sub-100ms search across millions of indexed entries.
- **Local AI** -- WASM extension system for domain-specific analysis. Photos extension does face recognition and scene classification. Chronicle extension does document analysis and knowledge graphs.
- **P2P sync** -- leaderless synchronization via Iroh/QUIC. No central server. Device-specific data uses state replication, shared metadata uses HLC-ordered logs.
- **Daemon architecture** -- Spacedrive runs as a daemon process. Clients (CLI, desktop, mobile) connect via Unix domain sockets or WebSocket.

Spacedrive's Rust core is ~183k lines. Same tech DNA as Spacebot: Rust, tokio, SQLite, redb.

## Integration Model

Spacedrive exposes its functionality through a daemon with a typed RPC interface. Spacebot's worker model is the natural integration point.

### Spacedrive Worker

A new worker type that connects to a running Spacedrive daemon:

```
Channel
  → branch thinks: "user wants to find that contract from last quarter"
    → spawn_worker { task, worker_type: "spacedrive" }
      → SpacedriveWorker connects to daemon via Unix socket
      → Queries file index, runs semantic search
      → Returns structured results to branch
    → Branch curates results, returns conclusion to channel
  → Channel responds with clean answer
```

The worker talks to Spacedrive's daemon the same way Spacedrive's own CLI and desktop app do. No custom protocol needed.

### Worker Tools

A Spacedrive worker would have access to:

| Tool           | Description                                                                            |
| -------------- | -------------------------------------------------------------------------------------- |
| `sd_search`    | Semantic and metadata search across all indexed locations                              |
| `sd_browse`    | Navigate the virtual filesystem, list entries with metadata                            |
| `sd_file_info` | Get detailed metadata for a specific file (content hash, locations, tags, AI analysis) |
| `sd_tags`      | Query and manage semantic tags across the file graph                                   |
| `sd_jobs`      | Trigger Spacedrive jobs (indexing, thumbnail generation, AI analysis)                  |

These map to Spacedrive's existing CQRS operations. The worker translates between Spacebot's task model and Spacedrive's action/query system.

### Memory Bridge

The interesting part. Spacedrive's content metadata can feed Spacebot's memory graph:

- File analysis results (transcriptions, OCR text, face identities, scene descriptions) become **Fact** memories with source attribution.
- Tag hierarchies and relationships map to memory graph edges.
- File events (new files indexed, duplicates found, content changes) become **Event** memories.
- User file organization patterns become **Observation** memories.

This doesn't mean dumping every file into the memory graph. A branch queries Spacedrive when relevant, and decides what's worth remembering. Same curation model as memory recall -- the branch absorbs the noise, the channel gets clean results.

### Cross-Device Recall

Spacedrive's P2P sync means the Spacedrive daemon knows about files on devices that aren't currently connected. When an agent searches for a file, it can find content that exists on an offline device and report where it lives, even if it can't retrieve it right now.

```
User: "where's that presentation I was working on last week?"
  → Branch spawns Spacedrive worker
  → Worker searches: finds matching file indexed on work laptop
  → Worker returns: "Found 'Q4-Review.pptx' on your work laptop (last seen 2 days ago).
     Also found a copy on your NAS that's 3 versions behind."
  → Branch returns conclusion with locations and version info
```

Content addressing makes this reliable. The agent isn't guessing based on filenames -- it knows the content identity of the file and where copies exist.

## Configuration

```toml
[defaults.spacedrive]
enabled = true
socket_path = "auto"      # auto-discover daemon socket, or explicit path
timeout_secs = 30
```

When `socket_path` is `"auto"`, the worker looks for the Spacedrive daemon at its default socket location. If the daemon isn't running, the worker reports an error and the branch handles it gracefully.

## What This Enables

Some concrete scenarios where the combination is more than the sum of its parts:

**Research assistant** -- "Find everything we have about the Johnson project." The agent searches its memory graph for conversation history and decisions, then searches Spacedrive for related documents, emails (via archive extension), and files. Returns a unified briefing spanning both what was discussed and what exists on disk.

**Proactive organization** -- A cron job queries Spacedrive for recently added files, runs them through local AI analysis, and saves relevant facts to the agent's memory graph. Next time the user asks about something related, the agent already knows about it.

**Multi-device awareness** -- "Is my backup drive up to date?" The agent queries Spacedrive's redundancy tracking across all indexed devices and reports which files are only on one device, which have stale copies, and what's fully backed up.

## Phasing

### Phase 1: Read-Only Worker

Connect to Spacedrive daemon, execute search and browse queries, return results to branches. No writes, no memory bridge. Validates the integration model works.

### Phase 2: Memory Bridge

Spacedrive query results flow into the memory graph when the branch decides they're worth remembering. File metadata enriches the agent's knowledge over time.

### Phase 3: Bidirectional

Agent can trigger Spacedrive operations -- tag files, organize locations, queue analysis jobs. Spacedrive events (new files indexed, sync completed) surface as agent events.

### Phase 4: Extension

Spacebot as a Spacedrive WASM extension. The agent runs inside Spacedrive's extension sandbox with direct access to the core API instead of going through the daemon socket. This inverts the relationship -- instead of Spacebot consuming Spacedrive as a data source, Spacedrive hosts Spacebot as a capability.

## Dependencies

This integration depends on:

- Spacedrive v2 reaching a stable daemon API (alpha.1 released December 2025, in active development)
- Spacebot's worker model being production-ready (current focus)
- A thin Rust client for Spacedrive's RPC protocol (Spacedrive ships `sd-client` crate)

Neither project needs to wait for the other. Spacebot ships and operates independently. The integration is additive.
