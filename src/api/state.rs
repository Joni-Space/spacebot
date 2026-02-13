//! Shared state for the HTTP API.

use crate::ProcessEvent;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

/// State shared across all API handlers.
pub struct ApiState {
    pub started_at: Instant,
    /// Aggregated event stream from all agents. SSE clients subscribe here.
    pub event_tx: broadcast::Sender<ApiEvent>,
    /// Per-agent SQLite pools for querying channel/conversation data.
    pub agent_pools: arc_swap::ArcSwap<HashMap<String, sqlx::SqlitePool>>,
}

/// Events sent to SSE clients. Wraps ProcessEvents with agent context.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApiEvent {
    /// A process event from an agent.
    ProcessEvent {
        agent_id: String,
        event: ProcessEvent,
    },
    /// An inbound message from a user.
    InboundMessage {
        agent_id: String,
        channel_id: String,
        sender_id: String,
        text: String,
    },
    /// An outbound message sent by the bot.
    OutboundMessage {
        agent_id: String,
        channel_id: String,
        text: String,
    },
    /// Typing indicator state change.
    TypingState {
        agent_id: String,
        channel_id: String,
        is_typing: bool,
    },
}

impl ApiState {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(512);
        Self {
            started_at: Instant::now(),
            event_tx,
            agent_pools: arc_swap::ArcSwap::from_pointee(HashMap::new()),
        }
    }

    /// Register an agent's event stream. Spawns a task that forwards
    /// ProcessEvents into the aggregated API event stream.
    pub fn register_agent_events(
        &self,
        agent_id: String,
        mut agent_event_rx: broadcast::Receiver<ProcessEvent>,
    ) {
        let api_tx = self.event_tx.clone();
        tokio::spawn(async move {
            loop {
                match agent_event_rx.recv().await {
                    Ok(event) => {
                        let api_event = ApiEvent::ProcessEvent {
                            agent_id: agent_id.clone(),
                            event,
                        };
                        // Ignore send errors (no SSE clients connected)
                        api_tx.send(api_event).ok();
                    }
                    Err(broadcast::error::RecvError::Lagged(count)) => {
                        tracing::debug!(agent_id = %agent_id, count, "API event forwarder lagged, skipped events");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    /// Set the SQLite pools for all agents.
    pub fn set_agent_pools(&self, pools: HashMap<String, sqlx::SqlitePool>) {
        self.agent_pools.store(Arc::new(pools));
    }
}
