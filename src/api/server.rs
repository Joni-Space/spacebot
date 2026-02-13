//! HTTP server setup: router, static file serving, and API routes.

use axum::extract::{Query, State};
use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Json, Response, Sse};
use axum::routing::get;
use axum::Router;
use futures::stream::Stream;
use rust_embed::Embed;
use serde::Deserialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use super::state::{ApiEvent, ApiState};
use crate::conversation::channels::ChannelStore;
use crate::conversation::history::ConversationLogger;

/// Embedded frontend assets from the Vite build output.
#[derive(Embed)]
#[folder = "interface/dist/"]
#[allow(unused)]
struct InterfaceAssets;

/// Start the HTTP server on the given address.
///
/// The caller provides a pre-built `ApiState` so agent event streams and
/// DB pools can be registered after startup.
pub async fn start_http_server(
    bind: SocketAddr,
    state: Arc<ApiState>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api_routes = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/events", get(events_sse))
        .route("/channels", get(list_channels))
        .route("/channels/messages", get(channel_messages));

    let app = Router::new()
        .nest("/api", api_routes)
        .fallback(static_handler)
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let mut shutdown = shutdown_rx;
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown.wait_for(|v| *v).await;
            })
            .await
            .ok();
    });

    Ok(handle)
}

// -- API handlers --

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn status(State(state): State<Arc<ApiState>>) -> Json<serde_json::Value> {
    let uptime = state.started_at.elapsed();
    Json(serde_json::json!({
        "status": "running",
        "pid": std::process::id(),
        "uptime_seconds": uptime.as_secs(),
    }))
}

/// SSE endpoint streaming all agent events to connected clients.
async fn events_sse(
    State(state): State<Arc<ApiState>>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>> {
    let mut rx = state.event_tx.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if let Ok(json) = serde_json::to_string(&event) {
                        let event_type = match &event {
                            ApiEvent::ProcessEvent { .. } => "process_event",
                            ApiEvent::InboundMessage { .. } => "inbound_message",
                            ApiEvent::OutboundMessage { .. } => "outbound_message",
                            ApiEvent::TypingState { .. } => "typing_state",
                        };
                        yield Ok(axum::response::sse::Event::default()
                            .event(event_type)
                            .data(json));
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(count)) => {
                    tracing::debug!(count, "SSE client lagged");
                    yield Ok(axum::response::sse::Event::default()
                        .event("lagged")
                        .data(format!("{{\"skipped\":{count}}}")));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// List active channels across all agents.
async fn list_channels(State(state): State<Arc<ApiState>>) -> Json<serde_json::Value> {
    let pools = state.agent_pools.load();
    let mut all_channels = Vec::new();

    for (agent_id, pool) in pools.iter() {
        let store = ChannelStore::new(pool.clone());
        match store.list_active().await {
            Ok(channels) => {
                for channel in channels {
                    all_channels.push(serde_json::json!({
                        "agent_id": agent_id,
                        "id": channel.id,
                        "platform": channel.platform,
                        "display_name": channel.display_name,
                        "is_active": channel.is_active,
                        "last_activity_at": channel.last_activity_at.to_rfc3339(),
                        "created_at": channel.created_at.to_rfc3339(),
                    }));
                }
            }
            Err(error) => {
                tracing::warn!(%error, agent_id, "failed to list channels");
            }
        }
    }

    Json(serde_json::json!({ "channels": all_channels }))
}

#[derive(Deserialize)]
struct MessagesQuery {
    channel_id: String,
    #[serde(default = "default_message_limit")]
    limit: i64,
}

fn default_message_limit() -> i64 {
    20
}

/// Get recent messages for a specific channel.
async fn channel_messages(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<MessagesQuery>,
) -> Json<serde_json::Value> {
    let pools = state.agent_pools.load();
    let limit = query.limit.min(100);

    for (_agent_id, pool) in pools.iter() {
        let logger = ConversationLogger::new(pool.clone());
        match logger.load_channel_transcript(&query.channel_id, limit).await {
            Ok(messages) if !messages.is_empty() => {
                let result: Vec<serde_json::Value> = messages
                    .into_iter()
                    .map(|message| {
                        serde_json::json!({
                            "id": message.id,
                            "role": message.role,
                            "sender_name": message.sender_name,
                            "sender_id": message.sender_id,
                            "content": message.content,
                            "created_at": message.created_at.to_rfc3339(),
                        })
                    })
                    .collect();
                return Json(serde_json::json!({ "messages": result }));
            }
            Ok(_) => continue,
            Err(error) => {
                tracing::warn!(%error, channel_id = %query.channel_id, "failed to load messages");
                continue;
            }
        }
    }

    Json(serde_json::json!({ "messages": [] }))
}

// -- Static file serving --

async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    if let Some(content) = InterfaceAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, mime.as_ref())],
            content.data,
        )
            .into_response();
    }

    // SPA fallback
    if let Some(content) = InterfaceAssets::get("index.html") {
        return Html(
            std::str::from_utf8(&content.data)
                .unwrap_or("")
                .to_string(),
        )
        .into_response();
    }

    (StatusCode::NOT_FOUND, "not found").into_response()
}
