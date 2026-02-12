//! Types for communicating with OpenCode's HTTP API.
//!
//! Only the subset of the API surface that Spacebot needs is modeled here.
//! OpenCode has a much larger API (PTY, LSP, TUI, MCP, etc.) that we ignore.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// -- Request types --

/// Body for `POST /session` (create session).
#[derive(Debug, Serialize)]
pub struct CreateSessionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// A single part within a message prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartInput {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        synthetic: Option<bool>,
    },
    File {
        mime: String,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Model selection for a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelParam {
    pub provider_id: String,
    pub model_id: String,
}

/// Body for `POST /session/{id}/message` (send prompt).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SendPromptRequest {
    pub parts: Vec<PartInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

/// Body for `POST /permission/{id}/reply`.
#[derive(Debug, Serialize)]
pub struct PermissionReplyRequest {
    pub reply: PermissionReply,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Permission reply options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionReply {
    Once,
    Always,
    Reject,
}

/// A single question answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnswer {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Body for `POST /question/{id}/reply`.
#[derive(Debug, Serialize)]
pub struct QuestionReplyRequest {
    pub answers: Vec<QuestionAnswer>,
}

// -- Response types --

/// Session object returned by the API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Health check response from `GET /global/health` or `GET /api/health`.
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    #[serde(default)]
    pub healthy: bool,
    #[serde(default)]
    pub version: Option<String>,
}

/// Time span for message/part timing.
#[derive(Debug, Clone, Deserialize)]
pub struct TimeSpan {
    #[serde(default)]
    pub start: Option<f64>,
    #[serde(default)]
    pub end: Option<f64>,
}

/// A message in a session.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    pub role: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub time: Option<TimeSpan>,
}

// -- SSE Event types --

/// Top-level SSE event wrapper. OpenCode sends events as `data: <json>` lines
/// with the event type embedded in the JSON payload's `type` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SseEvent {
    #[serde(rename = "message.updated")]
    MessageUpdated {
        #[serde(default)]
        info: Option<serde_json::Value>,
    },
    #[serde(rename = "message.part.updated")]
    MessagePartUpdated {
        part: Part,
        #[serde(default)]
        delta: Option<String>,
    },
    #[serde(rename = "session.idle")]
    SessionIdle {
        #[serde(rename = "sessionID")]
        session_id: String,
    },
    #[serde(rename = "session.error")]
    SessionError {
        #[serde(rename = "sessionID", default)]
        session_id: Option<String>,
        #[serde(default)]
        error: Option<serde_json::Value>,
    },
    #[serde(rename = "session.status")]
    SessionStatus {
        #[serde(rename = "sessionID")]
        session_id: String,
        status: SessionStatusPayload,
    },
    #[serde(rename = "permission.asked")]
    PermissionAsked(PermissionRequest),
    #[serde(rename = "permission.replied")]
    PermissionReplied {
        #[serde(rename = "sessionID")]
        session_id: String,
        #[serde(rename = "requestID")]
        request_id: String,
        reply: String,
    },
    #[serde(rename = "question.asked")]
    QuestionAsked(QuestionRequest),
    #[serde(rename = "question.replied")]
    QuestionReplied {
        #[serde(rename = "sessionID")]
        session_id: String,
        #[serde(rename = "requestID")]
        request_id: String,
    },
    /// Catch-all for events we don't care about.
    #[serde(other)]
    Unknown,
}

/// A content part within a message.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Part {
    #[serde(rename = "text")]
    Text {
        id: String,
        #[serde(rename = "sessionID", default)]
        session_id: Option<String>,
        #[serde(rename = "messageID", default)]
        message_id: Option<String>,
        #[serde(default)]
        text: String,
        #[serde(default)]
        time: Option<TimeSpan>,
    },
    #[serde(rename = "tool")]
    Tool {
        id: String,
        #[serde(rename = "sessionID", default)]
        session_id: Option<String>,
        #[serde(rename = "messageID", default)]
        message_id: Option<String>,
        #[serde(rename = "callID", default)]
        call_id: Option<String>,
        #[serde(default)]
        tool: Option<String>,
        #[serde(default)]
        state: Option<ToolState>,
    },
    #[serde(rename = "step-start")]
    StepStart {
        id: String,
        #[serde(rename = "sessionID", default)]
        session_id: Option<String>,
    },
    #[serde(rename = "step-finish")]
    StepFinish {
        id: String,
        #[serde(rename = "sessionID", default)]
        session_id: Option<String>,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Catch-all for part types we don't process (reasoning, file, subtask, etc.)
    #[serde(other)]
    Other,
}

/// Tool execution state.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolState {
    Pending,
    Running,
    Completed,
    Error,
}

/// Session status payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionStatusPayload {
    Idle,
    Busy,
    Retry {
        attempt: u32,
        #[serde(default)]
        message: Option<String>,
    },
}

/// Permission request from OpenCode.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequest {
    pub id: String,
    #[serde(rename = "sessionID")]
    pub session_id: String,
    #[serde(default)]
    pub permission: Option<String>,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Question request from OpenCode.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionRequest {
    pub id: String,
    #[serde(rename = "sessionID")]
    pub session_id: String,
    #[serde(default)]
    pub questions: Vec<QuestionInfo>,
}

/// Individual question within a question request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionInfo {
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub header: Option<String>,
    #[serde(default)]
    pub options: Vec<QuestionOption>,
}

/// An option within a question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

// -- OpenCode server config injected via env --

/// Configuration passed to OpenCode via `OPENCODE_CONFIG_CONTENT` env var.
#[derive(Debug, Serialize)]
pub struct OpenCodeEnvConfig {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub lsp: bool,
    pub formatter: bool,
    pub permission: OpenCodePermissions,
}

/// Permission settings for headless OpenCode operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenCodePermissions {
    pub edit: String,
    pub bash: String,
    #[serde(default = "default_webfetch_permission")]
    pub webfetch: String,
}

fn default_webfetch_permission() -> String {
    "allow".to_string()
}

impl Default for OpenCodePermissions {
    fn default() -> Self {
        Self {
            edit: "allow".to_string(),
            bash: "allow".to_string(),
            webfetch: "allow".to_string(),
        }
    }
}

impl OpenCodeEnvConfig {
    /// Build the config JSON that gets passed as `OPENCODE_CONFIG_CONTENT`.
    pub fn new(permissions: &OpenCodePermissions) -> Self {
        Self {
            schema: "https://opencode.ai/config.json".to_string(),
            lsp: false,
            formatter: false,
            permission: permissions.clone(),
        }
    }
}
