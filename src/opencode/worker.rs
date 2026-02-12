//! OpenCode worker: drives an OpenCode session for coding tasks.
//!
//! Instead of running a Rig agent loop with shell/file/exec tools, this worker
//! delegates to an OpenCode subprocess that has its own codebase exploration,
//! context management, and tool suite. Communication happens over HTTP + SSE.

use crate::opencode::server::OpenCodeServerPool;
use crate::opencode::types::*;
use crate::{AgentId, ChannelId, ProcessEvent, WorkerId};

use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};
use uuid::Uuid;

/// An OpenCode-backed worker that drives a coding session via subprocess.
pub struct OpenCodeWorker {
    pub id: WorkerId,
    pub channel_id: Option<ChannelId>,
    pub agent_id: AgentId,
    pub task: String,
    pub directory: PathBuf,
    pub server_pool: Arc<OpenCodeServerPool>,
    pub event_tx: broadcast::Sender<ProcessEvent>,
    /// Input channel for interactive follow-ups (permissions, questions, user messages).
    pub input_rx: Option<mpsc::Receiver<String>>,
    /// System prompt injected into each OpenCode prompt.
    pub system_prompt: Option<String>,
    /// Model override (provider/model format like "anthropic/claude-sonnet-4-20250514").
    pub model: Option<String>,
}

/// Result of an OpenCode worker run.
pub struct OpenCodeWorkerResult {
    pub session_id: String,
    pub result_text: String,
}

impl OpenCodeWorker {
    /// Create a new OpenCode worker.
    pub fn new(
        channel_id: Option<ChannelId>,
        agent_id: AgentId,
        task: impl Into<String>,
        directory: PathBuf,
        server_pool: Arc<OpenCodeServerPool>,
        event_tx: broadcast::Sender<ProcessEvent>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            channel_id,
            agent_id,
            task: task.into(),
            directory,
            server_pool,
            event_tx,
            input_rx: None,
            system_prompt: None,
            model: None,
        }
    }

    /// Create an interactive OpenCode worker that accepts follow-up messages.
    pub fn new_interactive(
        channel_id: Option<ChannelId>,
        agent_id: AgentId,
        task: impl Into<String>,
        directory: PathBuf,
        server_pool: Arc<OpenCodeServerPool>,
        event_tx: broadcast::Sender<ProcessEvent>,
    ) -> (Self, mpsc::Sender<String>) {
        let (input_tx, input_rx) = mpsc::channel(32);
        let mut worker = Self::new(channel_id, agent_id, task, directory, server_pool, event_tx);
        worker.input_rx = Some(input_rx);
        (worker, input_tx)
    }

    /// Set the system prompt injected into OpenCode prompts.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the model to use for this worker.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Run the worker: spawn/reuse an OpenCode server, create a session,
    /// send the task, monitor via SSE, and return the result.
    pub async fn run(mut self) -> anyhow::Result<OpenCodeWorkerResult> {
        self.send_status("starting OpenCode server");

        // Get or create server for this directory
        let server = self.server_pool
            .get_or_create(&self.directory)
            .await
            .with_context(|| format!(
                "failed to get OpenCode server for '{}'",
                self.directory.display()
            ))?;

        self.send_status("creating session");

        // Create a session
        let session = {
            let guard = server.lock().await;
            guard.create_session(Some(format!("spacebot-worker-{}", self.id))).await?
        };
        let session_id = session.id.clone();

        tracing::info!(
            worker_id = %self.id,
            session_id = %session_id,
            directory = %self.directory.display(),
            "OpenCode session created"
        );

        // Subscribe to SSE events before sending the prompt
        let event_response = {
            let guard = server.lock().await;
            guard.subscribe_events().await?
        };

        // Build the prompt request
        let model_param = self.model.as_ref().and_then(|m| parse_model_param(m));
        let prompt_request = SendPromptRequest {
            parts: vec![PartInput::Text {
                text: self.task.clone(),
                synthetic: None,
            }],
            system: self.system_prompt.clone(),
            model: model_param,
            agent: None,
        };

        // Send prompt async so we can process SSE events while it runs
        self.send_status("sending task to OpenCode");
        {
            let guard = server.lock().await;
            guard.send_prompt_async(&session_id, &prompt_request).await?;
        }

        // Process SSE events until session goes idle or errors
        let result_text = self.process_events(
            event_response,
            &session_id,
            &server,
        ).await?;

        // Interactive follow-up loop
        if let Some(mut input_rx) = self.input_rx.take() {
            self.send_status("waiting for follow-up");

            while let Some(follow_up) = input_rx.recv().await {
                self.send_status("processing follow-up");

                // Subscribe to fresh events for the follow-up
                let event_response = {
                    let guard = server.lock().await;
                    guard.subscribe_events().await?
                };

                let follow_up_request = SendPromptRequest {
                    parts: vec![PartInput::Text {
                        text: follow_up,
                        synthetic: None,
                    }],
                    system: self.system_prompt.clone(),
                    model: self.model.as_ref().and_then(|m| parse_model_param(m)),
                    agent: None,
                };

                {
                    let guard = server.lock().await;
                    guard.send_prompt_async(&session_id, &follow_up_request).await?;
                }

                match self.process_events(event_response, &session_id, &server).await {
                    Ok(_) => {
                        self.send_status("waiting for follow-up");
                    }
                    Err(error) => {
                        tracing::error!(
                            worker_id = %self.id,
                            %error,
                            "OpenCode follow-up failed"
                        );
                        self.send_status("failed");
                        break;
                    }
                }
            }
        }

        self.send_status("completed");

        tracing::info!(
            worker_id = %self.id,
            session_id = %session_id,
            "OpenCode worker completed"
        );

        Ok(OpenCodeWorkerResult {
            session_id,
            result_text,
        })
    }

    /// Process SSE events from the OpenCode event stream until the session
    /// goes idle or encounters an error.
    async fn process_events(
        &self,
        response: reqwest::Response,
        session_id: &str,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
    ) -> anyhow::Result<String> {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut last_text = String::new();
        let mut current_tool: Option<String> = None;

        loop {
            let chunk = tokio::select! {
                chunk = stream.next() => chunk,
                // Give the event loop a chance to process other tasks
                _ = tokio::time::sleep(std::time::Duration::from_secs(600)) => {
                    bail!("OpenCode session timed out after 10 minutes of inactivity");
                }
            };

            let Some(chunk) = chunk else {
                // Stream ended unexpectedly
                bail!("OpenCode event stream ended before session completed");
            };

            let bytes = chunk.context("failed to read SSE chunk")?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Parse SSE lines from buffer
            while let Some(event) = extract_sse_event(&mut buffer) {
                match self.handle_sse_event(
                    &event,
                    session_id,
                    server,
                    &mut last_text,
                    &mut current_tool,
                ).await {
                    EventAction::Continue => {}
                    EventAction::Complete => return Ok(last_text.clone()),
                    EventAction::Error(message) => bail!("OpenCode session error: {message}"),
                }
            }
        }
    }

    /// Handle a single SSE event. Returns whether to continue, complete, or error.
    async fn handle_sse_event(
        &self,
        event: &SseEvent,
        session_id: &str,
        server: &Arc<Mutex<crate::opencode::server::OpenCodeServer>>,
        last_text: &mut String,
        current_tool: &mut Option<String>,
    ) -> EventAction {
        match event {
            SseEvent::MessagePartUpdated { part, .. } => {
                match part {
                    Part::Text { text, session_id: part_session, .. } => {
                        // Only process events for our session
                        if let Some(sid) = part_session {
                            if sid != session_id {
                                return EventAction::Continue;
                            }
                        }
                        *last_text = text.clone();
                    }
                    Part::Tool { tool, state, session_id: part_session, .. } => {
                        if let Some(sid) = part_session {
                            if sid != session_id {
                                return EventAction::Continue;
                            }
                        }
                        if let Some(tool_name) = tool {
                            match state {
                                Some(ToolState::Running) => {
                                    *current_tool = Some(tool_name.clone());
                                    self.send_status(&format!("running: {tool_name}"));
                                }
                                Some(ToolState::Completed) => {
                                    if current_tool.as_deref() == Some(tool_name.as_str()) {
                                        *current_tool = None;
                                    }
                                    self.send_status("working");
                                }
                                Some(ToolState::Error) => {
                                    self.send_status(&format!("tool error: {tool_name}"));
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
                EventAction::Continue
            }

            SseEvent::SessionIdle { session_id: event_session_id } => {
                if event_session_id == session_id {
                    return EventAction::Complete;
                }
                EventAction::Continue
            }

            SseEvent::SessionError { session_id: event_session_id, error } => {
                if event_session_id.as_deref() == Some(session_id) {
                    let message = error
                        .as_ref()
                        .and_then(|e| e.get("message").and_then(|v| v.as_str()))
                        .unwrap_or("unknown error")
                        .to_string();
                    return EventAction::Error(message);
                }
                EventAction::Continue
            }

            SseEvent::PermissionAsked(permission) => {
                if permission.session_id != session_id {
                    return EventAction::Continue;
                }

                tracing::info!(
                    worker_id = %self.id,
                    permission_id = %permission.id,
                    permission_type = ?permission.permission,
                    patterns = ?permission.patterns,
                    "OpenCode requesting permission"
                );

                // Send permission request to channel for user decision
                let _ = self.event_tx.send(ProcessEvent::WorkerPermission {
                    agent_id: self.agent_id.clone(),
                    worker_id: self.id,
                    channel_id: self.channel_id.clone(),
                    permission_id: permission.id.clone(),
                    description: format!(
                        "{}: {}",
                        permission.permission.as_deref().unwrap_or("unknown"),
                        permission.patterns.join(", ")
                    ),
                    patterns: permission.patterns.clone(),
                });

                // For now, auto-allow. The channel event handler can override
                // this by calling reply_permission on the server before we do.
                // In practice, the OPENCODE_CONFIG_CONTENT permissions should
                // prevent most permission prompts from appearing.
                let guard = server.lock().await;
                if let Err(error) = guard.reply_permission(&permission.id, PermissionReply::Once).await {
                    tracing::warn!(
                        worker_id = %self.id,
                        permission_id = %permission.id,
                        %error,
                        "failed to auto-reply permission"
                    );
                }

                EventAction::Continue
            }

            SseEvent::QuestionAsked(question) => {
                if question.session_id != session_id {
                    return EventAction::Continue;
                }

                tracing::info!(
                    worker_id = %self.id,
                    question_id = %question.id,
                    question_count = question.questions.len(),
                    "OpenCode asking question"
                );

                // Send question to channel for user answer
                let _ = self.event_tx.send(ProcessEvent::WorkerQuestion {
                    agent_id: self.agent_id.clone(),
                    worker_id: self.id,
                    channel_id: self.channel_id.clone(),
                    question_id: question.id.clone(),
                    questions: question.questions.iter().map(|q| {
                        crate::opencode::types::QuestionInfo {
                            question: q.question.clone(),
                            header: q.header.clone(),
                            options: q.options.clone(),
                        }
                    }).collect(),
                });

                // Auto-select first option if available, otherwise we need
                // to wait for user input via the input_rx channel
                let answers: Vec<QuestionAnswer> = question.questions.iter().map(|q| {
                    if let Some(first_option) = q.options.first() {
                        QuestionAnswer {
                            label: first_option.label.clone(),
                            description: first_option.description.clone(),
                        }
                    } else {
                        QuestionAnswer {
                            label: "continue".to_string(),
                            description: None,
                        }
                    }
                }).collect();

                let guard = server.lock().await;
                if let Err(error) = guard.reply_question(&question.id, answers).await {
                    tracing::warn!(
                        worker_id = %self.id,
                        question_id = %question.id,
                        %error,
                        "failed to auto-reply question"
                    );
                }

                EventAction::Continue
            }

            SseEvent::SessionStatus { session_id: event_session_id, status } => {
                if event_session_id == session_id {
                    match status {
                        SessionStatusPayload::Retry { attempt, message, .. } => {
                            let description = message.as_deref().unwrap_or("rate limited");
                            self.send_status(&format!("retry attempt {attempt}: {description}"));
                        }
                        SessionStatusPayload::Busy => {
                            self.send_status("working");
                        }
                        SessionStatusPayload::Idle => {
                            // Handled by SessionIdle event
                        }
                    }
                }
                EventAction::Continue
            }

            _ => EventAction::Continue,
        }
    }

    /// Send a status update via the process event bus.
    fn send_status(&self, status: &str) {
        let _ = self.event_tx.send(ProcessEvent::WorkerStatus {
            agent_id: self.agent_id.clone(),
            worker_id: self.id,
            channel_id: self.channel_id.clone(),
            status: status.to_string(),
        });
    }
}

/// Result of processing a single SSE event.
enum EventAction {
    Continue,
    Complete,
    Error(String),
}

/// Parse an SSE event from a buffer. Returns the parsed event and removes the
/// consumed bytes from the buffer. Returns None if no complete event is available.
fn extract_sse_event(buffer: &mut String) -> Option<SseEvent> {
    // SSE format: lines starting with "data: " followed by JSON, terminated by
    // a blank line. We may also see "event:" and "id:" lines which we ignore.
    loop {
        let double_newline = buffer.find("\n\n")?;
        let block = buffer[..double_newline].to_string();
        *buffer = buffer[double_newline + 2..].to_string();

        // Extract all data lines from the block
        let mut data_parts = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                data_parts.push(data);
            } else if let Some(data) = line.strip_prefix("data:") {
                data_parts.push(data);
            }
        }

        if data_parts.is_empty() {
            continue;
        }

        let json_str = data_parts.join("\n");
        if json_str.is_empty() {
            continue;
        }

        match serde_json::from_str::<SseEvent>(&json_str) {
            Ok(event) => return Some(event),
            Err(error) => {
                tracing::trace!(
                    %error,
                    json = %json_str,
                    "failed to parse SSE event, skipping"
                );
                continue;
            }
        }
    }
}

/// Parse a model string like "anthropic/claude-sonnet-4-20250514" into a ModelParam.
fn parse_model_param(model: &str) -> Option<ModelParam> {
    let (provider, model_id) = model.split_once('/')?;
    Some(ModelParam {
        provider_id: provider.to_string(),
        model_id: model_id.to_string(),
    })
}
