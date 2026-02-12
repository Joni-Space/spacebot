//! OpenCode server process management and HTTP client.
//!
//! Manages persistent OpenCode server processes (one per working directory).
//! Each server is spawned as `opencode serve --port <port>` and communicated
//! with via its HTTP API. Servers are reused across worker tasks targeting
//! the same directory.

use crate::opencode::types::*;

use anyhow::{Context as _, bail};
use reqwest::Client;
use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Maximum health check attempts during server startup.
const HEALTH_CHECK_MAX_ATTEMPTS: u32 = 30;
/// Delay between health check attempts in milliseconds.
const HEALTH_CHECK_INTERVAL_MS: u64 = 1000;
/// Maximum restart attempts before giving up.
const MAX_RESTART_RETRIES: u32 = 5;

/// A running OpenCode server process bound to a specific directory.
pub struct OpenCodeServer {
    directory: PathBuf,
    port: u16,
    process: Option<Child>,
    base_url: String,
    client: Client,
    restart_count: u32,
    opencode_path: String,
    permissions: OpenCodePermissions,
}

impl OpenCodeServer {
    /// Spawn a new OpenCode server for the given directory.
    pub async fn spawn(
        directory: PathBuf,
        opencode_path: &str,
        permissions: &OpenCodePermissions,
    ) -> anyhow::Result<Self> {
        let port = find_free_port()?;
        let base_url = format!("http://127.0.0.1:{port}");

        let env_config = OpenCodeEnvConfig::new(permissions);
        let config_json = serde_json::to_string(&env_config)
            .context("failed to serialize OpenCode config")?;

        tracing::info!(
            directory = %directory.display(),
            port,
            "spawning OpenCode server"
        );

        let process = Command::new(opencode_path)
            .args(["serve", "--port", &port.to_string()])
            .current_dir(&directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("OPENCODE_CONFIG_CONTENT", &config_json)
            .env("OPENCODE_PORT", port.to_string())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!(
                "failed to spawn OpenCode at '{}' for directory '{}'",
                opencode_path, directory.display()
            ))?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("failed to create HTTP client")?;

        let server = Self {
            directory,
            port,
            process: Some(process),
            base_url,
            client,
            restart_count: 0,
            opencode_path: opencode_path.to_string(),
            permissions: permissions.clone(),
        };

        server.wait_for_health().await?;

        tracing::info!(
            directory = %server.directory.display(),
            port = server.port,
            "OpenCode server ready"
        );

        Ok(server)
    }

    /// Poll the health endpoint until the server is ready.
    async fn wait_for_health(&self) -> anyhow::Result<()> {
        for attempt in 1..=HEALTH_CHECK_MAX_ATTEMPTS {
            match self.health_check().await {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    tracing::trace!(
                        attempt,
                        directory = %self.directory.display(),
                        "health check returned unhealthy, retrying"
                    );
                }
                Err(error) => {
                    tracing::trace!(
                        attempt,
                        %error,
                        directory = %self.directory.display(),
                        "health check failed, retrying"
                    );
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(HEALTH_CHECK_INTERVAL_MS)).await;
        }

        bail!(
            "OpenCode server for '{}' failed to become healthy after {} attempts",
            self.directory.display(),
            HEALTH_CHECK_MAX_ATTEMPTS
        );
    }

    /// Check if the server is healthy.
    async fn health_check(&self) -> anyhow::Result<bool> {
        // Try /global/health first (v2), fall back to /api/health
        let url = format!("{}/global/health", self.base_url);
        let response = self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;

        if response.status().is_success() {
            return Ok(true);
        }

        // Fallback
        let url = format!("{}/api/health", self.base_url);
        let response = self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    /// Check if the underlying process is still running.
    pub fn is_alive(&mut self) -> bool {
        match &mut self.process {
            Some(child) => child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }

    /// Restart the server process. Reuses the same directory and config.
    pub async fn restart(&mut self) -> anyhow::Result<()> {
        self.restart_count += 1;
        if self.restart_count > MAX_RESTART_RETRIES {
            bail!(
                "OpenCode server for '{}' exceeded max restart attempts ({})",
                self.directory.display(),
                MAX_RESTART_RETRIES
            );
        }

        tracing::warn!(
            directory = %self.directory.display(),
            restart_count = self.restart_count,
            "restarting OpenCode server"
        );

        // Kill existing process if still running
        if let Some(mut child) = self.process.take() {
            let _ = child.kill().await;
        }

        let port = find_free_port()?;
        let base_url = format!("http://127.0.0.1:{port}");

        let env_config = OpenCodeEnvConfig::new(&self.permissions);
        let config_json = serde_json::to_string(&env_config)?;

        let process = Command::new(&self.opencode_path)
            .args(["serve", "--port", &port.to_string()])
            .current_dir(&self.directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("OPENCODE_CONFIG_CONTENT", &config_json)
            .env("OPENCODE_PORT", port.to_string())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!(
                "failed to restart OpenCode server for '{}'",
                self.directory.display()
            ))?;

        self.port = port;
        self.base_url = base_url;
        self.process = Some(process);

        self.wait_for_health().await?;

        tracing::info!(
            directory = %self.directory.display(),
            port = self.port,
            restart_count = self.restart_count,
            "OpenCode server restarted"
        );

        Ok(())
    }

    /// Kill the server process.
    pub async fn kill(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill().await;
            tracing::info!(
                directory = %self.directory.display(),
                "OpenCode server killed"
            );
        }
    }

    // -- API methods --

    /// Get the base URL for this server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the directory this server is bound to.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Create a new session.
    pub async fn create_session(&self, title: Option<String>) -> anyhow::Result<Session> {
        let url = format!("{}/session", self.base_url);
        let body = CreateSessionRequest { title };

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .json(&body)
            .send()
            .await
            .context("failed to create OpenCode session")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("create session failed ({status}): {text}");
        }

        response.json::<Session>().await
            .context("failed to parse session response")
    }

    /// Send a prompt to a session (blocking until complete).
    pub async fn send_prompt(
        &self,
        session_id: &str,
        request: &SendPromptRequest,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/session/{}/message", self.base_url, session_id);

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .json(request)
            .send()
            .await
            .context("failed to send prompt to OpenCode session")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("send prompt failed ({status}): {text}");
        }

        response.json::<serde_json::Value>().await
            .context("failed to parse prompt response")
    }

    /// Send a prompt asynchronously (returns immediately, use SSE events for results).
    pub async fn send_prompt_async(
        &self,
        session_id: &str,
        request: &SendPromptRequest,
    ) -> anyhow::Result<()> {
        let url = format!("{}/session/{}/prompt_async", self.base_url, session_id);

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .json(request)
            .send()
            .await
            .context("failed to send async prompt")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("async prompt failed ({status}): {text}");
        }

        Ok(())
    }

    /// Abort a session.
    pub async fn abort_session(&self, session_id: &str) -> anyhow::Result<()> {
        let url = format!("{}/session/{}/abort", self.base_url, session_id);

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .send()
            .await
            .context("failed to abort OpenCode session")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("abort session failed ({status}): {text}");
        }

        Ok(())
    }

    /// Reply to a permission request.
    pub async fn reply_permission(
        &self,
        request_id: &str,
        reply: PermissionReply,
    ) -> anyhow::Result<()> {
        let url = format!("{}/permission/{}/reply", self.base_url, request_id);
        let body = PermissionReplyRequest {
            reply,
            message: None,
        };

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .json(&body)
            .send()
            .await
            .context("failed to reply to permission")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("permission reply failed ({status}): {text}");
        }

        Ok(())
    }

    /// Reply to a question request.
    pub async fn reply_question(
        &self,
        request_id: &str,
        answers: Vec<QuestionAnswer>,
    ) -> anyhow::Result<()> {
        let url = format!("{}/question/{}/reply", self.base_url, request_id);
        let body = QuestionReplyRequest { answers };

        let response = self.client
            .post(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .json(&body)
            .send()
            .await
            .context("failed to reply to question")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("question reply failed ({status}): {text}");
        }

        Ok(())
    }

    /// Subscribe to the SSE event stream. Returns a response whose body can
    /// be read as a byte stream and parsed line-by-line for SSE events.
    pub async fn subscribe_events(&self) -> anyhow::Result<reqwest::Response> {
        let url = format!("{}/event", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .header("Accept", "text/event-stream")
            .timeout(std::time::Duration::from_secs(86400)) // long-lived
            .send()
            .await
            .context("failed to subscribe to OpenCode event stream")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("event subscription failed ({status}): {text}");
        }

        Ok(response)
    }

    /// Get messages for a session (for reading final results).
    pub async fn get_messages(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let url = format!("{}/session/{}/message", self.base_url, session_id);

        let response = self.client
            .get(&url)
            .query(&[("directory", self.directory.to_str().unwrap_or("."))])
            .send()
            .await
            .context("failed to get session messages")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("get messages failed ({status}): {text}");
        }

        response.json::<Vec<serde_json::Value>>().await
            .context("failed to parse messages response")
    }
}

impl Drop for OpenCodeServer {
    fn drop(&mut self) {
        // Kill the process synchronously on drop to avoid orphans.
        // `kill_on_drop(true)` handles this for tokio::process::Child,
        // but we log it for visibility.
        if self.process.is_some() {
            tracing::debug!(
                directory = %self.directory.display(),
                "OpenCode server dropped, process will be killed"
            );
        }
    }
}

/// Pool of OpenCode server processes, one per working directory.
///
/// Shared across all agents and workers. Servers are lazily spawned on first
/// request for a directory and reused for subsequent tasks.
pub struct OpenCodeServerPool {
    servers: Mutex<HashMap<PathBuf, Arc<Mutex<OpenCodeServer>>>>,
    opencode_path: String,
    permissions: OpenCodePermissions,
    max_servers: usize,
}

impl OpenCodeServerPool {
    /// Create a new server pool.
    pub fn new(
        opencode_path: impl Into<String>,
        permissions: OpenCodePermissions,
        max_servers: usize,
    ) -> Self {
        Self {
            servers: Mutex::new(HashMap::new()),
            opencode_path: opencode_path.into(),
            permissions,
            max_servers,
        }
    }

    /// Get or create a server for the given directory.
    ///
    /// If a server exists and is healthy, returns it. If it exists but is dead,
    /// restarts it. If no server exists, spawns a new one (subject to pool limit).
    pub async fn get_or_create(
        &self,
        directory: &Path,
    ) -> anyhow::Result<Arc<Mutex<OpenCodeServer>>> {
        let canonical = directory.canonicalize()
            .with_context(|| format!("directory '{}' does not exist", directory.display()))?;

        let mut servers = self.servers.lock().await;

        // Check existing server
        if let Some(server) = servers.get(&canonical) {
            let mut guard = server.lock().await;
            if guard.is_alive() {
                return Ok(Arc::clone(server));
            }

            // Server died, try restarting
            tracing::warn!(
                directory = %canonical.display(),
                "OpenCode server found dead, restarting"
            );
            guard.restart().await?;
            return Ok(Arc::clone(server));
        }

        // Enforce pool limit
        if servers.len() >= self.max_servers {
            bail!(
                "OpenCode server pool limit reached ({}). Kill an existing server or increase the limit.",
                self.max_servers
            );
        }

        // Spawn new server
        let server = OpenCodeServer::spawn(
            canonical.clone(),
            &self.opencode_path,
            &self.permissions,
        ).await?;

        let server = Arc::new(Mutex::new(server));
        servers.insert(canonical, Arc::clone(&server));

        Ok(server)
    }

    /// Shut down all servers in the pool.
    pub async fn shutdown_all(&self) {
        let mut servers = self.servers.lock().await;
        for (directory, server) in servers.drain() {
            let mut guard = server.lock().await;
            guard.kill().await;
            tracing::info!(
                directory = %directory.display(),
                "OpenCode server shut down"
            );
        }
    }

    /// Number of active servers.
    pub async fn server_count(&self) -> usize {
        self.servers.lock().await.len()
    }
}

/// Find a free TCP port by binding to port 0 and reading the assigned port.
fn find_free_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .context("failed to find free port")?;
    let port = listener.local_addr()
        .context("failed to get local address")?
        .port();
    Ok(port)
}
