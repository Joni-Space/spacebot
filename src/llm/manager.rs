//! LLM manager for provider credentials and HTTP client.
//!
//! The manager is intentionally simple — it holds API keys, an HTTP client,
//! and shared rate limit state. Routing decisions (which model for which
//! process) live on the agent's RoutingConfig, not here.
//!
//! API keys are hot-reloadable via ArcSwap. The file watcher calls
//! `reload_config()` when config.toml changes, and all subsequent
//! `get_api_key()` calls read the new values lock-free.

use crate::config::{LlmConfig, ProviderConfig};
use crate::error::{LlmError, Result};
use anyhow::Context as _;
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Anthropic authentication method.
#[derive(Debug, Clone)]
pub enum AnthropicAuth {
    /// Standard API key (`x-api-key` header).
    ApiKey(String),
    /// Claude Max OAuth token (`Authorization: Bearer` + `anthropic-beta: oauth-2025-04-20`).
    OAuthToken(String),
}

/// Manages LLM provider clients and tracks rate limit state.
pub struct LlmManager {
    config: ArcSwap<LlmConfig>,
    http_client: reqwest::Client,
    /// Models currently in rate limit cooldown, with the time they were limited.
    rate_limited: Arc<RwLock<HashMap<String, Instant>>>,
}

impl LlmManager {
    /// Create a new LLM manager with the given configuration.
    pub async fn new(config: LlmConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .with_context(|| "failed to build HTTP client")?;

        Ok(Self {
            config: ArcSwap::from_pointee(config),
            http_client,
            rate_limited: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Atomically swap in new provider credentials.
    pub fn reload_config(&self, config: LlmConfig) {
        self.config.store(Arc::new(config));
        tracing::info!("LLM provider keys reloaded");
    }

    pub fn get_provider(&self, provider_id: &str) -> Result<ProviderConfig> {
        let normalized_provider_id = provider_id.to_lowercase();
        let config = self.config.load();

        config
            .providers
            .get(&normalized_provider_id)
            .cloned()
            .ok_or_else(|| LlmError::UnknownProvider(provider_id.to_string()).into())
    }

    /// Get the appropriate API key for a provider.
    pub fn get_api_key(&self, provider_id: &str) -> Result<String> {
        let provider = self.get_provider(provider_id)?;

        if provider.api_key.is_empty() {
            return Err(LlmError::MissingProviderKey(provider_id.to_string()).into());
        }

        Ok(provider.api_key)
    }

    /// Get configured Ollama base URL, if provided.
    pub fn ollama_base_url(&self) -> Option<String> {
        self.config.load().ollama_base_url.clone()
    }

    /// Resolve Anthropic authentication — prefers OAuth token over API key.
    pub fn get_anthropic_auth(&self) -> Result<AnthropicAuth> {
        if let Some(token) = &self.config.anthropic_oauth_token {
            if !token.is_empty() {
                return Ok(AnthropicAuth::OAuthToken(token.clone()));
            }
        }
        if let Some(key) = &self.config.anthropic_key {
            if !key.is_empty() {
                return Ok(AnthropicAuth::ApiKey(key.clone()));
            }
        }
        Err(LlmError::MissingProviderKey("anthropic".into()).into())
    }

    /// Get the HTTP client.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// Resolve a model name to provider and model components.
    /// Format: "provider/model-name" or just "model-name" (defaults to anthropic).
    pub fn resolve_model(&self, model_name: &str) -> Result<(String, String)> {
        if let Some((provider, model)) = model_name.split_once('/') {
            Ok((provider.to_string(), model.to_string()))
        } else {
            Ok(("anthropic".into(), model_name.into()))
        }
    }

    /// Record that a model hit a rate limit.
    pub async fn record_rate_limit(&self, model_name: &str) {
        self.rate_limited
            .write()
            .await
            .insert(model_name.to_string(), Instant::now());
        tracing::warn!(model = %model_name, "model rate limited, entering cooldown");
    }

    /// Check if a model is currently in rate limit cooldown.
    pub async fn is_rate_limited(&self, model_name: &str, cooldown_secs: u64) -> bool {
        let map = self.rate_limited.read().await;
        if let Some(limited_at) = map.get(model_name) {
            limited_at.elapsed().as_secs() < cooldown_secs
        } else {
            false
        }
    }

    /// Clean up expired rate limit entries.
    pub async fn cleanup_rate_limits(&self, cooldown_secs: u64) {
        self.rate_limited
            .write()
            .await
            .retain(|_, limited_at| limited_at.elapsed().as_secs() < cooldown_secs);
    }
}
