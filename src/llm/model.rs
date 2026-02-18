//! SpacebotModel: Custom CompletionModel implementation that routes through LlmManager.

use crate::llm::capabilities::{self, ThinkingSupport};
use crate::llm::manager::LlmManager;
use crate::llm::routing::{
    self, RoutingConfig, MAX_FALLBACK_ATTEMPTS, MAX_RETRIES_PER_MODEL, RETRY_BASE_DELAY_MS,
};

use rig::completion::{
    self, CompletionError, CompletionModel, CompletionRequest, GetTokenUsage,
};
use rig::message::{
    AssistantContent, DocumentSourceKind, Image, Message, MimeType, Reasoning, Text, ToolCall,
    ToolFunction, ToolResult, UserContent,
};
use rig::one_or_many::OneOrMany;
use rig::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Raw provider response. Wraps the JSON so Rig can carry it through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawResponse {
    pub body: serde_json::Value,
}

/// Streaming response placeholder. Streaming will be implemented per-provider
/// when we wire up SSE parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawStreamingResponse {
    pub body: serde_json::Value,
}

impl GetTokenUsage for RawStreamingResponse {
    fn token_usage(&self) -> Option<completion::Usage> {
        None
    }
}

/// Custom completion model that routes through LlmManager.
///
/// Optionally holds a RoutingConfig for fallback behavior. When present,
/// completion() will try fallback models on retriable errors.
#[derive(Clone)]
pub struct SpacebotModel {
    llm_manager: Arc<LlmManager>,
    model_name: String,
    provider: String,
    full_model_name: String,
    routing: Option<RoutingConfig>,
}

impl SpacebotModel {
    pub fn provider(&self) -> &str { &self.provider }
    pub fn model_name(&self) -> &str { &self.model_name }
    pub fn full_model_name(&self) -> &str { &self.full_model_name }

    /// Attach routing config for fallback behavior.
    pub fn with_routing(mut self, routing: RoutingConfig) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Direct call to the provider (no fallback logic).
    async fn attempt_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        match self.provider.as_str() {
            "anthropic" => self.call_anthropic(request).await,
            "openai" => self.call_openai(request).await,
            "openrouter" => self.call_openrouter(request).await,
            "zhipu" => self.call_zhipu(request).await,
            "groq" => self.call_groq(request).await,
            "together" => self.call_together(request).await,
            "fireworks" => self.call_fireworks(request).await,
            "deepseek" => self.call_deepseek(request).await,
            "xai" => self.call_xai(request).await,
            "mistral" => self.call_mistral(request).await,
            "ollama" => self.call_ollama(request).await,
            "opencode-zen" => self.call_opencode_zen(request).await,
            "nvidia" => self.call_nvidia(request).await,
            other => Err(CompletionError::ProviderError(format!(
                "unknown provider: {other}"
            ))),
        }
    }

    /// Try a model with retries and exponential backoff on transient errors.
    ///
    /// Returns `Ok(response)` on success, or `Err((last_error, was_rate_limit))`
    /// after exhausting retries. `was_rate_limit` indicates the final failure was
    /// a 429/rate-limit (as opposed to a timeout or server error), so the caller
    /// can decide whether to record cooldown.
    async fn attempt_with_retries(
        &self,
        model_name: &str,
        request: &CompletionRequest,
    ) -> Result<
        completion::CompletionResponse<RawResponse>,
        (CompletionError, bool),
    > {
        let model = if model_name == self.full_model_name {
            self.clone()
        } else {
            SpacebotModel::make(&self.llm_manager, model_name)
        };

        let mut last_error = None;
        for attempt in 0..MAX_RETRIES_PER_MODEL {
            if attempt > 0 {
                let delay_ms = RETRY_BASE_DELAY_MS * 2u64.pow((attempt - 1) as u32);
                tracing::debug!(
                    model = %model_name,
                    attempt = attempt + 1,
                    delay_ms,
                    "retrying after backoff"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }

            match model.attempt_completion(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let error_str = error.to_string();
                    if !routing::is_retriable_error(&error_str) {
                        // Non-retriable (auth error, bad request, etc) — bail immediately
                        return Err((error, false));
                    }
                    tracing::warn!(
                        model = %model_name,
                        attempt = attempt + 1,
                        %error,
                        "retriable error"
                    );
                    last_error = Some(error_str);
                }
            }
        }

        let error_str = last_error.unwrap_or_default();
        let was_rate_limit = routing::is_rate_limit_error(&error_str);
        Err((
            CompletionError::ProviderError(format!(
                "{model_name} failed after {MAX_RETRIES_PER_MODEL} attempts: {error_str}"
            )),
            was_rate_limit,
        ))
    }
}

impl CompletionModel for SpacebotModel {
    type Response = RawResponse;
    type StreamingResponse = RawStreamingResponse;
    type Client = Arc<LlmManager>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        let full_name = model.into();

        // OpenRouter model names have the form "openrouter/provider/model",
        // so split on the first "/" only and keep the rest as the model name.
        let (provider, model_name) = if let Some(rest) = full_name.strip_prefix("openrouter/") {
            ("openrouter".to_string(), rest.to_string())
        } else if let Some((p, m)) = full_name.split_once('/') {
            (p.to_string(), m.to_string())
        } else {
            ("anthropic".to_string(), full_name.clone())
        };

        let full_model_name = format!("{provider}/{model_name}");

        Self {
            llm_manager: client.clone(),
            model_name,
            provider,
            full_model_name,
            routing: None,
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

        let result = async move {
        let Some(routing) = &self.routing else {
            // No routing config — just call the model directly, no fallback/retry
            return self.attempt_completion(request).await;
        };

        let cooldown = routing.rate_limit_cooldown_secs;
        let fallbacks = routing.get_fallbacks(&self.full_model_name);
        let mut last_error: Option<CompletionError> = None;

        // Try the primary model (with retries) unless it's in rate-limit cooldown
        // and we have fallbacks to try instead.
        let primary_rate_limited = self
            .llm_manager
            .is_rate_limited(&self.full_model_name, cooldown)
            .await;

        let skip_primary = primary_rate_limited && !fallbacks.is_empty();

        if skip_primary {
            tracing::debug!(
                model = %self.full_model_name,
                "primary model in rate-limit cooldown, skipping to fallbacks"
            );
        } else {
            match self.attempt_with_retries(&self.full_model_name, &request).await {
                Ok(response) => return Ok(response),
                Err((error, was_rate_limit)) => {
                    if was_rate_limit {
                        self.llm_manager.record_rate_limit(&self.full_model_name).await;
                    }
                    if fallbacks.is_empty() {
                        // No fallbacks — this is the final error
                        return Err(error);
                    }
                    let error_str = error.to_string();
                    let is_retriable = routing::is_retriable_error(&error_str);
                    if is_retriable {
                        tracing::warn!(
                            model = %self.full_model_name,
                            "primary model exhausted retries (retriable errors), trying fallbacks"
                        );
                    } else {
                        // Non-retriable errors (400 bad request, auth failures, etc.)
                        // falling through to fallbacks is intentional for availability,
                        // but this should be investigated — it likely indicates a config issue.
                        tracing::error!(
                            model = %self.full_model_name,
                            error = %error_str,
                            "PRIMARY MODEL FAILED with non-retriable error — falling back to secondary model. \
                             This likely indicates a configuration issue (wrong model ID, invalid parameters, etc.)"
                        );
                    }
                    last_error = Some(error);
                }
            }
        }

        // Try fallback chain, each with their own retry loop
        for (index, fallback_name) in fallbacks.iter().take(MAX_FALLBACK_ATTEMPTS).enumerate() {
            if self.llm_manager.is_rate_limited(fallback_name, cooldown).await {
                tracing::debug!(
                    fallback = %fallback_name,
                    "fallback model in cooldown, skipping"
                );
                continue;
            }

            match self.attempt_with_retries(fallback_name, &request).await {
                Ok(response) => {
                    tracing::info!(
                        original = %self.full_model_name,
                        fallback = %fallback_name,
                        attempt = index + 1,
                        "fallback model succeeded"
                    );
                    return Ok(response);
                }
                Err((error, was_rate_limit)) => {
                    if was_rate_limit {
                        self.llm_manager.record_rate_limit(fallback_name).await;
                    }
                    tracing::warn!(
                        fallback = %fallback_name,
                        "fallback model exhausted retries, continuing chain"
                    );
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            CompletionError::ProviderError("all models in fallback chain failed".into())
        }))
        }
        .await;

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_secs_f64();
            let metrics = crate::telemetry::Metrics::global();
            // TODO: agent_id and tier are "unknown" because SpacebotModel doesn't
            // carry process context. Thread agent_id/ProcessType through to get
            // per-agent, per-tier breakdowns.
            metrics
                .llm_requests_total
                .with_label_values(&["unknown", &self.full_model_name, "unknown"])
                .inc();
            metrics
                .llm_request_duration_seconds
                .with_label_values(&["unknown", &self.full_model_name, "unknown"])
                .observe(elapsed);
        }

        result
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<RawStreamingResponse>, CompletionError> {
        Err(CompletionError::ProviderError(
            "streaming not yet implemented".into(),
        ))
    }
}

impl SpacebotModel {
    async fn call_anthropic(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let auth = self
            .llm_manager
            .get_anthropic_auth()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let messages = convert_messages_to_anthropic(&request.chat_history);

        // Look up model capabilities from the catalog.
        let caps = capabilities::get_capabilities(&self.model_name);
        let default_max_tokens = caps.max_output_tokens as u64;

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(default_max_tokens),
        });

        if let Some(preamble) = &request.preamble {
            body["system"] = serde_json::json!(preamble);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        // Configure thinking based on model capabilities.
        //
        // - Adaptive: model decides when/how much to think (`{type: "adaptive"}`).
        //   Available on Sonnet 4.6 and Opus 4.6.
        // - Extended: model must think with an explicit budget
        //   (`{type: "enabled", budget_tokens: N}`).
        //   Used by Haiku 4.5, Sonnet 4.5, and older Opus models.
        // - None: no thinking block sent.
        //
        // When thinking is active, temperature must be 1 or unset (Anthropic API
        // requirement). We strip any explicit temperature to avoid errors.
        match &caps.thinking {
            ThinkingSupport::Adaptive => {
                body["thinking"] = serde_json::json!({"type": "adaptive"});
                if body.get("temperature").is_some() {
                    tracing::debug!(
                        model = %self.model_name,
                        "stripping temperature for adaptive-thinking model (must be 1 or unset)"
                    );
                    body.as_object_mut().unwrap().remove("temperature");
                }
            }
            ThinkingSupport::Extended { default_budget } => {
                // Budget must be less than max_tokens. Leave room for visible output.
                let max_tokens = request.max_tokens.unwrap_or(default_max_tokens);
                let budget = std::cmp::min(*default_budget as u64, max_tokens.saturating_sub(1024));
                body["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget,
                });
                if body.get("temperature").is_some() {
                    tracing::debug!(
                        model = %self.model_name,
                        budget,
                        "stripping temperature for extended-thinking model (must be 1 or unset)"
                    );
                    body.as_object_mut().unwrap().remove("temperature");
                }
            }
            ThinkingSupport::None => {}
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let mut req = self
            .llm_manager
            .http_client()
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        let is_oauth = matches!(&auth, super::manager::AnthropicAuth::OAuthToken(_));

        match &auth {
            super::manager::AnthropicAuth::ApiKey(key) => {
                req = req.header("x-api-key", key);
            }
            super::manager::AnthropicAuth::OAuthToken(token) => {
                req = req
                    .header("Authorization", format!("Bearer {}", token))
                    .header("anthropic-beta", "oauth-2025-04-20");
            }
        }

        // Log tool_result presence at debug level (full payload at trace only)
        let has_tool_results = messages.iter().any(|m| {
            m["content"].as_array().map_or(false, |parts| {
                parts.iter().any(|p| p["type"].as_str() == Some("tool_result"))
            })
        });
        if has_tool_results {
            tracing::debug!(
                model = %self.model_name,
                message_count = messages.len(),
                "Anthropic request contains tool_result blocks"
            );
            if tracing::enabled!(tracing::Level::TRACE) {
                let msgs_str = serde_json::to_string_pretty(&messages).unwrap_or_default();
                tracing::trace!(
                    model = %self.model_name,
                    "Anthropic request messages:\n{}",
                    msgs_str
                );
            }
        }

        let response = req
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("failed to read response body: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| CompletionError::ProviderError(format!(
                "Anthropic response ({status}) is not valid JSON: {e}\nBody: {}", truncate_body(&response_text)
            )))?;

        // Log response metadata at info (never full payload)
        tracing::info!(
            requested_model = %self.model_name,
            response_model = %response_body["model"].as_str().unwrap_or("unknown"),
            stop_reason = %response_body["stop_reason"].as_str().unwrap_or("unknown"),
            input_tokens = response_body["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens = response_body["usage"]["output_tokens"].as_u64().unwrap_or(0),
            "Anthropic API response"
        );
        // Full response body only at trace level
        if tracing::enabled!(tracing::Level::TRACE) {
            let body_str = serde_json::to_string_pretty(&response_body).unwrap_or_default();
            tracing::trace!("Anthropic full response:\n{}", body_str);
        }

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");

            // Provide a clear message for expired OAuth tokens
            if is_oauth && status.as_u16() == 401 {
                return Err(CompletionError::ProviderError(format!(
                    "Anthropic OAuth token expired or invalid ({status}): {message}. \
                     Please run `claude setup-token` again to refresh your Claude Max token."
                )));
            }

            return Err(CompletionError::ProviderError(format!(
                "Anthropic API error ({status}): {message}"
            )));
        }

        parse_anthropic_response(response_body)
    }

    async fn call_openai(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let api_key = self
            .llm_manager
            .get_api_key("openai")
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post("https://api.openai.com/v1/chat/completions")
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("failed to read response body: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| CompletionError::ProviderError(format!(
                "OpenAI response ({status}) is not valid JSON: {e}\nBody: {}", truncate_body(&response_text)
            )))?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "OpenAI API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, "OpenAI")
    }

    async fn call_openrouter(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let api_key = self
            .llm_manager
            .get_api_key("openrouter")
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        // OpenRouter uses the OpenAI chat completions format.
        // model_name is the full OpenRouter model ID (e.g. "anthropic/claude-sonnet-4-20250514").
        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("failed to read response body: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| CompletionError::ProviderError(format!(
                "OpenRouter response ({status}) is not valid JSON: {e}\nBody: {}", truncate_body(&response_text)
            )))?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "OpenRouter API error ({status}): {message}"
            )));
        }

        // OpenRouter returns OpenAI-format responses
        parse_openai_response(response_body, "OpenRouter")
    }

    async fn call_zhipu(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let api_key = self
            .llm_manager
            .get_api_key("zhipu")
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post("https://api.z.ai/api/paas/v4/chat/completions")
            .header("authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("failed to read response body: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| CompletionError::ProviderError(format!(
                "Z.ai response ({status}) is not valid JSON: {e}\nBody: {}", truncate_body(&response_text)
            )))?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "Z.ai API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, "Z.ai")
    }

    /// Generic OpenAI-compatible API call.
    /// Used by providers that implement the OpenAI chat completions format.
    async fn call_openai_compatible(
        &self,
        request: CompletionRequest,
        provider_id: &str,
        provider_display_name: &str,
        endpoint: &str,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let api_key = self
            .llm_manager
            .get_api_key(provider_id)
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        self.call_openai_compatible_with_optional_auth(
            request,
            provider_display_name,
            endpoint,
            Some(api_key),
        ).await
    }

    /// Generic OpenAI-compatible API call with optional bearer auth.
    async fn call_openai_compatible_with_optional_auth(
        &self,
        request: CompletionRequest,
        provider_display_name: &str,
        endpoint: &str,
        api_key: Option<String>,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {

        let mut messages = Vec::new();

        if let Some(preamble) = &request.preamble {
            messages.push(serde_json::json!({
                "role": "system",
                "content": preamble,
            }));
        }

        messages.extend(convert_messages_to_openai(&request.chat_history));

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> = request
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);
        }

        let response = self
            .llm_manager
            .http_client()
            .post(endpoint);
        let response = if let Some(api_key) = api_key {
            response.header("authorization", format!("Bearer {api_key}"))
        } else {
            response
        };
        let response = response
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| CompletionError::ProviderError(format!("failed to read response body: {e}")))?;

        let response_body: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| CompletionError::ProviderError(format!(
                "{provider_display_name} response ({status}) is not valid JSON: {e}\nBody: {}", truncate_body(&response_text)
            )))?;

        if !status.is_success() {
            let message = response_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(CompletionError::ProviderError(format!(
                "{provider_display_name} API error ({status}): {message}"
            )));
        }

        parse_openai_response(response_body, provider_display_name)
    }

    async fn call_groq(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
        ).await
    }

    async fn call_together(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "together",
            "Together AI",
            "https://api.together.xyz/v1/chat/completions",
        ).await
    }

    async fn call_fireworks(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "fireworks",
            "Fireworks AI",
            "https://api.fireworks.ai/inference/v1/chat/completions",
        ).await
    }

    async fn call_deepseek(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com/v1/chat/completions",
        ).await
    }

    async fn call_xai(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "xai",
            "xAI",
            "https://api.x.ai/v1/chat/completions",
        ).await
    }

    async fn call_mistral(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "mistral",
            "Mistral AI",
            "https://api.mistral.ai/v1/chat/completions",
        ).await
    }

    async fn call_ollama(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        let base_url = normalize_ollama_base_url(self.llm_manager.ollama_base_url());
        let endpoint = format!("{base_url}/v1/chat/completions");
        let api_key = self.llm_manager.get_api_key("ollama").ok();

        self.call_openai_compatible_with_optional_auth(
            request,
            "Ollama",
            &endpoint,
            api_key,
        ).await
    }

    async fn call_opencode_zen(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "opencode-zen",
            "OpenCode Zen",
            "https://opencode.ai/zen/v1/chat/completions",
        ).await
    }

    async fn call_nvidia(
        &self,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
        self.call_openai_compatible(
            request,
            "nvidia",
            "NVIDIA NIM",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ).await
    }
}

// --- Helpers ---

fn normalize_ollama_base_url(configured: Option<String>) -> String {
    let mut base_url = configured
        .unwrap_or_else(|| "http://localhost:11434".to_string())
        .trim()
        .trim_end_matches('/')
        .to_string();

    if base_url.ends_with("/api") {
        base_url.truncate(base_url.len() - "/api".len());
    } else if base_url.ends_with("/v1") {
        base_url.truncate(base_url.len() - "/v1".len());
    }

    base_url
}


fn tool_result_content_to_string(content: &OneOrMany<rig::message::ToolResultContent>) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            rig::message::ToolResultContent::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// --- Message conversion ---

fn convert_messages_to_anthropic(messages: &OneOrMany<Message>) -> Vec<serde_json::Value> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => {
                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        UserContent::Text(t) if !t.text.is_empty() => {
                            Some(serde_json::json!({"type": "text", "text": t.text}))
                        }
                        UserContent::Text(_) => None, // Skip empty text blocks
                        UserContent::Image(image) => convert_image_anthropic(image),
                        UserContent::ToolResult(result) => Some(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": result.id,
                            "content": tool_result_content_to_string(&result.content),
                        })),
                        _ => None,
                    })
                    .collect();
                if parts.is_empty() {
                    None // Skip messages with no content
                } else {
                    Some(serde_json::json!({"role": "user", "content": parts}))
                }
            }
            Message::Assistant { content, .. } => {
                // Check if this message has real thinking/reasoning blocks
                let has_reasoning = content.iter().any(|c| matches!(c, AssistantContent::Reasoning(_)));

                let parts: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::Text(t) if !t.text.is_empty() => {
                            // Skip the "[thinking]" placeholder if we have real
                            // reasoning blocks — the placeholder was only needed
                            // for display; the API should see the actual thinking.
                            if has_reasoning && t.text == "[thinking]" {
                                return None;
                            }
                            Some(serde_json::json!({"type": "text", "text": t.text}))
                        }
                        AssistantContent::Text(_) => None, // Skip empty text blocks
                        AssistantContent::ToolCall(tc) => Some(serde_json::json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        })),
                        AssistantContent::Reasoning(reasoning) => {
                            // Convert reasoning blocks back to Anthropic's format.
                            // This is required for tool-use continuations and
                            // recommended for multi-turn conversations.
                            if reasoning.id.as_deref() == Some("redacted_thinking") {
                                // Redacted thinking must be passed back as-is
                                Some(serde_json::json!({
                                    "type": "redacted_thinking",
                                    "data": reasoning.reasoning.first().cloned().unwrap_or_default(),
                                }))
                            } else {
                                let mut block = serde_json::json!({
                                    "type": "thinking",
                                    "thinking": reasoning.reasoning.first().cloned().unwrap_or_default(),
                                });
                                if let Some(sig) = &reasoning.signature {
                                    block["signature"] = serde_json::json!(sig);
                                }
                                Some(block)
                            }
                        }
                        _ => None,
                    })
                    .collect();
                if parts.is_empty() {
                    // Don't skip assistant messages entirely — this breaks the
                    // user/assistant alternation pattern that Anthropic requires.
                    // Insert a minimal text block as fallback.
                    Some(serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "[thinking]"}]}))
                } else {
                    Some(serde_json::json!({"role": "assistant", "content": parts}))
                }
            }
        })
        .collect()
}

fn convert_messages_to_openai(messages: &OneOrMany<Message>) -> Vec<serde_json::Value> {
    let mut result = Vec::new();

    for message in messages.iter() {
        match message {
            Message::User { content } => {
                // Separate tool results (they need their own messages) from content parts
                let mut content_parts: Vec<serde_json::Value> = Vec::new();
                let mut tool_results: Vec<serde_json::Value> = Vec::new();

                for item in content.iter() {
                    match item {
                        UserContent::Text(t) => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": t.text,
                            }));
                        }
                        UserContent::Image(image) => {
                            if let Some(part) = convert_image_openai(image) {
                                content_parts.push(part);
                            }
                        }
                        UserContent::ToolResult(tr) => {
                            tool_results.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tr.id,
                                "content": tool_result_content_to_string(&tr.content),
                            }));
                        }
                        _ => {}
                    }
                }

                if !content_parts.is_empty() {
                    // If there's only one text part and no images, use simple string format
                    if content_parts.len() == 1 && content_parts[0]["type"] == "text" {
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content_parts[0]["text"],
                        }));
                    } else {
                        // Mixed content (text + images): use array-of-parts format
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content_parts,
                        }));
                    }
                }

                result.extend(tool_results);
            }
            Message::Assistant { content, .. } => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Text(t) => {
                            text_parts.push(t.text.clone());
                        }
                        AssistantContent::ToolCall(tc) => {
                            // OpenAI expects arguments as a JSON string
                            let args_string = serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_else(|_| "{}".to_string());
                            tool_calls.push(serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": args_string,
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                let mut msg = serde_json::json!({"role": "assistant"});
                if !text_parts.is_empty() {
                    msg["content"] = serde_json::json!(text_parts.join("\n"));
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::json!(tool_calls);
                }
                result.push(msg);
            }
        }
    }

    result
}

// --- Image conversion helpers ---

/// Convert a rig Image to an Anthropic image content block.
/// Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
fn convert_image_anthropic(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mt| mt.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => Some(serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        })),
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        })),
        _ => None,
    }
}

/// Convert a rig Image to an OpenAI image_url content part.
/// OpenAI/OpenRouter format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
fn convert_image_openai(image: &Image) -> Option<serde_json::Value> {
    let media_type = image
        .media_type
        .as_ref()
        .map(|mt| mt.to_mime_type())
        .unwrap_or("image/jpeg");

    match &image.data {
        DocumentSourceKind::Base64(data) => {
            let data_url = format!("data:{media_type};base64,{data}");
            Some(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            }))
        }
        DocumentSourceKind::Url(url) => Some(serde_json::json!({
            "type": "image_url",
            "image_url": { "url": url }
        })),
        _ => None,
    }
}

/// Truncate a response body for error messages to avoid dumping megabytes of HTML.
fn truncate_body(body: &str) -> &str {
    let limit = 500;
    if body.len() <= limit {
        body
    } else {
        &body[..limit]
    }
}

// --- Response parsing ---

fn make_tool_call(id: String, name: String, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id,
        call_id: None,
        function: ToolFunction { name: name.trim().to_string(), arguments },
        signature: None,
        additional_params: None,
    }
}

fn parse_anthropic_response(
    body: serde_json::Value,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let content_blocks = body["content"]
        .as_array()
        .ok_or_else(|| CompletionError::ResponseError("missing content array".into()))?;

    let mut assistant_content = Vec::new();

    for block in content_blocks {
        match block["type"].as_str() {
            Some("text") => {
                let text = block["text"].as_str().unwrap_or("").to_string();
                assistant_content.push(AssistantContent::Text(Text { text }));
            }
            Some("tool_use") => {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                let arguments = block["input"].clone();
                assistant_content
                    .push(AssistantContent::ToolCall(make_tool_call(id, name, arguments)));
            }
            Some("thinking") => {
                // Adaptive thinking blocks from Opus 4+/Sonnet 4.5+.
                // Preserve these in conversation history so they can be passed back
                // to the API. This is REQUIRED for tool-use continuations (the API
                // rejects requests missing thinking blocks from the last assistant
                // turn) and recommended for multi-turn conversations (the API will
                // auto-filter older thinking blocks and only bill for relevant ones).
                let thinking_text = block["thinking"].as_str().unwrap_or("").to_string();
                let signature = block["signature"].as_str().map(|s| s.to_string());
                tracing::debug!(
                    thinking_len = thinking_text.len(),
                    has_signature = signature.is_some(),
                    "preserving thinking block in conversation history"
                );
                assistant_content.push(AssistantContent::Reasoning(
                    Reasoning::new(&thinking_text).with_signature(signature),
                ));
            }
            Some("redacted_thinking") => {
                // Redacted thinking blocks contain encrypted reasoning that must be
                // passed back to the API exactly as received. We use id="redacted_thinking"
                // as a marker so convert_messages_to_anthropic() can reconstruct the
                // correct JSON format ({"type": "redacted_thinking", "data": "..."}).
                let data = block["data"].as_str().unwrap_or("").to_string();
                tracing::debug!(
                    data_len = data.len(),
                    "preserving redacted_thinking block in conversation history"
                );
                assistant_content.push(AssistantContent::Reasoning(
                    Reasoning::new(&data)
                        .with_id("redacted_thinking".to_string()),
                ));
            }
            Some(other) => {
                tracing::warn!("unknown content block type in Anthropic response: {}", other);
            }
            _ => {}
        }
    }

    // Check if we have any "visible" content (text or tool_use). Reasoning blocks
    // alone aren't sufficient — rig needs at least one Text/ToolCall, and we need
    // something to display to the user.
    let has_visible_content = assistant_content.iter().any(|c| {
        matches!(c, AssistantContent::Text(_) | AssistantContent::ToolCall(_))
    });

    if !has_visible_content {
        let block_types: Vec<&str> = content_blocks
            .iter()
            .filter_map(|b| b["type"].as_str())
            .collect();
        let stop_reason = body["stop_reason"].as_str().unwrap_or("");
        let has_thinking = block_types.iter().any(|&t| t == "thinking");
        let has_reasoning = assistant_content.iter().any(|c| matches!(c, AssistantContent::Reasoning(_)));

        if stop_reason == "end_turn" || has_thinking || has_reasoning || content_blocks.is_empty() {
            // Valid cases where there's no visible text:
            // - end_turn with no text/tool_use (model chose not to respond)
            // - Response contained only thinking blocks (adaptive thinking with no visible output)
            // - Completely empty content array
            //
            // If we already have Reasoning blocks, don't add a text placeholder —
            // the Reasoning blocks are sufficient content for OneOrMany and the
            // "[thinking]" placeholder was leaking into visible Discord replies.
            if has_reasoning {
                tracing::debug!(
                    block_types = ?block_types,
                    stop_reason,
                    "Anthropic response had only thinking blocks, reasoning preserved (no text placeholder needed)"
                );
            } else {
                // No reasoning blocks either — we need a minimal text placeholder
                // to satisfy OneOrMany and the alternation pattern.
                tracing::debug!(
                    block_types = ?block_types,
                    stop_reason,
                    has_reasoning,
                    "Anthropic response had no visible text/tool_use and no reasoning, adding placeholder"
                );
                assistant_content.push(AssistantContent::Text(Text { text: "[thinking]".to_string() }));
            }
        } else if assistant_content.is_empty() {
            // Unexpected empty content — no thinking blocks either
            tracing::warn!(
                content_blocks = content_blocks.len(),
                block_types = ?block_types,
                "Anthropic response had no text/tool_use content blocks"
            );
            if tracing::enabled!(tracing::Level::TRACE) {
                let full_body_str = serde_json::to_string_pretty(&body).unwrap_or_default();
                tracing::trace!("Anthropic empty-content response body:\n{}", full_body_str);
            }
        }
    }

    let choice = OneOrMany::many(assistant_content)
        .map_err(|_| CompletionError::ResponseError("empty response from Anthropic".into()))?;

    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let cached = body["usage"]["cache_read_input_tokens"]
        .as_u64()
        .unwrap_or(0);

    Ok(completion::CompletionResponse {
        choice,
        usage: completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: cached,
        },
        raw_response: RawResponse { body },
    })
}

fn parse_openai_response(
    body: serde_json::Value,
    provider_label: &str,
) -> Result<completion::CompletionResponse<RawResponse>, CompletionError> {
    let choice = &body["choices"][0]["message"];

    let mut assistant_content = Vec::new();

    if let Some(text) = choice["content"].as_str() {
        if !text.is_empty() {
            assistant_content.push(AssistantContent::Text(Text {
                text: text.to_string(),
            }));
        }
    }

    // Some reasoning models (e.g., NVIDIA kimi-k2.5) return reasoning in a separate field
    if assistant_content.is_empty() {
        if let Some(reasoning) = choice["reasoning_content"].as_str() {
            if !reasoning.is_empty() {
                tracing::debug!(
                    provider = %provider_label,
                    "extracted reasoning_content as main content"
                );
                assistant_content.push(AssistantContent::Text(Text {
                    text: reasoning.to_string(),
                }));
            }
        }
    }

    if let Some(tool_calls) = choice["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            // OpenAI-compatible APIs usually return arguments as a JSON string.
            // Some providers return it as a raw JSON object instead.
            let arguments_field = &tc["function"]["arguments"];
            let arguments = arguments_field
                .as_str()
                .and_then(|raw| serde_json::from_str(raw).ok())
                .or_else(|| arguments_field.as_object().map(|_| arguments_field.clone()))
                .unwrap_or(serde_json::json!({}));
            assistant_content
                .push(AssistantContent::ToolCall(make_tool_call(id, name, arguments)));
        }
    }

    let result_choice = OneOrMany::many(assistant_content.clone())
        .map_err(|_| {
            tracing::warn!(
                provider = %provider_label,
                choice = ?choice,
                "empty response from provider"
            );
            CompletionError::ResponseError(format!("empty response from {provider_label}"))
        })?;

    let input_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let cached = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);

    Ok(completion::CompletionResponse {
        choice: result_choice,
        usage: completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: cached,
        },
        raw_response: RawResponse { body },
    })
}
