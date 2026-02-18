//! Model capabilities catalog.
//!
//! Centralizes knowledge about model-specific behavior (token limits, thinking
//! support, context windows) so the rest of the codebase can query capabilities
//! instead of pattern-matching on model name strings.
//!
//! # Usage
//!
//! ```rust
//! use spacebot::llm::capabilities::{get_capabilities, ThinkingSupport};
//!
//! let caps = get_capabilities("claude-opus-4-6");
//! assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
//! assert_eq!(caps.max_output_tokens, 131_072);
//! ```

/// How a model supports (or doesn't support) thinking/reasoning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingSupport {
    /// Model does not support thinking at all.
    None,

    /// Model supports extended thinking with an explicit budget.
    ///
    /// The API requires `thinking: {type: "enabled", budget_tokens: N}`.
    /// The model *must* think — you specify how many tokens it may use.
    /// Used by Haiku 4.5 and older thinking-capable models (Sonnet 4.5, Opus 4).
    Extended {
        /// Suggested default budget when the caller doesn't specify one.
        default_budget: u32,
    },

    /// Model supports adaptive thinking.
    ///
    /// The API accepts `thinking: {type: "adaptive"}` — the model decides
    /// autonomously when and how much to think. Only available on the latest
    /// generation (Sonnet 4.6, Opus 4.6).
    Adaptive,
}

/// Capabilities for a specific model or model family.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Maximum tokens the model can produce in a single response.
    pub max_output_tokens: u32,

    /// What kind of thinking/reasoning the model supports.
    pub thinking: ThinkingSupport,

    /// Context window size in tokens.
    pub context_window: u32,
}

/// Default capabilities for unknown models.
///
/// Conservative defaults: 8k output, no thinking, 200k context.
/// This ensures we never accidentally enable thinking for a model that
/// doesn't support it, while still providing reasonable token limits.
const DEFAULT_CAPABILITIES: ModelCapabilities = ModelCapabilities {
    max_output_tokens: 8_192,
    thinking: ThinkingSupport::None,
    context_window: 200_000,
};

/// Model catalog entry: a name prefix and its associated capabilities.
struct CatalogEntry {
    prefix: &'static str,
    capabilities: ModelCapabilities,
}

/// The model catalog, ordered from most-specific to least-specific prefixes.
///
/// Lookup stops at the first match, so more specific entries (e.g. dated
/// model IDs) must come before broader family prefixes.
const CATALOG: &[CatalogEntry] = &[
    // ── Opus 4.6 ────────────────────────────────────────────────────
    CatalogEntry {
        prefix: "claude-opus-4-6",
        capabilities: ModelCapabilities {
            max_output_tokens: 131_072,
            thinking: ThinkingSupport::Adaptive,
            context_window: 1_000_000,
        },
    },
    // ── Sonnet 4.6 ──────────────────────────────────────────────────
    CatalogEntry {
        prefix: "claude-sonnet-4-6",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Adaptive,
            context_window: 1_000_000,
        },
    },
    // ── Haiku 4.5 (correct date: 20251001) ──────────────────────────
    CatalogEntry {
        prefix: "claude-haiku-4-5",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Extended {
                default_budget: 10_240,
            },
            context_window: 200_000,
        },
    },
    // Also match the dot-separated variant (claude-haiku-4.5)
    CatalogEntry {
        prefix: "claude-haiku-4.5",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Extended {
                default_budget: 10_240,
            },
            context_window: 200_000,
        },
    },
    // ── Sonnet 4.5 (legacy, extended thinking) ──────────────────────
    CatalogEntry {
        prefix: "claude-sonnet-4-5",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Extended {
                default_budget: 10_240,
            },
            context_window: 200_000,
        },
    },
    CatalogEntry {
        prefix: "claude-sonnet-4.5",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Extended {
                default_budget: 10_240,
            },
            context_window: 200_000,
        },
    },
    // ── Opus 4.0 / 4.1 / 4.5 (legacy, extended thinking) ──────────
    // These all start with "claude-opus-4" but are NOT 4.6 (matched above).
    CatalogEntry {
        prefix: "claude-opus-4",
        capabilities: ModelCapabilities {
            max_output_tokens: 65_536,
            thinking: ThinkingSupport::Extended {
                default_budget: 10_240,
            },
            context_window: 200_000,
        },
    },
    // ── Sonnet 4.0 (no thinking) ────────────────────────────────────
    CatalogEntry {
        prefix: "claude-sonnet-4",
        capabilities: ModelCapabilities {
            max_output_tokens: 8_192,
            thinking: ThinkingSupport::None,
            context_window: 200_000,
        },
    },
];

/// Look up capabilities for a model by name.
///
/// The model name is matched against the catalog using prefix matching.
/// Provider prefixes (e.g. `anthropic/`, `openrouter/anthropic/`) are
/// stripped automatically before matching.
///
/// Returns conservative defaults for unknown models.
///
/// # Examples
///
/// ```rust
/// # use spacebot::llm::capabilities::{get_capabilities, ThinkingSupport};
/// // Direct model name
/// let caps = get_capabilities("claude-opus-4-6");
/// assert_eq!(caps.max_output_tokens, 131_072);
/// assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
///
/// // With provider prefix
/// let caps = get_capabilities("anthropic/claude-sonnet-4-6");
/// assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
///
/// // Unknown model gets safe defaults
/// let caps = get_capabilities("gpt-4o");
/// assert!(matches!(caps.thinking, ThinkingSupport::None));
/// ```
pub fn get_capabilities(model_name: &str) -> ModelCapabilities {
    let bare = strip_provider_prefix(model_name);
    for entry in CATALOG {
        if bare.starts_with(entry.prefix) {
            return entry.capabilities.clone();
        }
    }
    DEFAULT_CAPABILITIES
}

/// Returns true if the model supports any form of thinking (extended or adaptive).
///
/// Convenience helper for code that just needs to know "does this model think?"
/// without caring about the specific variant.
pub fn supports_thinking(model_name: &str) -> bool {
    !matches!(get_capabilities(model_name).thinking, ThinkingSupport::None)
}

/// Strip provider routing prefixes to get the bare model name.
///
/// Handles both single-level (`anthropic/claude-...`) and double-level
/// (`openrouter/anthropic/claude-...`) prefixes.
fn strip_provider_prefix(model_name: &str) -> &str {
    // Known provider prefixes — strip the outermost, then check for a nested one.
    const PROVIDERS: &[&str] = &[
        "openrouter/",
        "anthropic/",
        "openai/",
        "zhipu/",
        "groq/",
        "together/",
        "fireworks/",
        "deepseek/",
        "xai/",
        "mistral/",
        "opencode-zen/",
    ];

    let mut name = model_name;
    // Strip up to two levels of provider prefix (e.g. openrouter/anthropic/)
    for _ in 0..2 {
        let mut matched = false;
        for prefix in PROVIDERS {
            if let Some(rest) = name.strip_prefix(prefix) {
                name = rest;
                matched = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_4_6_adaptive_thinking() {
        let caps = get_capabilities("claude-opus-4-6");
        assert_eq!(caps.max_output_tokens, 131_072);
        assert_eq!(caps.context_window, 1_000_000);
        assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
    }

    #[test]
    fn sonnet_4_6_adaptive_thinking() {
        let caps = get_capabilities("claude-sonnet-4-6");
        assert_eq!(caps.max_output_tokens, 65_536);
        assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
    }

    #[test]
    fn haiku_4_5_extended_thinking() {
        let caps = get_capabilities("claude-haiku-4-5-20251001");
        assert_eq!(caps.max_output_tokens, 65_536);
        assert!(matches!(
            caps.thinking,
            ThinkingSupport::Extended { default_budget: 10_240 }
        ));
    }

    #[test]
    fn sonnet_4_5_legacy_extended_thinking() {
        let caps = get_capabilities("claude-sonnet-4-5-20250514");
        assert_eq!(caps.max_output_tokens, 65_536);
        assert!(matches!(caps.thinking, ThinkingSupport::Extended { .. }));
    }

    #[test]
    fn sonnet_4_no_thinking() {
        let caps = get_capabilities("claude-sonnet-4-20250514");
        assert_eq!(caps.max_output_tokens, 8_192);
        assert!(matches!(caps.thinking, ThinkingSupport::None));
    }

    #[test]
    fn opus_4_legacy_extended_thinking() {
        let caps = get_capabilities("claude-opus-4-20250514");
        assert_eq!(caps.max_output_tokens, 65_536);
        assert!(matches!(caps.thinking, ThinkingSupport::Extended { .. }));
    }

    #[test]
    fn provider_prefix_stripped() {
        let caps = get_capabilities("anthropic/claude-opus-4-6");
        assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));

        let caps = get_capabilities("openrouter/anthropic/claude-opus-4-6");
        assert!(matches!(caps.thinking, ThinkingSupport::Adaptive));
    }

    #[test]
    fn unknown_model_defaults() {
        let caps = get_capabilities("gpt-4o");
        assert_eq!(caps.max_output_tokens, 8_192);
        assert!(matches!(caps.thinking, ThinkingSupport::None));
    }

    #[test]
    fn dot_separated_variants() {
        let caps = get_capabilities("claude-haiku-4.5-20251001");
        assert!(matches!(caps.thinking, ThinkingSupport::Extended { .. }));

        let caps = get_capabilities("claude-sonnet-4.5-20250514");
        assert!(matches!(caps.thinking, ThinkingSupport::Extended { .. }));
    }

    #[test]
    fn supports_thinking_helper() {
        assert!(supports_thinking("claude-opus-4-6"));
        assert!(supports_thinking("claude-sonnet-4-6"));
        assert!(supports_thinking("claude-haiku-4-5-20251001"));
        assert!(supports_thinking("claude-sonnet-4-5-20250514"));
        assert!(!supports_thinking("claude-sonnet-4-20250514"));
        assert!(!supports_thinking("gpt-4o"));
    }
}
