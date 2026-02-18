//! LLM provider management and routing.

pub mod capabilities;
pub mod manager;
pub mod model;
pub mod providers;
pub mod routing;

pub use manager::{AnthropicAuth, LlmManager};
pub use model::SpacebotModel;
pub use routing::RoutingConfig;

/// Truncate a string for log output, appending an ellipsis and byte count if truncated.
///
/// Use this for debug-level logging of payloads, prompts, and API responses
/// to keep logs scannable without losing context entirely.
///
/// ```
/// assert_eq!(truncate_for_log("short", 100), "short");
/// assert_eq!(truncate_for_log("hello world", 5), "hello... [11 bytes total]");
/// ```
pub fn truncate_for_log(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find a valid char boundary at or before max_len
        let end = s.floor_char_boundary(max_len);
        format!("{}... [{} bytes total]", &s[..end], s.len())
    }
}
