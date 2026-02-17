//! LLM provider management and routing.

pub mod manager;
pub mod model;
pub mod providers;
pub mod routing;

pub use manager::{AnthropicAuth, LlmManager};
pub use model::SpacebotModel;
pub use routing::RoutingConfig;
