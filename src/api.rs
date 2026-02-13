//! HTTP API server for the Spacebot control interface.
//!
//! Serves the embedded frontend assets and provides a JSON API for
//! managing agents, viewing status, and interacting with the system.
//! Includes an SSE endpoint for realtime event streaming.

mod server;
mod state;

pub use server::start_http_server;
pub use state::{ApiEvent, ApiState};
