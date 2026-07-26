//! An [Agent Client Protocol](https://agentclientprotocol.com) client for Lapce.
//!
//! ACP is an open standard for editors to talk to coding agents they spawn as
//! subprocesses. Implementing the client side once makes Lapce work with every
//! agent that speaks it -- Claude Code, Gemini CLI, Codex, Knossos -- rather
//! than integrating with any one of them.
//!
//! This crate is deliberately headless: it spawns a process, drives a turn, and
//! emits [`AgentEvent`]s. It knows nothing about Floem, panels or views, so it
//! can be tested against a real agent without building the editor.
//!
//! ```no_run
//! use std::time::Duration;
//! use lapce_acp::{AcpClient, AgentEvent, SessionUpdate};
//!
//! # fn main() -> anyhow::Result<()> {
//! let client = AcpClient::spawn("python", &["-m".into(), "knossos".into()],
//!                               "/path/to/workspace", &[])?;
//! client.initialize(Duration::from_secs(10))?;
//! let session = client.new_session("/path/to/workspace", Duration::from_secs(60))?;
//!
//! let events = client.events.clone();
//! std::thread::spawn(move || {
//!     for event in events {
//!         if let AgentEvent::Update(n) = event {
//!             if let SessionUpdate::AgentMessageChunk { content } = n.update {
//!                 print!("{}", content.as_text().unwrap_or(""));
//!             }
//!         }
//!     }
//! });
//!
//! let stop = client.prompt(&session, "where is the router defined?",
//!                          Duration::from_secs(300))?;
//! println!("\n{}", stop.describe());
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod protocol;

pub use client::{AcpClient, AgentEvent, RpcErr};
pub use protocol::{
    ContentBlock, Implementation, InitializeResult, PlanEntry, SessionNotification,
    SessionUpdate, StopReason, ToolCall, ToolCallLocation, ToolCallStatus,
    ToolCallUpdate, ToolKind, PROTOCOL_VERSION,
};
