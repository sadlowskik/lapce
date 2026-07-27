//! Wire types for the Agent Client Protocol.
//!
//! ACP (Zed Industries, August 2025) is how an editor talks to a coding agent it
//! spawns as a subprocess: JSON-RPC 2.0 over newline-delimited stdio. Claude
//! Code, Gemini CLI, Codex and Knossos all speak it, so a client here is a
//! client for all of them rather than an integration with one.
//!
//! Only the parts an editor must understand are typed. Everything else is left
//! as [`serde_json::Value`] on purpose: the protocol grows, and an unknown
//! `sessionUpdate` variant should render as "something happened" rather than
//! drop the connection. Refusing to parse a message you merely do not recognise
//! is how a client breaks against a newer agent.
//!
//! Field names are camelCase on the wire and snake_case here, via serde rename.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The protocol version this client implements.
pub const PROTOCOL_VERSION: u32 = 1;

// ---------------------------------------------------------------- initialize

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: u32,
    pub client_capabilities: ClientCapabilities,
    pub client_info: Implementation,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fs: Option<FsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal: Option<bool>,
}

/// What the *agent* may ask the editor to do on its behalf.
///
/// Claiming a capability is a promise to service the matching request, so these
/// default to false and are turned on as the handlers are implemented.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FsCapability {
    pub read_text_file: bool,
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Implementation {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: u32,
    #[serde(default)]
    pub agent_capabilities: Value,
    #[serde(default)]
    pub agent_info: Option<Implementation>,
    #[serde(default)]
    pub auth_methods: Vec<Value>,
}

// ------------------------------------------------------------------ sessions

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    /// Absolute path. The spec requires it, and agents reject relative paths.
    pub cwd: String,
    pub mcp_servers: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionResult {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    pub stop_reason: StopReason,
}

/// Why a turn ended. `EndTurn` is the only one meaning "this answer is complete";
/// the rest each need saying out loud in the UI, because a reply truncated at the
/// token limit looks identical to a finished one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
    /// Anything a newer agent introduces. Unknown is not an error.
    #[serde(other)]
    Unknown,
}

impl StopReason {
    /// Whether the turn finished normally, as opposed to being cut short.
    pub fn is_complete(self) -> bool {
        matches!(self, StopReason::EndTurn)
    }

    pub fn describe(self) -> &'static str {
        match self {
            StopReason::EndTurn => "done",
            StopReason::MaxTokens => "stopped: token limit reached",
            StopReason::MaxTurnRequests => "stopped: too many requests this turn",
            StopReason::Refusal => "the agent declined",
            StopReason::Cancelled => "cancelled",
            StopReason::Unknown => "stopped for an unrecognised reason",
        }
    }
}

// -------------------------------------------------------------- content blocks

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ResourceLink {
        uri: String,
    },
    /// Images, embedded resources and anything added later.
    #[serde(other)]
    Other,
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock::Text { text: s.into() }
    }

    /// The displayable text, if this block has any.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }
}

// ----------------------------------------------------------- session updates

/// One `session/update` notification: the agent narrating its turn.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    pub session_id: String,
    pub update: SessionUpdate,
}

/// The variants an editor renders differently.
///
/// `Unknown` is deliberate and load-bearing: ACP gains variants over time, and an
/// editor that errors on one it has not heard of stops working the day an agent
/// updates. Unknown updates are kept whole so the UI can still say something.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    #[serde(rename_all = "camelCase")]
    AgentMessageChunk { content: ContentBlock },

    /// Reasoning, not the answer. Rendered collapsed -- a model's scratchpad is
    /// not what the user asked for.
    #[serde(rename_all = "camelCase")]
    AgentThoughtChunk { content: ContentBlock },

    #[serde(rename_all = "camelCase")]
    UserMessageChunk { content: ContentBlock },

    #[serde(rename_all = "camelCase")]
    ToolCall(ToolCall),

    #[serde(rename_all = "camelCase")]
    ToolCallUpdate(ToolCallUpdate),

    #[serde(rename_all = "camelCase")]
    Plan { entries: Vec<PlanEntry> },

    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool_call_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub kind: Option<ToolKind>,
    #[serde(default)]
    pub status: Option<ToolCallStatus>,
    #[serde(default)]
    pub locations: Vec<ToolCallLocation>,
}

/// Every field but the id is optional: an update carries only what changed.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    pub tool_call_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub status: Option<ToolCallStatus>,
    #[serde(default)]
    pub locations: Vec<ToolCallLocation>,
    #[serde(default)]
    pub content: Vec<Value>,
}

/// A file the tool touched. Paths are absolute per the spec, and lines are
/// 1-based -- which is what makes these clickable in the editor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallLocation {
    pub path: String,
    #[serde(default)]
    pub line: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    SwitchMode,
    Other,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanEntry {
    pub content: String,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_an_agent_message_chunk() {
        let v: SessionNotification = serde_json::from_str(
            r#"{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk",
                "content":{"type":"text","text":"hello"}}}"#,
        )
        .unwrap();
        assert_eq!(v.session_id, "s1");
        match v.update {
            SessionUpdate::AgentMessageChunk { content } => {
                assert_eq!(content.as_text(), Some("hello"))
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn thinking_is_a_separate_variant_from_the_reply() {
        // Rendering a model's scratchpad as its answer is the bug this prevents.
        let v: SessionNotification = serde_json::from_str(
            r#"{"sessionId":"s","update":{"sessionUpdate":"agent_thought_chunk",
                "content":{"type":"text","text":"hmm"}}}"#,
        )
        .unwrap();
        assert!(matches!(v.update, SessionUpdate::AgentThoughtChunk { .. }));
    }

    #[test]
    fn tool_calls_carry_clickable_locations() {
        let v: SessionNotification = serde_json::from_str(
            r#"{"sessionId":"s","update":{"sessionUpdate":"tool_call_update",
                "toolCallId":"c1","status":"completed",
                "locations":[{"path":"/abs/moe.py","line":29}]}}"#,
        )
        .unwrap();
        match v.update {
            SessionUpdate::ToolCallUpdate(u) => {
                assert_eq!(u.status, Some(ToolCallStatus::Completed));
                assert_eq!(u.locations[0].path, "/abs/moe.py");
                assert_eq!(u.locations[0].line, Some(29));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn an_unknown_update_variant_does_not_fail_the_parse() {
        // The day an agent ships a new variant, this client must keep working.
        let v: SessionNotification = serde_json::from_str(
            r#"{"sessionId":"s","update":{"sessionUpdate":"telepathy_chunk",
                "wat":true}}"#,
        )
        .unwrap();
        assert!(matches!(v.update, SessionUpdate::Unknown));
    }

    #[test]
    fn an_unknown_stop_reason_does_not_fail_the_parse() {
        let r: PromptResult =
            serde_json::from_str(r#"{"stopReason":"ran_out_of_ideas"}"#).unwrap();
        assert_eq!(r.stop_reason, StopReason::Unknown);
        assert!(!r.stop_reason.is_complete());
    }

    #[test]
    fn only_end_turn_counts_as_complete() {
        assert!(StopReason::EndTurn.is_complete());
        for r in [
            StopReason::MaxTokens,
            StopReason::Refusal,
            StopReason::Cancelled,
            StopReason::MaxTurnRequests,
        ] {
            assert!(!r.is_complete(), "{r:?} must not read as a finished answer");
        }
    }

    #[test]
    fn prompt_params_serialise_to_camel_case() {
        let p = PromptParams {
            session_id: "s1".into(),
            prompt: vec![ContentBlock::text("hi")],
        };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["sessionId"], "s1");
        assert_eq!(v["prompt"][0]["type"], "text");
        assert_eq!(v["prompt"][0]["text"], "hi");
    }
}

// ----------------------------------------------------------- permissions

/// An agent asking before it does something consequential.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionParams {
    pub session_id: String,
    #[serde(default)]
    pub tool_call: Option<ToolCallUpdate>,
    #[serde(default)]
    pub options: Vec<PermissionOption>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    pub option_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: Option<PermissionKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
    #[serde(other)]
    Unknown,
}

impl PermissionKind {
    pub fn is_allow(self) -> bool {
        matches!(self, PermissionKind::AllowOnce | PermissionKind::AllowAlways)
    }
}

/// What the client does when an agent asks for permission.
///
/// The default is to refuse. An editor that silently grants whatever it is
/// asked is worse than one that cannot grant at all, because the user believes
/// they are being consulted. Until the panel can actually ask, refusing and
/// saying so is the honest behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionPolicy {
    #[default]
    DenyAll,
    /// Grant single-use permission automatically. For a trusted local agent.
    AllowOnce,
}

#[cfg(test)]
mod permission_tests {
    use super::*;

    #[test]
    fn parses_a_permission_request() {
        let p: RequestPermissionParams = serde_json::from_str(
            r#"{"sessionId":"s1","toolCall":{"toolCallId":"c1","title":"write file"},
                "options":[
                  {"optionId":"y","name":"Allow","kind":"allow_once"},
                  {"optionId":"n","name":"Reject","kind":"reject_once"}]}"#,
        )
        .unwrap();
        assert_eq!(p.session_id, "s1");
        assert_eq!(p.options.len(), 2);
        assert_eq!(p.options[0].kind, Some(PermissionKind::AllowOnce));
        assert!(p.options[0].kind.unwrap().is_allow());
        assert!(!p.options[1].kind.unwrap().is_allow());
    }

    #[test]
    fn an_unknown_option_kind_parses() {
        let p: RequestPermissionParams = serde_json::from_str(
            r#"{"sessionId":"s","options":[{"optionId":"x","kind":"maybe"}]}"#,
        )
        .unwrap();
        assert_eq!(p.options[0].kind, Some(PermissionKind::Unknown));
        assert!(!p.options[0].kind.unwrap().is_allow());
    }

    #[test]
    fn the_default_policy_refuses() {
        // Silently granting is worse than not being able to grant: the user
        // believes they are being consulted.
        assert_eq!(PermissionPolicy::default(), PermissionPolicy::DenyAll);
    }
}
