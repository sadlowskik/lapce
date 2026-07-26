//! Editor-side state for an ACP agent session.
//!
//! [`lapce_acp`] is headless: it spawns a process and emits [`AgentEvent`]s from
//! a background thread. This module is the bridge to the UI -- it turns that
//! stream into Floem signals a panel can render, and owns the session lifecycle.
//!
//! Two threading rules shape everything here, and both are enforced by the type
//! system rather than by care:
//!
//! 1. **`AgentData` never crosses a thread boundary.** It holds `Rc<CommonData>`
//!    and Floem signals, none of which are `Send`. Work that must happen off the
//!    UI thread is given a `create_ext_action` callback and plain owned data;
//!    the callback runs back on the UI thread when the work finishes. This is
//!    the same pattern `debug.rs` uses.
//! 2. **Blocking calls never run on the UI thread.** `session/prompt` blocks
//!    until the entire turn ends, which on a local model is minutes. Running it
//!    inline would freeze the editor for the duration.
//!
//! The event stream needs a third piece: `create_ext_action` is one-shot, so a
//! continuous stream uses `create_signal_from_channel`. Floem compiles that
//! against `std::sync::mpsc` here (its `crossbeam` feature is off), while
//! `lapce-acp` uses crossbeam channels -- so [`AgentData::attach`] runs a small
//! forwarding thread between the two.

use std::{
    path::PathBuf,
    rc::Rc,
    sync::{Arc, mpsc},
    time::Duration,
};

use floem::{
    ext_event::{create_ext_action, create_signal_from_channel},
    reactive::{
        RwSignal, Scope, SignalGet, SignalUpdate, SignalWith, create_effect,
    },
};
use lapce_acp::{AcpClient, AgentEvent, SessionUpdate, StopReason, ToolCallStatus};

use crate::{
    command::{CommandExecuted, CommandKind},
    editor::EditorData,
    keypress::{KeyPressFocus, condition::Condition},
    main_split::MainSplitData,
    window_tab::CommonData,
};

/// Generous: an agent may be a cold interpreter importing a large runtime.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
/// A turn can legitimately take minutes on a local model.
const TURN_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentStatus {
    Stopped,
    Connecting,
    Ready,
    Working,
    /// Shown to the user verbatim.
    Failed(String),
}

impl AgentStatus {
    pub fn describe(&self) -> String {
        match self {
            AgentStatus::Stopped => "not started".into(),
            AgentStatus::Connecting => "connecting…".into(),
            AgentStatus::Ready => "ready".into(),
            AgentStatus::Working => "working…".into(),
            AgentStatus::Failed(e) => format!("failed: {e}"),
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(self, AgentStatus::Connecting | AgentStatus::Working)
    }
}

/// One row in the transcript.
#[derive(Clone, Debug)]
pub enum Entry {
    User(String),
    /// The answer. Streamed chunks append to the trailing entry rather than
    /// adding rows, or a streamed reply becomes one row per token.
    Agent(String),
    /// Reasoning. Separate from the answer and rendered dimmed -- a model's
    /// scratchpad is not what was asked for.
    Thought(String),
    Tool(ToolEntry),
    /// Status and errors: things the user should see that the agent did not say.
    Notice(String),
}

#[derive(Clone, Debug)]
pub struct ToolEntry {
    pub id: String,
    pub title: String,
    pub status: Option<ToolCallStatus>,
    /// Absolute paths with 0-based lines, ready for the editor. These are what
    /// make a retrieval auditable: you can open what the agent actually read.
    pub locations: Vec<(PathBuf, u32)>,
}

/// What the connect thread hands back to the UI thread.
type Connected = Result<(Arc<AcpClient>, String, String), String>;

#[derive(Clone)]
pub struct AgentData {
    pub scope: Scope,
    pub entries: RwSignal<im::Vector<Entry>>,
    pub status: RwSignal<AgentStatus>,
    pub session_id: RwSignal<Option<String>>,
    pub agent_name: RwSignal<Option<String>>,
    client: RwSignal<Option<Arc<AcpClient>>>,
    /// The prompt box. A local editor, the same mechanism the search panel uses,
    /// so it gets Lapce's own editing, keymaps and modal behaviour for free.
    pub input: EditorData,
    pub common: Rc<CommonData>,
}

/// `KeyPressFocus` requires `Debug`, but neither `Rc<CommonData>` nor
/// `EditorData` implements it, and deriving would be noise regardless -- the
/// transcript can be thousands of lines.
impl std::fmt::Debug for AgentData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentData")
            .field("status", &self.status.get_untracked())
            .field("session", &self.session_id.get_untracked())
            .finish_non_exhaustive()
    }
}

impl KeyPressFocus for AgentData {
    fn get_mode(&self) -> lapce_core::mode::Mode {
        lapce_core::mode::Mode::Insert
    }

    fn check_condition(&self, condition: Condition) -> bool {
        matches!(condition, Condition::PanelFocus)
    }

    fn run_command(
        &self,
        command: &crate::command::LapceCommand,
        count: Option<usize>,
        mods: floem::keyboard::Modifiers,
    ) -> CommandExecuted {
        match &command.kind {
            CommandKind::Edit(_)
            | CommandKind::Move(_)
            | CommandKind::MultiSelection(_) => {
                self.input.run_command(command, count, mods)
            }
            _ => CommandExecuted::No,
        }
    }

    fn receive_char(&self, c: &str) {
        self.input.receive_char(c);
    }
}

impl AgentData {
    pub fn new(cx: Scope, main_split: MainSplitData, common: Rc<CommonData>) -> Self {
        let input = main_split.editors.make_local(cx, common.clone());
        Self {
            scope: cx,
            input,
            entries: cx.create_rw_signal(im::Vector::new()),
            status: cx.create_rw_signal(AgentStatus::Stopped),
            session_id: cx.create_rw_signal(None),
            agent_name: cx.create_rw_signal(None),
            client: cx.create_rw_signal(None),
            common,
        }
    }

    fn push(&self, entry: Entry) {
        self.entries.update(|e| e.push_back(entry));
    }

    /// Append streamed text to the trailing entry when it is the same kind,
    /// otherwise start a new one.
    fn append_streamed(&self, text: &str, thought: bool) {
        self.entries.update(|entries| {
            match entries.back_mut() {
                Some(Entry::Agent(s)) if !thought => s.push_str(text),
                Some(Entry::Thought(s)) if thought => s.push_str(text),
                _ => entries.push_back(if thought {
                    Entry::Thought(text.to_string())
                } else {
                    Entry::Agent(text.to_string())
                }),
            }
        });
    }

    /// Start an agent and open a session against `cwd`.
    ///
    /// Returns immediately. Spawning, the handshake and `session/new` all run on
    /// a worker thread, because a cold agent takes seconds to answer.
    pub fn start(&self, program: String, args: Vec<String>, cwd: PathBuf) {
        if self.status.get_untracked().is_busy() {
            return;
        }
        self.status.set(AgentStatus::Connecting);
        self.entries.update(|e| e.clear());
        self.push(Entry::Notice(format!("starting {program}…")));

        let this = self.clone();
        let fallback_name = program.clone();
        // Built here, on the UI thread, so the worker carries only this callback
        // and owned data -- never `AgentData`, which is not Send.
        let send = create_ext_action(self.scope, move |result: Connected| {
            match result {
                Ok((client, session, name)) => this.attach(client, session, name),
                Err(e) => {
                    this.push(Entry::Notice(e.clone()));
                    this.status.set(AgentStatus::Failed(e));
                }
            }
        });

        let cwd_str = cwd.to_string_lossy().to_string();
        std::thread::spawn(move || {
            let result = (|| -> Connected {
                let client = AcpClient::spawn(&program, &args, &cwd_str, &[])
                    .map_err(|e| e.to_string())?;
                let init = client
                    .initialize(CONNECT_TIMEOUT)
                    .map_err(|e| format!("handshake: {e}"))?;
                let session = client
                    .new_session(&cwd_str, CONNECT_TIMEOUT)
                    .map_err(|e| format!("session: {e}"))?;
                let name = init
                    .agent_info
                    .as_ref()
                    .map(|i| i.name.clone())
                    .unwrap_or(fallback_name);
                Ok((client, session, name))
            })();
            send(result);
        });
    }

    /// Adopt a connected client. Runs on the UI thread.
    fn attach(&self, client: Arc<AcpClient>, session: String, name: String) {
        // Floem's create_signal_from_channel is compiled against std::sync::mpsc
        // here, so bridge the agent's crossbeam receiver across.
        let (tx, rx) = mpsc::channel::<AgentEvent>();
        let events = client.events.clone();
        std::thread::spawn(move || {
            for event in events {
                if tx.send(event).is_err() {
                    break;
                }
            }
        });

        let this = self.clone();
        let signal = create_signal_from_channel(rx);
        create_effect(move |_| {
            if let Some(event) = signal.get() {
                this.apply(event);
            }
        });

        self.client.set(Some(client));
        self.session_id.set(Some(session));
        self.agent_name.set(Some(name));
        self.status.set(AgentStatus::Ready);
        self.push(Entry::Notice("connected".into()));
    }

    /// Send a prompt. No-op unless a session is open and idle.
    pub fn send(&self, text: String) {
        let (Some(client), Some(session)) =
            (self.client.get_untracked(), self.session_id.get_untracked())
        else {
            return;
        };
        if self.status.get_untracked() != AgentStatus::Ready
            || text.trim().is_empty()
        {
            return;
        }

        self.push(Entry::User(text.clone()));
        self.status.set(AgentStatus::Working);

        let this = self.clone();
        let send = create_ext_action(
            self.scope,
            move |result: Result<StopReason, String>| match result {
                Ok(stop) => {
                    // Only end_turn means the answer is complete. Saying so
                    // matters: a reply truncated at the token limit looks
                    // identical to a finished one.
                    if !stop.is_complete() {
                        this.push(Entry::Notice(stop.describe().to_string()));
                    }
                    this.status.set(AgentStatus::Ready);
                }
                Err(e) => {
                    this.push(Entry::Notice(format!("turn failed: {e}")));
                    this.status.set(AgentStatus::Failed(e));
                }
            },
        );

        std::thread::spawn(move || {
            let result = client
                .prompt(&session, &text, TURN_TIMEOUT)
                .map_err(|e| e.to_string());
            send(result);
        });
    }

    /// Send the contents of the prompt box, then clear it.
    pub fn submit(&self) {
        let text = self.input.doc().buffer.with_untracked(|b| b.to_string());
        if text.trim().is_empty() {
            return;
        }
        self.send(text);
        self.input
            .doc()
            .reload(lapce_xi_rope::Rope::from(""), true);
    }

    /// Interrupt the running turn.
    pub fn cancel(&self) {
        if let (Some(client), Some(session)) =
            (self.client.get_untracked(), self.session_id.get_untracked())
        {
            let _ = client.cancel(&session);
        }
    }

    pub fn stop(&self) {
        if let Some(client) = self.client.get_untracked() {
            client.shutdown();
        }
        self.client.set(None);
        self.session_id.set(None);
        self.status.set(AgentStatus::Stopped);
    }

    // -------------------------------------------------------------- events

    fn apply(&self, event: AgentEvent) {
        match event {
            AgentEvent::Update(n) => match n.update {
                SessionUpdate::AgentMessageChunk { content } => {
                    if let Some(t) = content.as_text() {
                        self.append_streamed(t, false);
                    }
                }
                SessionUpdate::AgentThoughtChunk { content } => {
                    if let Some(t) = content.as_text() {
                        self.append_streamed(t, true);
                    }
                }
                SessionUpdate::ToolCall(c) => self.push(Entry::Tool(ToolEntry {
                    id: c.tool_call_id,
                    title: c.title.unwrap_or_else(|| "tool".into()),
                    status: c.status,
                    locations: to_locations(&c.locations),
                })),
                SessionUpdate::ToolCallUpdate(u) => {
                    let locations = to_locations(&u.locations);
                    self.entries.update(|entries| {
                        let found = entries.iter_mut().rev().find(
                            |e| matches!(e, Entry::Tool(t) if t.tool_id() == u.tool_call_id),
                        );
                        if let Some(Entry::Tool(t)) = found {
                            if let Some(title) = u.title.clone() {
                                t.title = title;
                            }
                            if u.status.is_some() {
                                t.status = u.status;
                            }
                            if !locations.is_empty() {
                                t.locations = locations;
                            }
                        }
                    });
                }
                // Plans and anything newer are not rendered specially yet, but
                // must not be treated as an error.
                _ => {}
            },
            // The agent's log is not its answer, so it stays out of the
            // transcript. Failures surface through status instead.
            AgentEvent::Log(line) => tracing::debug!("agent: {line}"),
            AgentEvent::Exited(code) => {
                self.push(Entry::Notice(match code {
                    Some(c) => format!("agent exited ({c})"),
                    None => "agent exited".into(),
                }));
                self.status.set(AgentStatus::Stopped);
                self.client.set(None);
                self.session_id.set(None);
            }
        }
    }
}

impl ToolEntry {
    fn tool_id(&self) -> &str {
        &self.id
    }
}

fn to_locations(locations: &[lapce_acp::ToolCallLocation]) -> Vec<(PathBuf, u32)> {
    locations
        .iter()
        .map(|l| {
            // ACP lines are 1-based; the editor's are 0-based.
            (PathBuf::from(&l.path), l.line.unwrap_or(1).saturating_sub(1))
        })
        .collect()
}
