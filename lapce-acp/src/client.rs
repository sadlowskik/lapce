//! An ACP client: spawn an agent, drive a turn, stream what it says back.
//!
//! Structured after `lapce_rpc::stdio`, which already does newline-delimited
//! JSON over a reader/writer thread pair -- the same framing ACP requires. It is
//! not reused directly because Lapce's `write_msg` omits the `"jsonrpc": "2.0"`
//! field that ACP mandates on every message.
//!
//! Threading, and why it has to be this shape:
//!
//! - a **reader thread** owns stdout. Responses wake whoever is blocked on that
//!   id; notifications go to the caller's channel.
//! - the **calling thread** blocks on a response. It cannot deadlock against
//!   itself, because a different thread does the reading.
//! - a **stderr thread** drains the agent's log. Without it a chatty agent fills
//!   the pipe buffer and blocks forever, which looks exactly like a hang.
//!
//! That last one is not hypothetical: an agent writing progress to stderr with
//! nobody draining it will stop mid-turn with no error anywhere.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use parking_lot::Mutex;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::protocol::*;

/// What the client emits while a turn runs.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The agent narrating: a message chunk, a tool call, a plan.
    Update(SessionNotification),
    /// A line the agent wrote to stderr. Its log, not its answer.
    Log(String),
    /// The agent process ended. Nothing more will arrive.
    Exited(Option<i32>),
}

/// A running agent subprocess.
pub struct AcpClient {
    child: Mutex<Child>,
    stdin: Arc<Mutex<ChildStdin>>,
    next_id: AtomicU64,
    pending: Arc<Mutex<HashMap<u64, Sender<Result<Value, RpcErr>>>>>,
    pub events: Receiver<AgentEvent>,
    /// Agent identity from `initialize`, once the handshake has run.
    pub agent_info: Mutex<Option<Implementation>>,
    /// What to do when the agent asks permission. Refuses by default.
    pub permission_policy: Arc<Mutex<PermissionPolicy>>,
}

#[derive(Debug, Clone)]
pub struct RpcErr {
    pub code: i64,
    pub message: String,
}

impl std::fmt::Display for RpcErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "agent error {}: {}", self.code, self.message)
    }
}

impl AcpClient {
    /// Spawn `program args...` with `cwd` as its working directory.
    ///
    /// `env` is applied on top of the inherited environment -- an agent usually
    /// needs an API key or a PYTHONPATH, and inheriting nothing would break more
    /// than it protects.
    pub fn spawn(
        program: &str,
        args: &[String],
        cwd: &str,
        env: &[(String, String)],
    ) -> Result<Arc<Self>> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("could not start agent `{program}`: {e}"))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;

        // Shared so the reader thread can *answer* agent->client requests. It
        // has to: an unanswered request leaves the agent blocked forever, and
        // from outside that is indistinguishable from a slow model.
        let stdin = Arc::new(Mutex::new(stdin));
        let policy = Arc::new(Mutex::new(PermissionPolicy::default()));

        let (tx, events) = unbounded();
        let pending: Arc<Mutex<HashMap<u64, Sender<Result<Value, RpcErr>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Reader: responses resolve waiters, notifications become events,
        // requests get answered.
        {
            let pending = pending.clone();
            let tx = tx.clone();
            let stdin = stdin.clone();
            let policy = policy.clone();
            thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let msg: Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("acp: unparseable line: {e}");
                            continue;
                        }
                    };

                    // `method` first: a *request* carries both `method` and
                    // `id`, so keying on `id` alone would mistake one for a
                    // response and silently drop it -- hanging the agent.
                    if let Some(method) = msg.get("method").and_then(|m| m.as_str())
                    {
                        if let Some(id) = msg.get("id").cloned() {
                            let body = answer_request(
                                method,
                                msg.get("params"),
                                *policy.lock(),
                                &tx,
                            );
                            let mut reply = serde_json::Map::new();
                            reply.insert("jsonrpc".into(), json!("2.0"));
                            reply.insert("id".into(), id);
                            for (k, v) in body {
                                reply.insert(k, v);
                            }
                            let line = format!("{}\n", Value::Object(reply));
                            let mut out = stdin.lock();
                            let _ = out.write_all(line.as_bytes());
                            let _ = out.flush();
                        } else if method == "session/update" {
                            match serde_json::from_value::<SessionNotification>(
                                msg.get("params").cloned().unwrap_or(Value::Null),
                            ) {
                                Ok(n) => {
                                    let _ = tx.send(AgentEvent::Update(n));
                                }
                                Err(e) => {
                                    tracing::warn!("acp: bad session/update: {e}")
                                }
                            }
                        }
                    } else if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                        let waiter = pending.lock().remove(&id);
                        if let Some(w) = waiter {
                            let outcome = match msg.get("error") {
                                Some(e) if !e.is_null() => Err(RpcErr {
                                    code: e.get("code").and_then(|c| c.as_i64()).unwrap_or(-1),
                                    message: e
                                        .get("message")
                                        .and_then(|m| m.as_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                }),
                                _ => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
                            };
                            let _ = w.send(outcome);
                        }
                    }
                }

                // Stream closed: nothing will ever answer an outstanding call.
                let stranded: Vec<_> = pending.lock().drain().map(|(_, w)| w).collect();
                for w in stranded {
                    let _ = w.send(Err(RpcErr {
                        code: -32603,
                        message: "agent closed the connection".into(),
                    }));
                }
                let _ = tx.send(AgentEvent::Exited(None));
            });
        }

        // Stderr: drained so a chatty agent cannot fill the pipe and hang.
        {
            let tx = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    let _ = tx.send(AgentEvent::Log(line));
                }
            });
        }

        Ok(Arc::new(Self {
            child: Mutex::new(child),
            stdin,
            next_id: AtomicU64::new(1),
            pending,
            events,
            agent_info: Mutex::new(None),
            permission_policy: policy,
        }))
    }

    fn send_raw(&self, msg: &Value) -> Result<()> {
        let line = format!("{}\n", serde_json::to_string(msg)?);
        debug_assert!(
            !line[..line.len() - 1].contains('\n'),
            "framing violation: embedded newline"
        );
        let mut stdin = self.stdin.lock();
        stdin.write_all(line.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }

    /// Call a method and block for its reply.
    pub fn request<P: Serialize, R: DeserializeOwned>(
        &self,
        method: &str,
        params: P,
        timeout: Duration,
    ) -> Result<R> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = bounded(1);
        self.pending.lock().insert(id, tx);

        self.send_raw(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))?;

        match rx.recv_timeout(timeout) {
            Ok(Ok(v)) => Ok(serde_json::from_value(v)?),
            Ok(Err(e)) => bail!(e),
            Err(_) => {
                self.pending.lock().remove(&id);
                bail!("timed out after {timeout:?} waiting for `{method}`")
            }
        }
    }

    pub fn notify<P: Serialize>(&self, method: &str, params: P) -> Result<()> {
        self.send_raw(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))
    }

    // ------------------------------------------------------------- protocol

    pub fn initialize(&self, timeout: Duration) -> Result<InitializeResult> {
        let result: InitializeResult = self.request(
            "initialize",
            InitializeParams {
                protocol_version: PROTOCOL_VERSION,
                // Both false until the matching handlers exist. Advertising a
                // capability the client cannot service is worse than lacking it:
                // the agent will call it and the turn will stall.
                client_capabilities: ClientCapabilities {
                    fs: Some(FsCapability {
                        read_text_file: false,
                        write_text_file: false,
                    }),
                    terminal: Some(false),
                },
                client_info: Implementation {
                    name: "lapce".into(),
                    title: Some("Lapce".into()),
                    version: Some(env!("CARGO_PKG_VERSION").into()),
                },
            },
            timeout,
        )?;

        if result.protocol_version > PROTOCOL_VERSION {
            bail!(
                "agent speaks protocol v{}, this client implements v{}",
                result.protocol_version,
                PROTOCOL_VERSION
            );
        }
        *self.agent_info.lock() = result.agent_info.clone();
        Ok(result)
    }

    pub fn new_session(&self, cwd: &str, timeout: Duration) -> Result<String> {
        let r: NewSessionResult = self.request(
            "session/new",
            NewSessionParams {
                cwd: cwd.to_string(),
                mcp_servers: vec![],
            },
            timeout,
        )?;
        Ok(r.session_id)
    }

    pub fn prompt(
        &self,
        session_id: &str,
        text: &str,
        timeout: Duration,
    ) -> Result<StopReason> {
        let r: PromptResult = self.request(
            "session/prompt",
            PromptParams {
                session_id: session_id.to_string(),
                prompt: vec![ContentBlock::text(text)],
            },
            timeout,
        )?;
        Ok(r.stop_reason)
    }

    /// Interrupt a running turn. A notification, so it does not wait for the
    /// turn it is interrupting -- which is the entire point.
    pub fn cancel(&self, session_id: &str) -> Result<()> {
        self.notify(
            "session/cancel",
            CancelParams {
                session_id: session_id.to_string(),
            },
        )
    }

    pub fn shutdown(&self) {
        let mut child = self.child.lock();
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        let mut child = self.child.lock();
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Build the body of a reply to an agent->client request.
///
/// Returns the `result` or `error` member as key/value pairs. Every branch
/// answers something: a request left unanswered blocks the agent for as long as
/// it is willing to wait, and no error ever surfaces.
fn answer_request(
    method: &str,
    params: Option<&Value>,
    policy: PermissionPolicy,
    tx: &Sender<AgentEvent>,
) -> Vec<(String, Value)> {
    match method {
        "session/request_permission" => {
            let parsed: Option<RequestPermissionParams> = params
                .cloned()
                .and_then(|p| serde_json::from_value(p).ok());

            let what = parsed
                .as_ref()
                .and_then(|p| p.tool_call.as_ref())
                .and_then(|t| t.title.clone())
                .unwrap_or_else(|| "an operation".to_string());

            // Pick from the options the agent offered rather than inventing an
            // id: option ids are agent-defined and a guess would be rejected.
            let wanted_allow = matches!(policy, PermissionPolicy::AllowOnce);
            let chosen = parsed.as_ref().and_then(|p| {
                p.options
                    .iter()
                    .find(|o| {
                        o.kind.map(|k| k.is_allow()).unwrap_or(false) == wanted_allow
                    })
                    .or_else(|| p.options.first())
            });

            match chosen {
                Some(option) => {
                    let granted =
                        option.kind.map(|k| k.is_allow()).unwrap_or(false);
                    let _ = tx.send(AgentEvent::Log(format!(
                        "permission {} for {what}",
                        if granted { "granted" } else { "refused" }
                    )));
                    vec![(
                        "result".into(),
                        json!({
                            "outcome": {
                                "outcome": "selected",
                                "optionId": option.option_id,
                            }
                        }),
                    )]
                }
                None => {
                    // No options offered, so nothing can be selected. Cancelled
                    // is the spec's answer for "no choice was made".
                    let _ = tx.send(AgentEvent::Log(format!(
                        "permission request for {what} had no options; cancelled"
                    )));
                    vec![(
                        "result".into(),
                        json!({ "outcome": { "outcome": "cancelled" } }),
                    )]
                }
            }
        }
        // Capabilities for these are advertised false, so a conforming agent
        // should not ask -- but answering with an error beats not answering.
        _ => vec![(
            "error".into(),
            json!({
                "code": -32601,
                "message": format!("{method} is not supported by this client"),
            }),
        )],
    }
}
