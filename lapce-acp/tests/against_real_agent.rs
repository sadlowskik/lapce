//! Drive a real ACP agent end to end.
//!
//! The agent is Knossos in retrieval-only mode, which needs no model, no API key
//! and no network -- so this exercises the whole protocol deterministically:
//! handshake, session, a turn, streamed updates, and a stop reason.
//!
//! Mocking the agent would test the client against my own understanding of the
//! protocol, which is exactly the thing most likely to be wrong. A real
//! subprocess catches framing and threading mistakes a mock cannot.
//!
//! Skipped unless both are set, so the suite still passes on a machine without
//! a Python checkout:
//!
//! ```text
//! LAPCE_ACP_TEST_PYTHON=C:/path/to/python.exe
//! LAPCE_ACP_TEST_AGENT_CWD=C:/path/to/daedalus/model
//! ```

use std::{env, time::Duration};

use lapce_acp::{AcpClient, AgentEvent, SessionUpdate, StopReason};

const TIMEOUT: Duration = Duration::from_secs(120);

/// Returns `None` (and the test no-ops) when the environment is not configured.
fn spawn_agent() -> Option<std::sync::Arc<AcpClient>> {
    let python = env::var("LAPCE_ACP_TEST_PYTHON").ok()?;
    let cwd = env::var("LAPCE_ACP_TEST_AGENT_CWD").ok()?;

    let args: Vec<String> = ["-m", "knossos", "--engine", "retrieval"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let env_vars = vec![("PYTHONPATH".to_string(), cwd.clone())];

    Some(
        AcpClient::spawn(&python, &args, &cwd, &env_vars)
            .expect("agent should start"),
    )
}

fn agent_cwd() -> String {
    env::var("LAPCE_ACP_TEST_AGENT_CWD").unwrap()
}

#[test]
fn handshake_reports_the_agent_identity() {
    let Some(client) = spawn_agent() else {
        eprintln!("skipped: LAPCE_ACP_TEST_* not set");
        return;
    };
    let init = client.initialize(TIMEOUT).expect("initialize");

    assert_eq!(init.protocol_version, lapce_acp::PROTOCOL_VERSION);
    let info = init.agent_info.expect("agent should identify itself");
    assert!(!info.name.is_empty());
    client.shutdown();
}

#[test]
fn a_full_turn_streams_updates_and_ends_cleanly() {
    let Some(client) = spawn_agent() else {
        eprintln!("skipped: LAPCE_ACP_TEST_* not set");
        return;
    };
    let cwd = agent_cwd();
    client.initialize(TIMEOUT).expect("initialize");
    let session = client.new_session(&cwd, TIMEOUT).expect("session/new");
    assert!(!session.is_empty());

    // Collect on another thread: the prompt call blocks until the turn ends,
    // so draining events inline would deadlock.
    let events = client.events.clone();
    let collector = std::thread::spawn(move || {
        let mut messages = String::new();
        let mut tool_calls = 0usize;
        let mut located = 0usize;
        for event in events {
            match event {
                AgentEvent::Update(n) => match n.update {
                    SessionUpdate::AgentMessageChunk { content } => {
                        messages.push_str(content.as_text().unwrap_or(""))
                    }
                    SessionUpdate::ToolCall(_) => tool_calls += 1,
                    SessionUpdate::ToolCallUpdate(u) => located += u.locations.len(),
                    _ => {}
                },
                AgentEvent::Exited(_) => break,
                AgentEvent::Log(_) => {}
            }
        }
        (messages, tool_calls, located)
    });

    let stop = client
        .prompt(&session, "where is the halting probability computed?", TIMEOUT)
        .expect("session/prompt");
    assert_eq!(stop, StopReason::EndTurn);
    assert!(stop.is_complete());

    client.shutdown();
    let (messages, tool_calls, located) = collector.join().unwrap();

    assert!(!messages.trim().is_empty(), "expected some reply text");
    assert!(tool_calls >= 1, "retrieval should surface as a tool call");
    assert!(located >= 1, "tool call should report file locations");
}

#[test]
fn a_prompt_on_an_unknown_session_is_an_error_not_a_panic() {
    let Some(client) = spawn_agent() else {
        eprintln!("skipped: LAPCE_ACP_TEST_* not set");
        return;
    };
    client.initialize(TIMEOUT).expect("initialize");

    let err = client
        .prompt("sess_does_not_exist", "hello", TIMEOUT)
        .expect_err("unknown session must be rejected");
    assert!(
        err.to_string().contains("session"),
        "unhelpful error: {err}"
    );

    // and the connection survives it
    let cwd = agent_cwd();
    client.new_session(&cwd, TIMEOUT).expect("still usable");
    client.shutdown();
}

#[test]
fn a_dead_agent_errors_instead_of_hanging() {
    // A crashed agent must not leave the UI thread waiting forever. If the
    // stranded-waiter path in the reader thread regresses, this test does not
    // fail -- it hangs until the harness timeout, which is the intended signal.
    //
    // The agent is killed *before* prompting rather than mid-turn: with the
    // retrieval engine a turn completes in well under a second, so racing a
    // sleep against it tested nothing reliable. An earlier version of this test
    // passed only because the turn beat the kill.
    let Some(client) = spawn_agent() else {
        eprintln!("skipped: LAPCE_ACP_TEST_* not set");
        return;
    };
    client.initialize(TIMEOUT).expect("initialize");
    let cwd = agent_cwd();
    let session = client.new_session(&cwd, TIMEOUT).unwrap();

    client.shutdown();

    let result = client.prompt(&session, "a question", Duration::from_secs(10));
    assert!(result.is_err(), "a dead agent must not hang the caller");
}
