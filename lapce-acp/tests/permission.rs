//! An agent->client request must always be answered.
//!
//! This is the failure mode the client had: it read requests and dropped them.
//! The agent then waits for a reply that never comes, and from the outside that
//! is indistinguishable from a slow model -- no error, no log, just a turn that
//! never ends.
//!
//! The fake agent here refuses to finish its turn until its permission request
//! is answered, so if the client regresses these tests hang rather than fail.
//! That is the intended signal: a hang is what the bug looks like.
//!
//! No Python, no network, no env vars -- the agent is a shell one-liner.

use std::time::Duration;

use lapce_acp::{AcpClient, StopReason};

const TIMEOUT: Duration = Duration::from_secs(20);

/// A minimal ACP agent that asks permission before finishing.
///
/// Written as a portable script so the test needs nothing installed. It answers
/// `initialize` and `session/new`, then on `session/prompt` sends a permission
/// request and waits: it only replies to the prompt once the client answers.
const FAKE_AGENT: &str = r#"
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method, mid = msg.get("method"), msg.get("id")

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": 1, "agentInfo": {"name": "fake"},
            "agentCapabilities": {}, "authMethods": []}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "s1"}})
    elif method == "session/prompt":
        # Ask, then block until answered. No answer -> no reply -> the client
        # hangs, which is exactly the regression being guarded against.
        send({"jsonrpc": "2.0", "id": 9001, "method": "session/request_permission",
              "params": {"sessionId": "s1",
                         "toolCall": {"toolCallId": "c1", "title": "delete everything"},
                         "options": [
                             {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                             {"optionId": "no", "name": "Reject", "kind": "reject_once"}]}})
        answer = None
        for reply in sys.stdin:
            reply = reply.strip()
            if not reply:
                continue
            r = json.loads(reply)
            if r.get("id") == 9001:
                answer = r.get("result", {}).get("outcome", {})
                break
        send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "s1", "update": {"sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": json.dumps(answer)}}}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    elif method == "unsupported/thing":
        send({"jsonrpc": "2.0", "id": mid, "result": {"echo": "should not happen"}})
"#;

fn python() -> Option<String> {
    for candidate in [
        std::env::var("LAPCE_ACP_TEST_PYTHON").ok(),
        Some("python3".to_string()),
        Some("python".to_string()),
    ]
    .into_iter()
    .flatten()
    {
        if std::process::Command::new(&candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(candidate);
        }
    }
    None
}

fn spawn_fake() -> Option<std::sync::Arc<AcpClient>> {
    let py = python()?;
    let dir = std::env::temp_dir();
    let script = dir.join("lapce_acp_fake_agent.py");
    std::fs::write(&script, FAKE_AGENT).ok()?;

    AcpClient::spawn(
        &py,
        &[script.to_string_lossy().to_string()],
        &dir.to_string_lossy(),
        &[("PYTHONUNBUFFERED".to_string(), "1".to_string())],
    )
    .ok()
}

#[test]
fn a_permission_request_is_answered_so_the_turn_can_finish() {
    let Some(client) = spawn_fake() else {
        eprintln!("skipped: no python available");
        return;
    };
    client.initialize(TIMEOUT).expect("initialize");
    let session = client
        .new_session(&std::env::temp_dir().to_string_lossy(), TIMEOUT)
        .expect("session/new");

    // Hangs to the harness timeout if the client stops answering requests.
    let stop = client
        .prompt(&session, "do the thing", TIMEOUT)
        .expect("the turn must complete");
    assert_eq!(stop, StopReason::EndTurn);
    client.shutdown();
}

#[test]
fn the_default_policy_refuses_rather_than_silently_granting() {
    let Some(client) = spawn_fake() else {
        eprintln!("skipped: no python available");
        return;
    };
    client.initialize(TIMEOUT).expect("initialize");
    let session = client
        .new_session(&std::env::temp_dir().to_string_lossy(), TIMEOUT)
        .expect("session/new");

    // The fake agent echoes back whichever option the client chose.
    let events = client.events.clone();
    let collector = std::thread::spawn(move || {
        let mut text = String::new();
        for event in events {
            match event {
                lapce_acp::AgentEvent::Update(n) => {
                    if let lapce_acp::SessionUpdate::AgentMessageChunk { content } =
                        n.update
                    {
                        text.push_str(content.as_text().unwrap_or(""));
                    }
                }
                lapce_acp::AgentEvent::Exited(_) => break,
                _ => {}
            }
        }
        text
    });

    client.prompt(&session, "do the thing", TIMEOUT).expect("turn");
    client.shutdown();
    let echoed = collector.join().unwrap();

    assert!(
        echoed.contains("\"no\""),
        "the default must refuse, not grant: {echoed}"
    );
    assert!(echoed.contains("selected"));
}
