//! The agent panel: a transcript of the current ACP session.
//!
//! Four kinds of row, deliberately distinguished rather than concatenated:
//!
//!   user     what was asked
//!   agent    the answer
//!   thought  reasoning, dimmed -- a model's scratchpad is not the answer, and
//!            rendering it as one is the most common way agent UIs mislead
//!   tool     what the agent did, with the files it touched
//!
//! Tool rows list their locations as clickable `file:line` entries. That is the
//! point of surfacing retrieval as a tool call rather than hiding it: when the
//! agent answers from the wrong file, you can see that it did, and open it.

use std::rc::Rc;

use floem::{
    View,
    event::{Event, EventListener},
    keyboard::{Key, NamedKey},
    reactive::{SignalGet, SignalUpdate},
    style::CursorStyle,
    views::{Decorators, container, dyn_stack, label, scroll, stack},
};
use lapce_acp::ToolCallStatus;

use super::position::PanelPosition;
use crate::{
    agent::{AgentStatus, Entry},
    text_input::TextInputBuilder,
    window_tab::Focus,
    command::InternalCommand,
    config::color::LapceColor,
    editor::location::{EditorLocation, EditorPosition},
    window_tab::WindowTabData,
};

pub fn agent_panel(
    window_tab_data: Rc<WindowTabData>,
    _position: PanelPosition,
) -> impl View {
    let config = window_tab_data.common.config;
    let agent = window_tab_data.agent.clone();
    let status = agent.status;

    stack((
        // Status line. Always visible, because "nothing is happening" and "the
        // agent died three minutes ago" must not look the same.
        container(label(move || {
            let name = agent
                .agent_name
                .get()
                .unwrap_or_else(|| "no agent".to_string());
            format!("{name} — {}", status.get().describe())
        }))
        .style(move |s| {
            let config = config.get();
            s.padding_horiz(10.0)
                .padding_vert(6.0)
                .width_pct(100.0)
                .color(config.color(match status.get() {
                    AgentStatus::Failed(_) => LapceColor::LAPCE_ERROR,
                    _ => LapceColor::EDITOR_DIM,
                }))
                .border_bottom(1.0)
                .border_color(config.color(LapceColor::LAPCE_BORDER))
        }),
        transcript(window_tab_data.clone()).style(|s| s.size_pct(100.0, 100.0)),
        permission_bar(window_tab_data.clone()),
        prompt_box(window_tab_data.clone()),
    ))
    .style(|s| s.flex_col().size_pct(100.0, 100.0))
    .debug_name("Agent Panel")
}

fn transcript(window_tab_data: Rc<WindowTabData>) -> impl View {
    let config = window_tab_data.common.config;
    let agent = window_tab_data.agent.clone();
    let entries = agent.entries;
    let internal_command = window_tab_data.common.internal_command;

    container(
        scroll(
            dyn_stack(
                move || entries.get().into_iter().enumerate(),
                |(i, _)| *i,
                move |(_, entry)| {
                    entry_row(entry, config, internal_command)
                        .style(|s| s.width_pct(100.0))
                },
            )
            .style(|s| s.flex_col().width_pct(100.0).padding_vert(4.0)),
        )
        .style(|s| s.absolute().size_pct(100.0, 100.0)),
    )
    .style(|s| s.size_pct(100.0, 100.0))
}

fn entry_row(
    entry: Entry,
    config: floem::reactive::ReadSignal<std::sync::Arc<crate::config::LapceConfig>>,
    internal_command: crate::listener::Listener<InternalCommand>,
) -> Box<dyn View> {
    match entry {
        Entry::User(text) => Box::new(
            label(move || text.clone()).style(move |s| {
                let config = config.get();
                s.padding_horiz(10.0)
                    .padding_vert(6.0)
                    .width_pct(100.0)
                    .color(config.color(LapceColor::EDITOR_FOREGROUND))
                    .background(config.color(LapceColor::PANEL_CURRENT_BACKGROUND))
            }),
        ),
        Entry::Agent(text) => Box::new(label(move || text.clone()).style(move |s| {
            s.padding_horiz(10.0)
                .padding_vert(6.0)
                .width_pct(100.0)
                .color(config.get().color(LapceColor::EDITOR_FOREGROUND))
        })),
        Entry::Thought(text) => Box::new(
            // Dimmed and truncated: visible enough to know it happened, quiet
            // enough not to be mistaken for the answer.
            label(move || {
                let t = text.trim();
                if t.chars().count() > 160 {
                    format!("{}…", t.chars().take(160).collect::<String>())
                } else {
                    t.to_string()
                }
            })
            .style(move |s| {
                s.padding_horiz(10.0)
                    .padding_vert(4.0)
                    .width_pct(100.0)
                    .font_size(config.get().ui.font_size() as f32 - 1.0)
                    .color(config.get().color(LapceColor::EDITOR_DIM))
            }),
        ),
        Entry::Notice(text) => Box::new(label(move || text.clone()).style(move |s| {
            s.padding_horiz(10.0)
                .padding_vert(4.0)
                .width_pct(100.0)
                .color(config.get().color(LapceColor::EDITOR_DIM))
        })),
        Entry::Tool(tool) => {
            let title = tool.title.clone();
            let status = tool.status;
            let locations = tool.locations.clone();

            Box::new(
                stack((
                    label(move || {
                        let mark = match status {
                            Some(ToolCallStatus::Completed) => "✓",
                            Some(ToolCallStatus::Failed) => "✗",
                            Some(ToolCallStatus::InProgress) => "…",
                            _ => "·",
                        };
                        format!("{mark} {title}")
                    })
                    .style(move |s| {
                        s.padding_horiz(10.0)
                            .padding_vert(4.0)
                            .width_pct(100.0)
                            .color(config.get().color(match status {
                                Some(ToolCallStatus::Failed) => LapceColor::LAPCE_ERROR,
                                _ => LapceColor::EDITOR_DIM,
                            }))
                    }),
                    dyn_stack(
                        move || locations.clone().into_iter().enumerate(),
                        |(i, _)| *i,
                        move |(_, (path, line))| {
                            let display = format!(
                                "{}:{}",
                                path.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| path.to_string_lossy().to_string()),
                                line + 1
                            );
                            let target = path.clone();
                            label(move || display.clone())
                                .style(move |s| {
                                    s.padding_left(28.0)
                                        .padding_vert(2.0)
                                        .cursor(CursorStyle::Pointer)
                                        .color(config.get().color(LapceColor::EDITOR_LINK))
                                })
                                .on_click_stop(move |_| {
                                    internal_command.send(
                                        InternalCommand::JumpToLocation {
                                            location: EditorLocation {
                                                path: target.clone(),
                                                position: Some(EditorPosition::Line(
                                                    line as usize,
                                                )),
                                                scroll_offset: None,
                                                ignore_unconfirmed: false,
                                                same_editor_tab: false,
                                            },
                                        },
                                    );
                                })
                        },
                    )
                    .style(|s| s.flex_col().width_pct(100.0)),
                ))
                .style(|s| s.flex_col().width_pct(100.0)),
            )
        }
    }
}

/// The prompt box. Enter submits; Shift+Enter is left to the editor so a
/// multi-line question is still possible.
fn prompt_box(window_tab_data: Rc<WindowTabData>) -> impl View {
    let config = window_tab_data.common.config;
    let focus = window_tab_data.common.focus;
    let agent = window_tab_data.agent.clone();
    let editor = agent.input.clone();
    let submit = agent.clone();

    container(
        TextInputBuilder::new()
            .is_focused(move || {
                focus.get() == Focus::Panel(crate::panel::kind::PanelKind::Agent)
            })
            .build_editor(editor)
            .on_event_stop(EventListener::KeyDown, move |event| {
                if let Event::KeyDown(key) = event {
                    if key.key.logical_key == Key::Named(NamedKey::Enter)
                        && !key.modifiers.shift()
                    {
                        submit.submit();
                    }
                }
            })
            .style(|s| s.width_pct(100.0)),
    )
    .style(move |s| {
        let config = config.get();
        s.padding(6.0)
            .width_pct(100.0)
            .border_top(1.0)
            .border_color(config.color(LapceColor::LAPCE_BORDER))
    })
}

/// Shown only while the agent is blocked on a decision.
///
/// The agent is waiting for this, so every way out sends an answer: each option
/// sends its id, and Dismiss sends `None`, which the client reports as
/// cancelled. Leaving without answering would hang the turn -- the exact bug
/// the client was fixed for.
fn permission_bar(window_tab_data: Rc<WindowTabData>) -> impl View {
    let config = window_tab_data.common.config;
    let agent = window_tab_data.agent.clone();
    let pending = agent.pending_permission;

    dyn_stack(
        // Zero or one row: a stack keyed on the title so the bar rebuilds when
        // a different request arrives.
        move || pending.get().into_iter().collect::<Vec<_>>(),
        |p| p.title.clone(),
        move |request| {
            let title = request.title.clone();
            let options = request.options.clone();
            let dismiss = request.clone();
            let dismiss_signal = pending;

            stack((
                label(move || format!("Allow: {title}?")).style(move |s| {
                    s.padding_horiz(10.0)
                        .padding_vert(6.0)
                        .width_pct(100.0)
                        .color(config.get().color(LapceColor::EDITOR_FOREGROUND))
                }),
                dyn_stack(
                    move || options.clone().into_iter().enumerate(),
                    |(i, _)| *i,
                    move |(_, (option_id, name))| {
                        let answer = request.clone();
                        let signal = pending;
                        label(move || format!("  [ {name} ]  "))
                            .style(move |s| {
                                s.padding_vert(4.0)
                                    .cursor(CursorStyle::Pointer)
                                    .color(
                                        config.get().color(LapceColor::EDITOR_LINK),
                                    )
                            })
                            .on_click_stop(move |_| {
                                answer.respond(Some(option_id.clone()));
                                signal.set(None);
                            })
                    },
                )
                .style(|s| s.padding_horiz(10.0).padding_bottom(6.0)),
                label(|| "  [ Dismiss ]  ".to_string())
                    .style(move |s| {
                        s.padding_horiz(10.0)
                            .padding_bottom(6.0)
                            .cursor(CursorStyle::Pointer)
                            .color(config.get().color(LapceColor::EDITOR_DIM))
                    })
                    .on_click_stop(move |_| {
                        // Dismissing is still an answer. Silence hangs the turn.
                        dismiss.respond(None);
                        dismiss_signal.set(None);
                    }),
            ))
            .style(move |s| {
                let config = config.get();
                s.flex_col()
                    .width_pct(100.0)
                    .border_top(1.0)
                    .border_color(config.color(LapceColor::LAPCE_WARN))
                    .background(config.color(LapceColor::PANEL_CURRENT_BACKGROUND))
            })
        },
    )
    .style(|s| s.flex_col().width_pct(100.0))
}
