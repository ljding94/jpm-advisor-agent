"""Streamlit frontend for jpm-advisor-agent.

Three modes (sidebar radio):
1. Watch persona run    — pick a built-in persona, click Run, watch the live transcript.
2. Human-in-loop        — type as the client; LLMs handle Advisor and Analyst.
3. Persona editor       — edit/save a ClientProfile JSON in the browser.

Run with:    streamlit run app.py

A working LLM provider must be configured via env vars (see .env.example).
"""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any

import streamlit as st

from src.graph.state import ConversationStatus, initial_state
from src.main import build_runtime
from src.observability.logger import render_transcript
from src.providers.llm import LLMProvider, get_llm_provider
from src.schemas import AgentRole, ClientProfile
from src.tools.web_search import DDGSearchProvider
from src.ui.human_client import HumanClientAgent
from src.ui.persona_editor import (
    PERSONA_FILES,
    form_dict_to_profile,
    load_persona,
    profile_to_form_dict,
    save_persona,
)
from src.ui.streaming import UITurnLogger

PROVIDER_OPTIONS = {
    "openrouter": "OpenRouter (multi-model gateway)",
    "openai": "OpenAI (native API)",
    "anthropic": "Anthropic (native API)",
    "ollama": "Ollama (local, no key)",
}

# --------- model choices ---------

# Curated, popular models per provider. First entry is the dropdown default.
# Users can also pick "Other (custom id)…" to type any slug — including
# `:free` variants if they want to experiment with free-tier models.
_MODEL_CHOICES = {
    "openrouter": [
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.1-70b-instruct",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o1-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5",
        "claude-sonnet-4",
    ],
    "ollama": [
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen2.5:7b",
        "mistral:7b",
    ],
}


def _model_choices_for(provider: str) -> list[str]:
    return list(_MODEL_CHOICES.get(provider, []))

# Per-provider env-var names so the sidebar can show / set the right credential.
PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # local, no key
}

ROLE_AVATAR = {
    AgentRole.CLIENT: "🧑",
    AgentRole.ADVISOR: "💼",
    AgentRole.ANALYST: "🔬",
}


def _ensure_state() -> None:
    """Initialize keys we need across reruns.

    Cross-thread note: Streamlit's `st.session_state` is thread-local, so a
    worker thread CAN'T write to it and have the main thread see the change.
    Anything the graph worker needs to publish (transcript, final state,
    completion flag) lives inside `holder`, a plain Python dict whose
    *reference* is stored in session_state. Mutations to that dict are
    visible across threads under the GIL.
    """
    ss = st.session_state
    ss.setdefault("holder", {
        "running": False,
        "final_state": None,
        "transcript": [],          # list[AgentMessage] — appended by stream loop
        "error": None,
    })
    ss.setdefault("ui_log", None)          # UITurnLogger (also a shared object)
    ss.setdefault("client_responses", None)  # queue.Queue for HumanClientAgent
    ss.setdefault("client_prompts", None)    # queue.Queue from agent -> UI
    ss.setdefault("pending_human_prompt", None)
    ss.setdefault("editor_profile", None)
    ss.setdefault("custom_profile", None)


def _apply_provider_selection(provider: str, model: str, api_key: str | None = None) -> None:
    """Push provider choice + (optional) api key into the process env.

    The LLM provider classes read these env vars on construction, so writing
    here before the next `get_llm_provider()` call is enough — no restart
    needed. We never *clear* an existing env var (e.g. one loaded from .env);
    the user's UI input only overrides when non-empty.
    """
    os.environ["LLM_PROVIDER"] = provider
    if provider == "openrouter":
        os.environ["LLM_MODEL"] = model
    elif provider == "openai":
        os.environ["OPENAI_MODEL"] = model
    elif provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model
    elif provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model

    key_var = PROVIDER_KEY_ENV.get(provider)
    if key_var and api_key:
        os.environ[key_var] = api_key


def _make_llm_or_error() -> LLMProvider | None:
    try:
        return get_llm_provider()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialize LLM provider: {exc}")
        return None


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.title("JPM Advisor")
    st.sidebar.caption("LangGraph 3-agent advisor")

    mode = st.sidebar.radio(
        "Mode",
        ["Watch persona run", "Human-in-loop", "Persona editor"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM provider")
    provider = st.sidebar.selectbox(
        "Provider",
        list(PROVIDER_OPTIONS),
        format_func=lambda k: PROVIDER_OPTIONS[k],
        index=0,
    )

    # Model picker: dropdown of common ids per provider, plus "Other...".
    OTHER = "Other (custom id)…"
    base_choices = _model_choices_for(provider)
    choices = base_choices + [OTHER]
    selected = st.sidebar.selectbox(
        "Model",
        choices,
        index=0,
        key=f"model_choice_{provider}",  # remember per-provider
        help="Pick a curated model id, or choose 'Other…' to type a custom one.",
    )
    if selected == OTHER:
        model = st.sidebar.text_input(
            "Custom model id",
            value="",
            placeholder=base_choices[0] if base_choices else "",
            key=f"custom_model_{provider}",
        ).strip() or (base_choices[0] if base_choices else "")
    else:
        model = selected

    # API key: per-provider, password-style. Empty input keeps the existing
    # env value (e.g. from .env) untouched. Ollama doesn't need one.
    key_var = PROVIDER_KEY_ENV.get(provider)
    api_key_input: str | None = None
    if key_var is not None:
        existing = os.getenv(key_var, "")
        placeholder = "(set in .env)" if existing else f"paste your {key_var}"
        api_key_input = st.sidebar.text_input(
            f"{key_var}",
            value="",
            type="password",
            placeholder=placeholder,
            key=f"apikey_{provider}",
            help=(
                "Stored only in this process's env — never written to disk. "
                "Leave blank to use the value from .env."
            ),
        ).strip() or None
        if existing and api_key_input is None:
            st.sidebar.caption(f"✓ {key_var} loaded from environment.")
        elif api_key_input:
            st.sidebar.caption(f"✓ {key_var} set from this session.")
        else:
            st.sidebar.warning(f"No {key_var} set. The run will fail until you provide one.")
    else:
        st.sidebar.caption("Local provider — no API key required.")

    _apply_provider_selection(provider, model, api_key_input)

    st.sidebar.markdown("---")
    persona = st.sidebar.selectbox(
        "Persona",
        list(PERSONA_FILES.keys()),
        index=1,  # david
        help="Built-in persona to use (ignored if you pick a custom one in the editor).",
    )

    if st.session_state.custom_profile is not None:
        st.sidebar.success(
            f"Custom persona active: {st.session_state.custom_profile.name}"
        )
        if st.sidebar.button("Clear custom persona"):
            st.session_state.custom_profile = None

    # `key_ok` tells the modes whether they can enable the Run button.
    key_ok = (key_var is None) or bool(api_key_input) or bool(os.getenv(key_var, ""))
    return {
        "mode": mode, "provider": provider, "model": model, "persona": persona,
        "key_ok": key_ok, "key_var": key_var,
    }


def _resolve_profile(persona_key: str) -> ClientProfile:
    if st.session_state.custom_profile is not None:
        return st.session_state.custom_profile
    return load_persona(persona_key)


def _safe_md(text: str) -> str:
    """Escape characters Streamlit's markdown renderer interprets specially.

    Most importantly: a pair of `$` signs is treated as inline LaTeX math, which
    mangles dollar amounts in LLM output (e.g. "$40K to $50K" becomes a math
    block with each character on its own line). Escaping `$` to `\\$` fixes
    this without breaking actual markdown formatting.
    """
    return text.replace("$", r"\$")


def _render_transcript_from_holder() -> None:
    holder = st.session_state.holder
    transcript = holder.get("transcript", [])
    if not transcript:
        st.info("Click **Run** to start.")
        return
    for m in transcript:
        avatar = ROLE_AVATAR.get(m.sender, "•")
        with st.chat_message(m.sender.value, avatar=avatar):
            st.markdown(
                f"**{m.sender.value} → {m.recipient.value}**  \n_{m.message_type.value}_"
            )
            st.markdown(_safe_md(m.content))


def _render_meter() -> None:
    log: UITurnLogger | None = st.session_state.ui_log
    if log is None:
        return
    summary = log.summary()
    cols = st.columns(4)
    cols[0].metric("Turns", summary["turns"])
    cols[1].metric("Input tokens", summary["total_input_tokens"])
    cols[2].metric("Output tokens", summary["total_output_tokens"])
    cols[3].metric("Cost (USD)", f"${summary['total_cost_usd']:.4f}")


def _render_advice(state: dict | None) -> None:
    if state is None:
        return
    advice = state.get("draft_advice")
    if advice is None:
        return
    st.markdown("## Final advice")
    st.markdown("**Recommendations**")
    for r in advice.recommendations:
        st.markdown(f"- {_safe_md(r)}")
    st.markdown(f"**Rationale:** {_safe_md(advice.rationale)}")
    if advice.disclaimers:
        with st.expander("Disclaimers"):
            for d in advice.disclaimers:
                st.markdown(f"- {_safe_md(d)}")


def _start_persona_run(persona_key: str) -> None:
    profile = _resolve_profile(persona_key)

    # Build the LLM in the main thread so we can show errors in the UI.
    # build_runtime would otherwise call sys.exit on missing creds — fine
    # for the CLI, fatal for Streamlit.
    try:
        llm = get_llm_provider()
    except Exception as exc:  # noqa: BLE001
        st.session_state.holder = {
            "running": False, "final_state": None, "transcript": [],
            "error": f"LLM init failed: {exc}",
        }
        return

    state = initial_state(profile)
    ui_log = UITurnLogger()
    runtime = build_runtime(
        profile, llm=llm, verbose=False, turn_logger=ui_log,
        web=DDGSearchProvider(),
    )
    client = runtime["client"]
    graph = runtime["graph"]
    state = client.open_conversation(state)

    holder = {
        "running": True, "final_state": None, "transcript": [],
        "error": None, "stop": False, "stopped": False,
    }
    st.session_state.holder = holder
    st.session_state.ui_log = ui_log

    def _runner():
        try:
            # graph.stream yields one full state per node hop. We snapshot the
            # conversation_history into the shared holder so the main thread's
            # next rerun reads the latest messages, and check the stop flag
            # between yields for cooperative interruption.
            last_state = state
            for partial in graph.stream(
                state, config={"recursion_limit": 80}, stream_mode="values"
            ):
                if holder.get("stop"):
                    holder["stopped"] = True
                    break
                last_state = partial
                holder["transcript"] = list(partial.get("conversation_history", []))
            holder["final_state"] = last_state
            holder["transcript"] = list(last_state.get("conversation_history", []))
        except Exception as exc:  # noqa: BLE001
            holder["error"] = repr(exc)
        finally:
            holder["running"] = False

    t = threading.Thread(target=_runner, name="advisor-graph", daemon=True)
    t.start()


def _start_human_run(persona_key: str) -> None:
    profile = _resolve_profile(persona_key)
    prompts_out: queue.Queue[dict[str, Any]] = queue.Queue()
    responses_in: queue.Queue[str] = queue.Queue()

    llm = _make_llm_or_error()
    if llm is None:
        return

    human_client = HumanClientAgent(
        profile=profile, llm=llm,
        prompts_out=prompts_out, responses_in=responses_in,
    )

    ui_log = UITurnLogger()
    runtime = build_runtime(
        profile, llm=llm, client=human_client,
        verbose=False, turn_logger=ui_log,
    )
    graph = runtime["graph"]
    state = initial_state(profile)
    state = human_client.open_conversation(state)

    holder = {
        "running": True, "final_state": None, "transcript": [],
        "error": None, "stop": False, "stopped": False,
    }
    st.session_state.holder = holder
    st.session_state.ui_log = ui_log
    st.session_state.client_prompts = prompts_out
    st.session_state.client_responses = responses_in
    st.session_state.pending_human_prompt = None

    def _runner():
        try:
            last_state = state
            for partial in graph.stream(
                state, config={"recursion_limit": 80}, stream_mode="values"
            ):
                if holder.get("stop"):
                    holder["stopped"] = True
                    break
                last_state = partial
                holder["transcript"] = list(partial.get("conversation_history", []))
            holder["final_state"] = last_state
            holder["transcript"] = list(last_state.get("conversation_history", []))
        except Exception as exc:  # noqa: BLE001
            holder["error"] = repr(exc)
        finally:
            holder["running"] = False

    t = threading.Thread(target=_runner, name="advisor-graph-human", daemon=True)
    t.start()


# ----------------- modes -----------------

def _mode_watch(persona_key: str, *, key_ok: bool, key_var: str | None) -> None:
    st.title("Watch persona run")
    st.caption("Pick a persona, click Run, and watch all three agents collaborate.")

    profile = _resolve_profile(persona_key)
    with st.expander("Persona", expanded=False):
        st.json(profile.model_dump())

    holder = st.session_state.holder
    running = bool(holder.get("running"))

    if not key_ok and key_var:
        st.warning(
            f"Paste your **{key_var}** in the sidebar before running. "
            "Without it, the LLM provider can't be reached."
        )

    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Run", disabled=running or not key_ok, type="primary"):
            _start_persona_run(persona_key)
            st.rerun()
    with cols[1]:
        if st.button("Stop", disabled=not running):
            holder["stop"] = True
            st.rerun()

    if holder.get("error"):
        st.error(f"Run failed: {holder['error']}")
    elif holder.get("stopped") and not running:
        st.warning("Stopped. Pick a different model and click Run to retry.")

    if running:
        st.info("Running… auto-refreshing.")

    _render_meter()
    _render_transcript_from_holder()
    _render_advice(holder.get("final_state"))

    if running:
        # Re-render shortly so the streamed transcript and meter advance.
        import time as _time

        _time.sleep(0.8)
        st.rerun()


def _mode_human(persona_key: str, *, key_ok: bool, key_var: str | None) -> None:
    st.title("Human-in-loop")
    st.caption("You type as the client. The Advisor and Analyst stay LLM-driven.")

    profile = _resolve_profile(persona_key)
    with st.expander("Your client profile", expanded=False):
        st.json(profile.model_dump())

    holder = st.session_state.holder
    running = bool(holder.get("running"))

    if not key_ok and key_var:
        st.warning(
            f"Paste your **{key_var}** in the sidebar before starting."
        )

    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Start", disabled=running or not key_ok, type="primary"):
            _start_human_run(persona_key)
            st.rerun()
    with cols[1]:
        if st.button("Stop", disabled=not running):
            holder["stop"] = True
            st.rerun()

    if holder.get("error"):
        st.error(f"Run failed: {holder['error']}")
    elif holder.get("stopped") and not running:
        st.warning("Stopped.")

    if holder.get("error"):
        st.error(f"Run failed: {holder['error']}")

    _render_meter()
    _render_transcript_from_holder()

    # Drain any pending prompt the agent has put on the queue.
    if st.session_state.client_prompts is not None and st.session_state.pending_human_prompt is None:
        try:
            st.session_state.pending_human_prompt = st.session_state.client_prompts.get_nowait()
        except queue.Empty:
            pass

    pending = st.session_state.pending_human_prompt
    if pending is not None:
        kind = pending.get("kind", "question")
        st.markdown(f"### Advisor needs your input ({kind})")
        st.info(pending.get("content", ""))
        with st.form("human-input"):
            text = st.text_area("Your reply", height=120)
            ok = st.form_submit_button("Send")
            if ok and text.strip():
                st.session_state.client_responses.put(text.strip())
                st.session_state.pending_human_prompt = None
                st.rerun()
    elif running:
        st.info("Advisor and Analyst are working… (the UI will prompt you when needed)")
        import time as _time

        _time.sleep(0.8)
        st.rerun()

    _render_advice(holder.get("final_state"))


def _mode_editor() -> None:
    st.title("Persona editor")
    st.caption(
        "Edit a built-in persona or create a new one. "
        "Apply it as the active client to use in the other modes."
    )

    base_key = st.selectbox("Start from", list(PERSONA_FILES.keys()), index=1)
    if st.session_state.editor_profile is None:
        st.session_state.editor_profile = profile_to_form_dict(load_persona(base_key))
    if st.button("Reload from disk"):
        st.session_state.editor_profile = profile_to_form_dict(load_persona(base_key))
        st.rerun()

    form = st.session_state.editor_profile
    col1, col2 = st.columns(2)
    form["name"] = col1.text_input("Name", value=form["name"])
    form["age"] = col2.number_input(
        "Age", min_value=18, max_value=100, value=int(form["age"]), step=1,
    )
    risk_idx = ["conservative", "moderate", "aggressive"].index(form["risk_tolerance"])
    form["risk_tolerance"] = col1.selectbox(
        "Risk tolerance", ["conservative", "moderate", "aggressive"], index=risk_idx,
    )
    form["time_horizon_years"] = col2.number_input(
        "Time horizon (years)", min_value=0, max_value=80,
        value=int(form["time_horizon_years"]), step=1,
    )
    form["annual_income"] = col1.number_input(
        "Annual income (USD)", min_value=0.0, value=float(form["annual_income"]), step=1000.0,
    )
    form["goals"] = st.text_area(
        "Goals (one per line)", value=form["goals"], height=80,
    )
    form["assets_json"] = st.text_area(
        "Assets (JSON object: account → USD value)",
        value=form["assets_json"], height=120,
    )
    form["investments_json"] = st.text_area(
        "Investments (JSON list of {name, asset_class, value_usd})",
        value=form["investments_json"], height=120,
    )
    form["notes"] = st.text_area("Notes", value=form["notes"], height=80)

    cols = st.columns(3)
    with cols[0]:
        if st.button("Validate"):
            try:
                profile = form_dict_to_profile(form)
                st.success(f"Valid. {profile.name}, total assets ${profile.total_assets:,.0f}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid: {exc}")
    with cols[1]:
        if st.button("Use as active persona"):
            try:
                profile = form_dict_to_profile(form)
                st.session_state.custom_profile = profile
                st.success(f"Active persona set to {profile.name}. Switch to Watch or Human-in-loop.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid: {exc}")
    with cols[2]:
        save_key = st.text_input("Save as <key>.json", value="", placeholder="custom_user")
        if st.button("Save to data/personas/"):
            try:
                profile = form_dict_to_profile(form)
                if not save_key.strip():
                    raise ValueError("provide a non-empty key")
                out = save_persona(profile, save_key.strip())
                st.success(f"Saved to {out}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save: {exc}")


def main() -> None:
    st.set_page_config(page_title="JPM Advisor", page_icon="💼", layout="wide")
    _ensure_state()
    sb = _render_sidebar()

    if sb["mode"] == "Watch persona run":
        _mode_watch(sb["persona"], key_ok=sb["key_ok"], key_var=sb["key_var"])
    elif sb["mode"] == "Human-in-loop":
        _mode_human(sb["persona"], key_ok=sb["key_ok"], key_var=sb["key_var"])
    else:
        _mode_editor()


if __name__ == "__main__":
    main()
