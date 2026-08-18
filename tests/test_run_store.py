"""
Shipment Delay & Exception — Agentic Operations Dashboard (Streamlit)

A thin, role-aware client of the FastAPI backend that drives the AUTONOMOUS
AGENTS. The primary run calls POST /v1/agentic/run, which launches two ReAct
tool-calling agents in parallel (a per-order triage agent and an aggregate
root-cause agent). The dashboard surfaces their activity (tool-call
trajectories, final answers), the ranked root causes, escalations, and the
human-approval queue.

Role split (enforced by the API's JWT scopes, not just the UI):
    Analyst (role=analyst)            -> analytics:run  (can run agents, view results)
    Manager (role=operations_manager) -> + notify:send  (can review & approve messages)

Run alongside the API:
    Terminal 1:  uvicorn app.api.app:create_app --factory --reload --port 8000
    Terminal 2:  streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import base64
import io
import json

import pandas as pd
import requests
import streamlit as st

try:
    import altair as alt
    HAVE_ALT = True
except Exception:
    HAVE_ALT = False

# --------------------------------------------------------------------------- #
# Config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Shipment Exception — Agentic Ops", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

INK = "#0F1B2D"; AMBER = "#E8A33D"; RED = "#D64545"; TEAL = "#2BB3A3"; STEEL = "#5B7089"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; }}
  .ops-band {{ background: linear-gradient(90deg, {INK} 0%, #16263c 100%);
     color:#EAF0F7; padding:14px 20px; border-radius:10px; margin-bottom:6px;
     border-left:5px solid {AMBER}; }}
  .ops-band h1 {{ font-size:1.35rem; margin:0; letter-spacing:.3px; }}
  .ops-band p  {{ margin:2px 0 0; color:#9DB2CC; font-size:.82rem; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:600; }}
  .grade-observed_fact       {{ background:#e6f6f3; color:{TEAL}; }}
  .grade-data_supported_risk {{ background:#fbf0dd; color:#a5741c; }}
  .grade-hypothesis          {{ background:#f2f4f7; color:{STEEL}; }}
  .agent-card {{ border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; background:#FBFCFE; }}
  .traj-step {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8rem;
     background:#0F1B2D; color:#CFE3FF; padding:6px 10px; border-radius:6px; margin:4px 0; }}
  div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
def _url(base, path): return base.rstrip("/") + path
def _auth(tok): return {"Authorization": f"Bearer {tok}"} if tok else {}

def get_token(base, username, role):
    r = requests.post(_url(base, "/auth/token"), json={"username": username, "role": role}, timeout=15)
    r.raise_for_status(); return r.json()["access_token"]

def get_health(base):
    r = requests.get(_url(base, "/health"), timeout=15); r.raise_for_status(); return r.json()

def post_file(base, path, tok, data, name, params=None):
    files = {"file": (name, io.BytesIO(data), "text/csv")}
    return requests.post(_url(base, path), headers=_auth(tok), files=files, params=params or {}, timeout=900)

def api_get(base, path, tok, params=None):
    return requests.get(_url(base, path), headers=_auth(tok), params=params or {}, timeout=60)

def api_post_json(base, path, tok, body):
    return requests.post(_url(base, path), headers=_auth(tok), json=body, timeout=60)

def decode_scopes(tok):
    try:
        seg = tok.split(".")[1]; seg += "=" * (-len(seg) % 4)
        return json.loads(base64.urlsafe_b64decode(seg)).get("scopes", [])
    except Exception:
        return []

def show_api_error(r):
    try: detail = r.json().get("detail", r.text)
    except Exception: detail = r.text
    st.error(f"API {r.status_code}: {detail}")

def grade_pill(grade):
    g = str(grade).replace("EvidenceGrade.", "").lower()
    label = {"observed_fact": "observed fact", "data_supported_risk": "data-supported",
             "hypothesis": "hypothesis"}.get(g, g)
    return f'<span class="pill grade-{g}">{label}</span>'


def render_escalation_details(e: dict):
    """Show the full EscalationDecision (same shape as /analyze) so an approver
    can decide on the narrative, mitigation and expected effect."""
    conf, thr = e.get("confidence"), e.get("threshold")
    xo, xm = e.get("excess_orders"), e.get("excess_margin_usd")
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else "—")
    cols[1].metric("Threshold", f"{thr:.2f}" if isinstance(thr, (int, float)) else "—")
    cols[2].metric("Excess late orders", f"{xo:,.0f}" if isinstance(xo, (int, float)) else "—")
    cols[3].metric("Margin at risk", f"${xm:,.0f}" if isinstance(xm, (int, float)) else "—")
    if e.get("narrative"):
        st.markdown(f"**What's happening.** {e['narrative']}")
    if e.get("mitigation"):
        st.markdown(f"**Recommended action.** {e['mitigation']}")
    if e.get("expected_effect"):
        st.markdown(f"**Expected effect.** {e['expected_effect']}")
    meta = []
    if e.get("finding_id"):
        meta.append(f"finding: {e['finding_id']}")
    if isinstance(e.get("candidates_evaluated"), int):
        meta.append(f"candidates evaluated: {e['candidates_evaluated']:,}")
    if isinstance(e.get("m_tests_conducted"), int):
        meta.append(f"tests conducted: {e['m_tests_conducted']:,}")
    if meta:
        st.caption(" · ".join(meta))

# --------------------------------------------------------------------------- #
# Session state + approval decision helpers
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("token", None); ss.setdefault("role", None); ss.setdefault("scopes", [])
ss.setdefault("agentic", None); ss.setdefault("file_bytes", None); ss.setdefault("file_name", None)
ss.setdefault("approvals", None)
ss.setdefault("appr_offset", 0); ss.setdefault("appr_search", ""); ss.setdefault("appr_page", None)
ss.setdefault("appr_done", {})  # approval_id -> "approved"/"rejected" (local record for this session)

APPR_PAGE_SIZE = 20

# How many escalations the Overview panel shows (backend may raise more; the
# remainder are always reachable from the Approvals queue).
TOP_N_ESCALATIONS = 2

def load_approvals_page(base):
    rid = (ss.agentic or {}).get("run_id")
    params = {"run_id": rid, "limit": APPR_PAGE_SIZE, "offset": ss.appr_offset}
    if ss.appr_search.strip():
        params["order_id"] = ss.appr_search.strip()
    r = api_get(base, "/v1/agentic/approvals", ss.token, params)
    if r.status_code == 200:
        ss.appr_page = r.json()
    else:
        ss.appr_page = None
    return r

def _refresh_approvals(base):
    # kept for compatibility; reload the current page
    return load_approvals_page(base)

def _after_decision(base, r, aid=None, decision=None):
    if r.status_code == 200:
        if aid and decision:
            ss.appr_done[aid] = decision
        load_approvals_page(base)
        st.rerun()
    else:
        show_api_error(r)

def _decision_buttons(base, item):
    aid = item.get("approval_id")
    prior = ss.appr_done.get(aid)
    if prior or (item.get("status") and item.get("status") != "pending"):
        st.caption(f"✔ Decision recorded: {prior or item.get('status')}")
        return
    c1, c2, _ = st.columns([1, 1, 5])
    if c1.button("✅ Approve & send", key=f"ap_{aid}"):
        _after_decision(base, api_post_json(base, f"/v1/agentic/approvals/{aid}", ss.token,
                        {"approve": True}), aid, "approved")
    if c2.button("✖️ Reject", key=f"rj_{aid}"):
        _after_decision(base, api_post_json(base, f"/v1/agentic/approvals/{aid}", ss.token,
                        {"approve": False}), aid, "rejected")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Control panel")
    api_base = st.text_input("API base URL", value="http://localhost:8000")
    role_label = st.radio("Sign in as", ["Analyst", "Operations Manager"],
                          help="Role decides what you can do — enforced by the API. Both roles run "
                               "the agents; only Managers can review and approve customer messages.")
    role_value = "analyst" if role_label == "Analyst" else "operations_manager"
    username = st.text_input("Username", value="demo")

    if st.button("🔌 Connect", use_container_width=True, type="primary"):
        try:
            tok = get_token(api_base, username, role_value)
            ss.token, ss.role, ss.scopes = tok, role_value, decode_scopes(tok)
            ss.agentic = ss.approvals = ss.appr_page = None
            ss.appr_offset = 0; ss.appr_search = ""; ss.appr_done = {}
            st.success(f"Signed in as {role_label}")
        except Exception as e:
            st.error(f"Could not connect / sign in: {e}")

    if ss.token:
        can_approve = "notify:send" in ss.scopes
        connected_label = "Operations Manager" if ss.role == "operations_manager" else "Analyst"
        st.caption(f"**Connected as:** {connected_label}")
        st.caption("**Scopes:** " + ", ".join(ss.scopes))
        st.caption("✅ Can approve notifications" if can_approve else "🔒 View-only — cannot approve (needs Manager)")
        if role_value != ss.role:
            st.warning(f"You selected **{role_label}** but are still connected as "
                       f"**{connected_label}**. Click **Connect** to switch.")
        try:
            h = get_health(api_base)
            st.caption(("🟢" if h.get("provider_healthy") else "🟡") + f" Provider: {h.get('provider')} · {h.get('model')}")
        except Exception:
            st.caption("🔴 API health check failed")

    st.divider()
    up = st.file_uploader("Order batch (CSV/XLSX)", type=["csv", "xlsx"])
    if up is not None:
        ss.file_bytes, ss.file_name = up.getvalue(), up.name

    disabled = not (ss.token and ss.file_bytes)
    if st.button("🤖 Run agents", use_container_width=True, disabled=disabled,
                 help="Calls POST /v1/agentic/run — launches the triage agent and root-cause agent."):
        with st.spinner("Running autonomous agents (triage + root cause)…"):
            r = post_file(api_base, "/v1/agentic/run", ss.token, ss.file_bytes, ss.file_name)
        if r.status_code == 200:
            ss.agentic = r.json(); ss.approvals = ss.appr_page = None
            ss.appr_offset = 0; ss.appr_search = ""; ss.appr_done = {}
            st.success("Agents finished")
        else:
            show_api_error(r)

# --------------------------------------------------------------------------- #
# Header + gates
# --------------------------------------------------------------------------- #
st.markdown('<div class="ops-band"><h1>🤖 Shipment Exception — Agentic Operations Dashboard</h1>'
            '<p>Two autonomous agents in parallel · systemic root-cause analysis · human-approved notifications</p></div>',
            unsafe_allow_html=True)

if not ss.token:
    st.info("👋 Connect in the left panel (Analyst or Operations Manager), upload a batch, and click **Run agents**.")
    st.stop()

A = ss.agentic or {}
is_manager = "notify:send" in ss.scopes
has_run = ss.agentic is not None

# A Manager can go straight to the Approvals queue without running anything —
# the queue is pulled from the shared store (populated by any run/trigger).
# Everyone else must run the agents first to have anything to look at.
# Anyone can reach Run history: a run triggered by a file drop or another user
# has no session state here, and a reviewer must still be able to open it.

root_causes = A.get("root_causes", [])
escalations = A.get("escalations", [])

tab_names = ["📊 Overview", "🤖 Agents", "🔍 Root causes", "✅ Approvals", "🗂️ Run history"]
tabs = st.tabs(tab_names)


def _load_run(base: str, run_id: str) -> None:
    """Pull a stored run into this session so every tab renders it.

    Loading into ss.agentic means the Overview / Agents / Root-cause views work
    unchanged on a run this user never triggered — which is the whole point:
    a file-drop batch has no uploader to show it to.
    """
    r = api_get(base, f"/v1/runs/{run_id}", ss.token)
    if r.status_code == 200:
        ss.agentic = r.json().get("result") or {}
        ss.loaded_run_id = run_id
        st.rerun()
    else:
        show_api_error(r)

# --------------------------- Overview --------------------------- #
with tabs[0]:
    if not has_run:
        st.info("No run loaded in this session. Upload a batch and click **Run agents** to "
                "populate this view — or go to **Approvals** to work the existing queue.")
    else:
        validated = sum(1 for r in root_causes if r.get("status") == "validated")
        c = st.columns(4)
        c[0].metric("At-risk orders", f"{A.get('at_risk_orders', 0):,}")
        c[1].metric("Notifications drafted", f"{A.get('notifications_drafted', 0):,}")
        c[2].metric("Queued for approval", f"{A.get('approvals_created', 0):,}")
        c[3].metric("Validated root causes", f"{validated:,}")

        c = st.columns(3)
        c[0].metric("Triage agent — tool calls", A.get("triage_agent", {}).get("tool_calls", 0))
        c[1].metric("Root-cause agent — tool calls", A.get("root_cause_agent", {}).get("tool_calls", 0))
        c[2].metric("Escalations raised", f"{len(escalations):,}")

        st.divider()
        if escalations:
            shown = escalations[:TOP_N_ESCALATIONS]
            if len(shown) > 1:
                st.error(f"🚨 **{len(shown)} escalations raised** — the strongest systemic "
                         f"causes that cleared the confidence threshold, highest impact first.")
            for i, e in enumerate(shown, start=1):
                label = e.get("finding_label") or e.get("finding_id")
                if len(shown) > 1:
                    st.markdown(f"##### #{e.get('rank', i)} · {label}")
                else:
                    st.error(f"🚨 **Escalation raised** — {label}")
                render_escalation_details(e)
                if i < len(shown):
                    st.divider()
            if len(escalations) > len(shown):
                st.caption(f"{len(escalations) - len(shown)} further escalation(s) raised — "
                           f"see the **Approvals** tab.")
        else:
            st.success("✅ **No escalation** — no systemic cause cleared the confidence threshold "
                       "(the alert-fatigue guard held).")

# --------------------------- Agents --------------------------- #
with tabs[1]:
    if not has_run:
        st.info("Run the agents to see their activity here.")
    st.caption("Each agent reasons and calls tools on its own (ReAct). Below is what each one did.")

    ca, cb = st.columns(2)
    for col, key, title, icon in [(ca, "triage_agent", "Triage agent", "🧭"),
                                  (cb, "root_cause_agent", "Root-cause agent", "🔬")]:
        ag = A.get(key, {})
        with col:
            st.markdown(f"#### {icon} {title}")
            st.metric("Tool calls", ag.get("tool_calls", 0))
            traj = ag.get("trajectory", []) or []
            if traj:
                st.caption("Trajectory (tools called, in order):")
                for i, step in enumerate(traj, 1):
                    args = step.get("args", {})
                    arg_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                    st.markdown(f'<div class="traj-step">{i}. {step.get("tool")}({arg_str})</div>',
                                unsafe_allow_html=True)
            else:
                st.info("No tool calls recorded. With the **mock** provider the agents don't call tools; "
                        "coverage and root causes below are still produced deterministically. "
                        "Use a real provider (e.g. OpenAI) to see full agent trajectories.")
            st.caption("Final answer:")
            st.write(ag.get("final_answer", "—"))

# --------------------------- Root causes --------------------------- #
with tabs[2]:
    if not has_run:
        st.info("Run the agents to see ranked root causes here.")
    elif not root_causes:
        st.info("No candidate patterns were found on this batch.")
    else:
        n_val = sum(1 for f in root_causes if f.get("status") == "validated")
        st.caption(f"Top {len(root_causes)} strongest delay patterns — {n_val} fully validated, "
                   f"the rest shown as **candidates** (did not clear every gate, graded *hypothesis*).")
        fdf = pd.DataFrame([{
            "label": f.get("finding_label"),
            "confidence": (f.get("confidence") or 0.0),
            "excess_orders": (f.get("excess_orders") or 0),
            "lift": (f.get("lift") or 0),
            "kind": "validated" if f.get("status") == "validated" else "candidate",
        } for f in root_causes])
        if HAVE_ALT:
            bar = alt.Chart(fdf).mark_bar().encode(
                x=alt.X("excess_orders:Q", title="Excess late orders"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("kind:N",
                                scale=alt.Scale(domain=["validated", "candidate"],
                                                range=[RED, STEEL]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["label", "excess_orders", "lift", "confidence", "kind"],
            ).properties(height=max(140, 32 * len(fdf)))
            st.altair_chart(bar, use_container_width=True)
            st.caption("Red = fully validated (cleared all five gates); grey = candidate signal.")

        for f in root_causes:
            conf = f.get("confidence")
            conf_str = f"conf {conf:.2f}" if isinstance(conf, (int, float)) else "candidate"
            status = f.get("status", "")
            badge = "✅ validated" if status == "validated" else f"🔎 candidate · {status}"
            head = (f"#{f.get('rank','?')} · {badge} · **{f.get('finding_label')}** — late "
                    f"{(f.get('late_rate') or 0)*100:.1f}% vs baseline "
                    f"{(f.get('baseline_rate') or 0)*100:.1f}% · lift {(f.get('lift') or 0):.2f} · {conf_str}")
            with st.expander(head):
                st.markdown(grade_pill(f.get("evidence_grade", "")), unsafe_allow_html=True)
                m = st.columns(3)
                m[0].metric("Excess late orders", f"{(f.get('excess_orders') or 0):.0f}")
                mv = f.get("excess_margin_usd")
                m[1].metric("Margin at risk", f"${mv:,.0f}" if isinstance(mv, (int, float)) else "—")
                m[2].metric("Orders in segment", f"{f.get('n',0):,}")
                if f.get("narrative"): st.markdown(f"**What's happening.** {f['narrative']}")
                if f.get("mitigation"): st.markdown(f"**Recommended action.** {f['mitigation']}")
                if f.get("expected_effect"): st.markdown(f"**Expected effect.** {f['expected_effect']}")

# --------------------------- Approvals --------------------------- #
with tabs[3]:
    if not is_manager:
        st.warning("🔒 Reviewing and approving the drafted customer messages requires the "
                   "**Operations Manager** role (notify:send).")
        if role_value == "operations_manager":
            st.info("You've selected **Operations Manager** in the left panel — now click "
                    "**🔌 Connect** to switch your session, then reopen this tab to review "
                    "and approve messages.")
        else:
            st.info("Switch **Sign in as → Operations Manager** in the left panel and click "
                    "**Connect** to review and approve messages.")
        st.metric("Messages queued for a manager to approve", f"{A.get('approvals_created', 0):,}")
    else:
        st.caption("Review the exact message that would be sent for each at-risk order, "
                   "then approve (send) or reject it.")
        if not has_run:
            st.info("Showing the **live pending queue** pulled from the store — you don't need to "
                    "upload or run a batch. (Search below is by order ID.)")

        # Search by order id + refresh
        sc1, sc2, sc3 = st.columns([3, 1, 1])
        search = sc1.text_input("Find an order by ID", value=ss.appr_search,
                                placeholder="e.g. 70524", label_visibility="collapsed")
        if sc2.button("🔎 Search", use_container_width=True):
            ss.appr_search = search; ss.appr_offset = 0; load_approvals_page(api_base)
        if sc3.button("🔄 Refresh", use_container_width=True):
            load_approvals_page(api_base)

        if ss.appr_page is None:
            load_approvals_page(api_base)

        page = ss.appr_page or {"total": 0, "pending": [], "offset": 0, "limit": APPR_PAGE_SIZE}
        pend = page.get("pending", [])
        total = page.get("total", 0)
        notes = [a for a in pend if a.get("type") == "send_notification"]
        escs = [a for a in pend if a.get("type") == "file_escalation"]

        top = st.columns(3)
        top[0].metric("Pending in queue", f"{total:,}")
        top[1].metric("Approved this session", sum(1 for v in ss.appr_done.values() if v == "approved"))
        top[2].metric("Rejected this session", sum(1 for v in ss.appr_done.values() if v == "rejected"))

        # Escalations first (if any land on this page)
        for a in escs:
            p = a.get("payload", {})
            st.error(f"🚨 **Escalation pending your approval** — "
                     f"{p.get('finding_label') or p.get('finding_id')}")
            with st.expander("Escalation details — narrative · mitigation · expected effect",
                             expanded=True):
                render_escalation_details(p)
            _decision_buttons(api_base, a)

        if not notes and not escs:
            st.info("No pending messages on this page. Adjust the search or refresh.")

        # One expander per message: the FULL draft + approve/reject
        for a in notes:
            p = a.get("payload", {})
            title = f"✉️ {p.get('subject','(no subject)')} · order {p.get('order_id')} · " \
                    f"P(late) {p.get('p_late')} · ETA {p.get('revised_eta')}"
            with st.expander(title):
                meta = "✅ passed guardrail" if not p.get("used_fallback") else "🛟 safe fallback"
                st.caption(f"{meta} · remedy tier {p.get('remedy_tier')} · reason {p.get('reason_code')}")
                st.text(p.get("body", "(message body unavailable)"))
                _decision_buttons(api_base, a)

        # Pager
        st.divider()
        pcol = st.columns([1, 2, 1])
        if pcol[0].button("◀ Previous", disabled=ss.appr_offset <= 0, use_container_width=True):
            ss.appr_offset = max(0, ss.appr_offset - APPR_PAGE_SIZE); load_approvals_page(api_base); st.rerun()
        shown_lo = 0 if total == 0 else ss.appr_offset + 1
        shown_hi = min(ss.appr_offset + APPR_PAGE_SIZE, total)
        pcol[1].markdown(f"<div style='text-align:center'>Showing {shown_lo}–{shown_hi} of {total:,}</div>",
                         unsafe_allow_html=True)
        if pcol[2].button("Next ▶", disabled=shown_hi >= total, use_container_width=True):
            ss.appr_offset = ss.appr_offset + APPR_PAGE_SIZE; load_approvals_page(api_base); st.rerun()

# --------------------------- Run history --------------------------- #
with tabs[4]:
    st.caption("Every run is stored, whichever trigger produced it — upload, webhook, "
               "file drop or scheduler. Open one to load its analysis into the tabs above.")
    cols = st.columns([1, 1, 2])
    src = cols[0].selectbox("Source", ["all", "upload", "webhook", "file_drop", "scheduler"])
    limit = cols[1].number_input("Show", min_value=5, max_value=200, value=25, step=5)
    if cols[2].button("🔄 Refresh", use_container_width=True):
        st.rerun()

    params = {"limit": int(limit)}
    if src != "all":
        params["source"] = src
    resp = api_get(api_base, "/v1/runs", ss.token, params)
    if resp.status_code != 200:
        show_api_error(resp)
    else:
        runs = resp.json().get("runs", [])
        if not runs:
            st.info("No runs recorded yet. Upload a batch, or drop a CSV into the "
                    "configured input folder.")
        for run in runs:
            rid = run["run_id"]
            when = str(run.get("created_at", ""))[:19].replace("T", " ")
            icon = {"file_drop": "📁", "webhook": "🔔", "scheduler": "⏰"}.get(run["source"], "⬆️")
            failed = run.get("status") == "failed"
            head = (f"{icon} {when} UTC · {run['source']}"
                    f" · {run.get('file_name') or '—'}"
                    f" · {'❌ failed' if failed else str(run.get('root_cause_count', 0)) + ' root causes'}")
            with st.expander(head, expanded=False):
                if failed:
                    st.error(run.get("error") or "run failed")
                else:
                    m = st.columns(4)
                    m[0].metric("At-risk orders", f"{run.get('at_risk_orders', 0):,}")
                    m[1].metric("Notifications", f"{run.get('notifications', 0):,}")
                    m[2].metric("Root causes", run.get("root_cause_count", 0))
                    m[3].metric("Escalations", run.get("escalation_count", 0))
                st.caption(f"run_id `{rid}` · triggered by **{run.get('triggered_by') or 'system'}**"
                           f" · {run.get('rows', 0):,} rows")
                if not failed and st.button("Open this run", key=f"open_{rid}",
                                            use_container_width=True):
                    _load_run(api_base, rid)