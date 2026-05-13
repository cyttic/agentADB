"""
api/app.py
===========
FastAPI web server for DB Assistant Framework.

Endpoints:
  GET  /              — serves the chat UI
  GET  /health        — health check
  POST /chat          — send a message, get agent response
  POST /reset         — clear conversation history
  GET  /config        — current LLM config (provider/model)
  POST /config        — change LLM provider/model at runtime

Run:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Or via python:
  python api/app.py
"""

import sys
from pathlib import Path

# ── project root on path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # noqa: E402

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from llm_factory  import LLMConfig, load_config
from orchestrator import Orchestrator
import agents.serializability_agent as _serial_mod
import agents.parallel_query_agent  as _query_mod
from version import VERSION

import uvicorn

# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="DB Assistant Framework",
    description="Multi-agent system for parallel DB query cost analysis and schedule serializability",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global orchestrator (initialised on startup) ─────────────
_orchestrator: Orchestrator | None = None
_llm_config:   LLMConfig | None    = None


def _default_config() -> LLMConfig:
    cfg      = load_config()
    provider = cfg.get("default_provider", "openai")
    model    = cfg.get("default_model",    "gpt-5.4-mini")
    base_url = None
    n_predict = 2048

    if provider == "local":
        s = cfg.get("local_server", {})
        base_url  = f"http://{s.get('host','127.0.0.1')}:{s.get('port',9001)}"
        n_predict = s.get("n_predict", 2048)
    elif provider == "ollama":
        s = cfg.get("ollama", {})
        base_url = f"http://{s.get('host','127.0.0.1')}:{s.get('port',11434)}"

    return LLMConfig(provider=provider, model=model,
                     base_url=base_url, n_predict=n_predict)


@app.on_event("startup")
async def startup():
    global _orchestrator, _llm_config
    _llm_config   = _default_config()
    _orchestrator = Orchestrator(llm_config=_llm_config)
    print(f"[startup] LLM: {_llm_config.provider} / {_llm_config.model}")


# ══════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    domain:   str      # SERIAL | QUERY | UNKNOWN

class ConfigRequest(BaseModel):
    provider:  str
    model:     str
    base_url:  Optional[str] = None
    n_predict: Optional[int] = 2048

class ConfigResponse(BaseModel):
    provider: str
    model:    str
    base_url: Optional[str] = None


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "provider": _llm_config.provider, "model": _llm_config.model}


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        provider=_llm_config.provider,
        model=_llm_config.model,
        base_url=_llm_config.base_url,
    )


@app.post("/config", response_model=ConfigResponse)
async def set_config(req: ConfigRequest):
    global _orchestrator, _llm_config
    try:
        new_cfg       = LLMConfig(provider=req.provider, model=req.model,
                                   base_url=req.base_url, n_predict=req.n_predict or 2048)
        _orchestrator = Orchestrator(llm_config=new_cfg)
        _llm_config   = new_cfg
        return ConfigResponse(provider=new_cfg.provider, model=new_cfg.model,
                               base_url=new_cfg.base_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        # Skip routing LLM call when waiting for RA selection (user sends 1/2/3)
        if _orchestrator._pending_ra is not None:
            domain = "QUERY"
        else:
            domain = _orchestrator._route(req.message)
        response = _orchestrator.handle(req.message)
        return ChatResponse(response=response, domain=domain)
    except Exception as e:
        import traceback
        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(detail)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    global _orchestrator
    _orchestrator = Orchestrator(llm_config=_llm_config)
    return {"status": "reset", "message": "Conversation history cleared"}


_current_lang: str = "en"

class LangRequest(BaseModel):
    lang: str   # "ru" or "en"

@app.post("/language")
async def set_language(req: LangRequest):
    global _current_lang, _orchestrator
    if req.lang not in ("ru", "en"):
        raise HTTPException(status_code=400, detail="lang must be 'ru' or 'en'")
    _current_lang = req.lang
    # Update language in both agent modules — takes effect on next LLM call
    _serial_mod.set_agent_lang(req.lang)
    _query_mod.set_agent_lang(req.lang)
    # Reset histories so agents start fresh in the new language
    _orchestrator.serial_history              = []
    _orchestrator.query_history              = []
    _orchestrator.query_db_context           = {}
    _orchestrator.mapreduce_agent.history    = []
    return {"status": "ok", "lang": req.lang}


# ══════════════════════════════════════════════════════════════
#  UI  (single-page chat interface)
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_PAGE.replace("__VERSION__", VERSION)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DB Assistant Framework</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --border:   #1e2d45;
    --accent:   #3b82f6;
    --purple:   #8b5cf6;
    --green:    #22c55e;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'DM Sans', sans-serif;
  }

  [data-theme="light"] {
    --bg:      #f1f5f9;
    --surface: #ffffff;
    --border:  #cbd5e1;
    --accent:  #2563eb;
    --purple:  #7c3aed;
    --green:   #16a34a;
    --amber:   #d97706;
    --red:     #dc2626;
    --text:    #1e293b;
    --muted:   #64748b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── AGREEMENT OVERLAY ── */
  .agreement-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(8px);
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .agreement-box {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 18px;
    padding: 36px 40px;
    max-width: 520px;
    width: 90%;
    animation: fadeUp 0.3s ease;
    box-shadow: 0 0 60px rgba(59,130,246,0.15);
  }
  .agreement-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }
  .agreement-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .agreement-logo-text {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 16px;
  }
  .agreement-logo-text span { color: var(--accent); }
  .agreement-box h2 {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 0.4px;
    margin-bottom: 14px;
    text-transform: uppercase;
  }
  .agreement-box p {
    font-size: 13.5px;
    line-height: 1.7;
    color: var(--text);
    margin-bottom: 10px;
  }
  .agreement-box p.muted {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 0;
  }
  .agreement-features {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 14px 0;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .agreement-feature {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--text);
  }
  .agreement-feature .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
  }
  .btn-agree {
    width: 100%;
    background: var(--accent);
    border: none;
    color: white;
    border-radius: 10px;
    padding: 13px;
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 20px;
    transition: background 0.2s, transform 0.1s;
    letter-spacing: 0.2px;
  }
  .btn-agree:hover { background: #2563eb; }
  .btn-agree:active { transform: scale(0.98); }

  /* ── HEADER ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    height: 56px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .logo-icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
  }
  .logo-text {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 14px;
    letter-spacing: -0.3px;
  }
  .logo-text span { color: var(--accent); }
  .logo-version {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 400;
    color: var(--muted);
    letter-spacing: 0.2px;
    margin-left: 4px;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  /* Language toggle */
  .github-link {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--muted);
    text-decoration: none;
    font-size: 12px;
    font-family: var(--mono);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    transition: all 0.2s;
  }
  .github-link:hover { color: var(--text); border-color: var(--text); }

  .model-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .model-badge:hover { border-color: var(--accent); color: var(--text); }
  .model-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

  .btn-reset {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: var(--sans);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-reset:hover { border-color: var(--red); color: var(--red); }

  .theme-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 4px 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    transition: all 0.2s;
  }
  .theme-btn:hover { border-color: var(--accent); color: var(--text); }

  /* ── MAIN ── */
  main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── SIDEBAR ── */
  aside {
    width: 220px;
    border-right: 1px solid var(--border);
    background: var(--surface);
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex-shrink: 0;
    overflow-y: auto;
  }
  .sidebar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 4px 8px 2px;
  }
  .example-btn {
    background: transparent;
    border: none;
    color: var(--text);
    font-family: var(--sans);
    font-size: 12px;
    text-align: left;
    padding: 7px 10px;
    border-radius: 6px;
    cursor: pointer;
    line-height: 1.4;
    transition: background 0.15s;
    border-left: 2px solid transparent;
  }
  .example-btn:hover { background: var(--bg); border-left-color: var(--accent); }
  .example-btn.serial:hover { border-left-color: var(--purple); }
  .sidebar-divider { height: 1px; background: var(--border); margin: 6px 0; }

  /* ── CHAT ── */
  .chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }
  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg { display: flex; gap: 12px; animation: fadeUp 0.25s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg.user { flex-direction: row-reverse; }
  .avatar {
    width: 30px; height: 30px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg.user .avatar  { background: var(--accent); }
  .msg.agent .avatar { background: linear-gradient(135deg, var(--accent), var(--purple)); }

  .bubble {
    max-width: 72%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 13.5px;
    line-height: 1.65;
    white-space: pre-wrap;
  }
  .msg.user .bubble {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
    border-top-right-radius: 3px;
  }
  .msg.agent .bubble { border-top-left-radius: 3px; }
  .msg.agent .bubble pre {
    font-family: var(--mono);
    font-size: 12.5px;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 0;
    line-height: 1.6;
  }

  .domain-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.6px;
    padding: 2px 6px;
    border-radius: 4px;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .domain-SERIAL  { background: rgba(139,92,246,0.15); color: var(--purple); }
  .domain-QUERY   { background: rgba(59,130,246,0.15); color: var(--accent); }
  .domain-JOIN      { background: rgba(34,197,94,0.15);   color: var(--green); }
  .domain-MAPREDUCE { background: rgba(245,158,11,0.15);  color: var(--amber); }
  .domain-UNKNOWN   { background: rgba(100,116,139,0.15); color: var(--muted); }

  .thinking {
    display: flex; align-items: center; gap: 6px;
    color: var(--muted); font-size: 12px; font-family: var(--mono);
  }
  .thinking-dots span {
    display: inline-block; width: 5px; height: 5px;
    border-radius: 50%; background: var(--muted);
    animation: blink 1.2s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink {
    0%,80%,100% { opacity: 0.2; }
    40%          { opacity: 1; }
  }

  /* ── INPUT ── */
  .input-area {
    border-top: 1px solid var(--border);
    padding: 16px 24px;
    background: var(--surface);
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-shrink: 0;
  }
  #input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 13.5px;
    line-height: 1.5;
    resize: none;
    min-height: 42px;
    max-height: 140px;
    outline: none;
    transition: border-color 0.2s;
  }
  #input:focus { border-color: var(--accent); }
  #input::placeholder { color: var(--muted); }

  #send {
    background: var(--accent);
    border: none;
    color: white;
    border-radius: 10px;
    width: 42px; height: 42px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.2s, transform 0.1s;
    flex-shrink: 0;
  }
  #send:hover { background: #2563eb; }
  #send:active { transform: scale(0.94); }
  #send:disabled { background: var(--border); cursor: not-allowed; }

  /* ── MODEL MODAL ── */
  .modal-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 100;
    align-items: center; justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    width: 380px;
    animation: fadeUp 0.2s ease;
  }
  .modal h2 {
    font-family: var(--mono);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 16px;
    color: var(--accent);
  }
  .form-group { margin-bottom: 12px; }
  .form-group label {
    display: block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted);
    margin-bottom: 5px;
  }
  .form-group select,
  .form-group input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
  }
  .form-group select:focus,
  .form-group input:focus { border-color: var(--accent); }
  .modal-actions {
    display: flex; gap: 8px;
    margin-top: 16px; justify-content: flex-end;
  }
  .btn-cancel {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    cursor: pointer;
  }
  .btn-apply {
    background: var(--accent);
    border: none;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }

  .welcome {
    text-align: center;
    padding: 48px 24px;
    color: var(--muted);
  }
  .welcome h1 {
    font-family: var(--mono);
    font-size: 20px;
    color: var(--text);
    margin-bottom: 8px;
  }
  .welcome p { font-size: 13px; line-height: 1.6; max-width: 420px; margin: 0 auto; }
</style>
</head>
<body>

<!-- AGREEMENT OVERLAY -->
<div class="agreement-overlay" id="agreement">
  <div class="agreement-box">
    <div class="agreement-logo">
      <div class="agreement-logo-icon">⚙</div>
      <div class="agreement-logo-text">DB <span>Assistant</span> Framework</div>
    </div>
    <h2>Welcome — Please read before continuing</h2>
    <p>
      This is an AI-powered framework for solving tasks from the course
      <strong style="color:var(--accent)">Advanced Databases</strong>.
      It is designed to assist with academic exercises — not to replace understanding.
    </p>
    <div class="agreement-features">
      <div class="agreement-feature">
        <div class="dot" style="background:var(--accent)"></div>
        <span><strong>Parallel Query Cost Analysis</strong> — computes Elapsed and Total time for Select, Sort, and Join operations across distributed processors</span>
      </div>
      <div class="agreement-feature">
        <div class="dot" style="background:var(--purple)"></div>
        <span><strong>Schedule Serializability</strong> — checks view-serializability and conflict-serializability with full precedence graph analysis</span>
      </div>
      <div class="agreement-feature">
        <div class="dot" style="background:var(--green)"></div>
        <span><strong>Map-Reduce Design</strong> — generates visualization table, chain description, and map()/reduce() pseudocode for any task</span>
      </div>
      <div class="agreement-feature">
        <div class="dot" style="background:var(--amber)"></div>
        <span><strong>LLM-powered</strong> — works with GPT-4o, Claude, Ollama, or local models; all cost formulas are computed deterministically by Python tools</span>
      </div>
    </div>
    <p class="muted">
      Results may contain errors. Always verify answers independently.
      Built for educational purposes only.
    </p>
    <button class="btn-agree" onclick="acceptAgreement()">I understand — Enter the Framework</button>
  </div>
</div>

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-icon">⚙</div>
    <div class="logo-text">DB <span>Assistant</span> Framework<span class="logo-version">[__VERSION__]</span></div>
  </div>
  <div class="header-right">

    <!-- GitHub -->
    <a class="github-link" href="https://github.com/cyttic/agentADB" target="_blank" rel="noopener">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
      </svg>
      cyttic/agentADB
    </a>

    <div class="model-badge" onclick="openModal()">
      <div class="model-dot"></div>
      <span id="model-label">loading...</span>
    </div>
    <button class="theme-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle light/dark mode"></button>
    <button class="btn-reset" onclick="resetChat()" id="btn-reset">↺ Reset</button>
  </div>
</header>

<main>
  <!-- SIDEBAR -->
  <aside>
    <div class="sidebar-label" id="sb-query-label">Query Examples</div>
    <button class="example-btn" onclick="send('Table Flights(fid, date, from, to, seats). fid is the key. 10,000 blocks. 10 processors. Round Robin distribution. Find: σ(fid = 777)(Flights).')">
      σ(fid=777)(Flights)<br>Round Robin
    </button>
    <button class="example-btn" onclick="send('Table Students(sid, name, grade, year). sid is the key. 5,000 blocks. 10 processors. hash(sid) distribution. Find: σ(sid = 42)(Students).')">
      σ(sid=42)(Students)<br>Hash(sid)
    </button>
    <button class="example-btn" onclick="send('Table Orders(oid, cid, date, amount). oid is the key. 20,000 blocks. 10 processors. Range by oid distribution. Find: σ(oid != 100)(Orders).')">
      σ(oid≠100)(Orders)<br>Range(oid)
    </button>
    <button class="example-btn" onclick="send('Table Employees(eid, name, dept, salary). eid is the key. 8,000 blocks. 10 processors. Round Robin distribution. Perform sort by field salary.')">
      Sort by salary<br>Round Robin
    </button>
    <div class="sidebar-divider"></div>
    <div class="sidebar-label" id="sb-mr-label">Map-Reduce Examples</div>
    <button class="example-btn" onclick="send('Count how many times each word appears across documents d1..dn using n servers. Input: documents. Output: (word, count).')">
      Word count<br>n servers
    </button>
    <div class="sidebar-divider"></div>
    <div class="sidebar-label" id="sb-join-label">Join Examples</div>
    <button class="example-btn" onclick="send('Tables: Flowers(name, petal, size, color) — 10^4 blocks, Sales(name, cname, amount, price) — 10^6 blocks. 10 processors. Perform Flowers Join Sales.')">
      Flowers Join Sales<br>10^4 + 10^6 blocks
    </button>
    <div class="sidebar-divider"></div>
    <div class="sidebar-label" id="sb-serial-label">Serial Examples</div>
    <button class="example-btn serial" onclick="send('Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?')">
      Conflict check<br>r1(A) w2(A) r2(B) w1(B)
    </button>
    <button class="example-btn serial" onclick="send('Check view serializability: r2(B) w2(A) r1(A) r3(A) w1(B) w2(B) w3(B)')">
      View check<br>3-transaction schedule
    </button>
    <button class="example-btn serial" onclick="send('w1(A) w2(A) r3(A) w2(B) w3(B) — is this view serializable?')">
      View check<br>blind write schedule
    </button>
  </aside>

  <!-- CHAT -->
  <div class="chat-area">
    <div id="messages">
      <div class="welcome" id="welcome-msg">
        <h1 id="welcome-title">DB Assistant Framework</h1>
        <p id="welcome-sub">Ask about parallel query costs (Select, Sort, Join) or transaction schedule serializability. Pick an example from the sidebar or type your own task.</p>
      </div>
    </div>
    <div class="input-area">
      <textarea id="input" placeholder="Type a task or question…" rows="1"
        onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button id="send" onclick="sendMessage()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
  </div>
</main>

<!-- MODEL MODAL -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2>⚙ LLM Configuration</h2>
    <div class="form-group">
      <label>Provider</label>
      <select id="m-provider" onchange="updateModelList()">
        <option value="openai">openai</option>
        <option value="anthropic">anthropic</option>
        <option value="ollama">ollama</option>
        <option value="local">local (llama.cpp)</option>
      </select>
    </div>
    <div class="form-group">
      <label>Model</label>
      <select id="m-model"></select>
    </div>
    <div class="form-group" id="m-url-group" style="display:none">
      <label>Base URL</label>
      <input id="m-url" type="text" placeholder="http://127.0.0.1:9001">
    </div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeModal()" id="btn-cancel">Cancel</button>
      <button class="btn-apply" onclick="applyConfig()" id="btn-apply">Apply</button>
    </div>
  </div>
</div>

<script>
const MODELS = {
  openai:    ['gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.4'],
  anthropic: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  ollama:    ['llama3', 'llama3.1', 'mistral', 'phi3', 'qwen2', 'gemma2'],
  local:     ['local'],
};


// ── Agreement ─────────────────────────────────────────────────
function acceptAgreement() {
  document.getElementById('agreement').style.display = 'none';
  sessionStorage.setItem('agreed', '1');
}

function checkAgreement() {
  if (!sessionStorage.getItem('agreed')) {
    document.getElementById('agreement').style.display = 'flex';
  }
}

// ── Config ────────────────────────────────────────────────────
let isLoading = false;

async function loadConfig() {
  try {
    const r = await fetch('/config');
    const d = await r.json();
    document.getElementById('model-label').textContent = `${d.provider} / ${d.model}`;
  } catch(e) {
    document.getElementById('model-label').textContent = 'offline';
  }
}

// ── Send ──────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('input');
  const text  = input.value.trim();
  if (!text || isLoading) return;

  const msgs    = document.getElementById('messages');
  const welcome = msgs.querySelector('.welcome');
  if (welcome) welcome.remove();

  appendMsg('user', text, null);
  input.value = '';
  autoResize(input);

  const thinkId = appendThinking();
  isLoading = true;
  document.getElementById('send').disabled = true;

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text}),
    });
    const d = await r.json();
    removeThinking(thinkId);
    if (r.ok) appendMsg('agent', d.response, d.domain);
    else      appendMsg('agent', `Error: ${d.detail}`, 'UNKNOWN');
  } catch(e) {
    removeThinking(thinkId);
    appendMsg('agent', `Connection error: ${e.message}`, 'UNKNOWN');
  }

  isLoading = false;
  document.getElementById('send').disabled = false;
  input.focus();
}

function send(text) {
  document.getElementById('input').value = text;
  sendMessage();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

// ── Messages ──────────────────────────────────────────────────
function appendMsg(role, text, domain) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '⚙';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'agent' && domain) {
    const tag = document.createElement('div');
    tag.className = `domain-tag domain-${domain}`;
    tag.textContent = `Agent called [${domain}]`;
    bubble.appendChild(tag);
  }

  const content = document.createElement(role === 'agent' ? 'pre' : 'div');
  content.textContent = text;
  bubble.appendChild(content);

  div.appendChild(avatar);
  div.appendChild(bubble);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

let thinkCounter = 0;
function appendThinking() {
  const id   = ++thinkCounter;
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'msg agent';
  div.id = `think-${id}`;
  div.innerHTML = `
    <div class="avatar">⚙</div>
    <div class="bubble">
      <div class="thinking">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        thinking...
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(`think-${id}`);
  if (el) el.remove();
}

// ── Reset ─────────────────────────────────────────────────────
async function resetChat() {
  try { await fetch('/reset', {method: 'POST'}); } catch(e) {}
  const msgs = document.getElementById('messages');
  msgs.innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome';
  w.innerHTML = `
    <h1>DB Assistant Framework</h1>
    <p>Chat cleared. Ask about parallel query costs, Join cost, schedule serializability, or Map-Reduce.</p>`;
  msgs.appendChild(w);
}

// ── Model Modal ───────────────────────────────────────────────
function openModal()  { document.getElementById('modal').classList.add('open'); }
function closeModal() { document.getElementById('modal').classList.remove('open'); }

function updateModelList() {
  const provider = document.getElementById('m-provider').value;
  const sel      = document.getElementById('m-model');
  const urlGroup = document.getElementById('m-url-group');
  sel.innerHTML  = (MODELS[provider] || []).map(m => `<option>${m}</option>`).join('');
  urlGroup.style.display = (provider === 'local' || provider === 'ollama') ? 'block' : 'none';
}

async function applyConfig() {
  const provider = document.getElementById('m-provider').value;
  const model    = document.getElementById('m-model').value;
  const base_url = document.getElementById('m-url').value || null;
  const r = await fetch('/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({provider, model, base_url}),
  });
  if (r.ok) {
    closeModal();
    loadConfig();
  } else {
    const d = await r.json();
    alert(`Error: ${d.detail}`);
  }
}

document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// ── Theme ─────────────────────────────────────────────────────
const SUN_ICON  = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
const MOON_ICON = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

let theme = localStorage.getItem('db_theme') || 'dark';

function applyTheme(t) {
  theme = t;
  document.documentElement.setAttribute('data-theme', t);
  const btn = document.getElementById('theme-btn');
  if (t === 'dark') {
    btn.innerHTML = SUN_ICON;
    btn.title = 'Switch to light mode';
  } else {
    btn.innerHTML = MOON_ICON;
    btn.title = 'Switch to dark mode';
  }
  localStorage.setItem('db_theme', t);
}

function toggleTheme() {
  applyTheme(theme === 'dark' ? 'light' : 'dark');
}

// ── Init ──────────────────────────────────────────────────────
applyTheme(theme);
updateModelList();
loadConfig();
checkAgreement();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":    
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
