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
import json
from pathlib import Path

# ── project root on path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from llm_factory  import build_llm, LLMConfig, load_config
from orchestrator import Orchestrator


# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="DB Assistant Framework",
    description="Multi-agent system for parallel DB query cost analysis and schedule serializability",
    version="1.0.0",
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
    model    = cfg.get("default_model",    "gpt-4o")
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
        # Capture the domain from routing
        domain   = _orchestrator._route(req.message)
        response = _orchestrator.handle(req.message)
        return ChatResponse(response=response, domain=domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    global _orchestrator
    _orchestrator = Orchestrator(llm_config=_llm_config)
    return {"status": "reset", "message": "Conversation history cleared"}


# ══════════════════════════════════════════════════════════════
#  UI  (single-page chat interface)
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML_PAGE


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

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

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
  .example-btn.serial { border-left-color: transparent; }
  .example-btn.serial:hover { border-left-color: var(--purple); }

  .sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 6px 0;
  }

  /* ── CHAT AREA ── */
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

  .msg {
    display: flex;
    gap: 12px;
    animation: fadeUp 0.25s ease;
  }

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
  .domain-UNKNOWN { background: rgba(100,116,139,0.15); color: var(--muted); }

  .thinking {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--mono);
  }
  .thinking-dots span {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--muted);
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

  .form-group {
    margin-bottom: 12px;
  }
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
    display: flex;
    gap: 8px;
    margin-top: 16px;
    justify-content: flex-end;
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

  /* ── WELCOME ── */
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

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-icon">⚙</div>
    <div class="logo-text">DB <span>Assistant</span> Framework</div>
  </div>
  <div class="header-right">
    <div class="model-badge" onclick="openModal()">
      <div class="model-dot"></div>
      <span id="model-label">loading...</span>
    </div>
    <button class="btn-reset" onclick="resetChat()">↺ Reset</button>
  </div>
</header>

<main>
  <!-- SIDEBAR -->
  <aside>
    <div class="sidebar-label">Query Examples</div>
    <button class="example-btn" onclick="send('Дана таблица Flights(fid, date, from, to, seats). fid — ключ. 10,000 блоков. 10 процессоров. Round Robin. Найти: σ_{fid = 777}(Flights).')">
      σ_{fid=777}(Flights)<br>Round Robin
    </button>
    <button class="example-btn" onclick="send('Дана таблица Students(sid, name, grade, year). sid — ключ. 5,000 блоков. 10 процессоров. Распределение: hash(sid). Найти: σ_{sid = 42}(Students).')">
      σ_{sid=42}(Students)<br>Hash(sid)
    </button>
    <button class="example-btn" onclick="send('Дана таблица Orders(oid, cid, date, amount). oid — ключ. 20,000 блоков. 10 процессоров. Range by oid. Найти: σ_{oid != 100}(Orders).')">
      σ_{oid≠100}(Orders)<br>Range(oid)
    </button>
    <button class="example-btn" onclick="send('Дана таблица Employees(eid, name, dept, salary). eid — ключ. 8,000 блоков. 10 процессоров. Round Robin. Выполнить сортировку по полю salary.')">
      Sort by salary<br>Round Robin
    </button>
    <div class="sidebar-divider"></div>
    <div class="sidebar-label">Serial Examples</div>
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
      <div class="welcome">
        <h1>DB Assistant Framework</h1>
        <p>Ask about parallel query costs (Select, Sort, Join) or transaction schedule serializability. Pick an example from the sidebar or type your own task.</p>
      </div>
    </div>
    <div class="input-area">
      <textarea id="input" placeholder="Type a task or question..." rows="1"
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
      <button class="btn-cancel" onclick="closeModal()">Cancel</button>
      <button class="btn-apply" onclick="applyConfig()">Apply</button>
    </div>
  </div>
</div>

<script>
const MODELS = {
  openai:    ['gpt-4o', 'gpt-4o-mini', 'gpt-5.4-mini', 'gpt-5.4-nano'],
  anthropic: ['claude-opus-4-5', 'claude-sonnet-4-5', 'claude-haiku-4-5'],
  ollama:    ['llama3', 'llama3.1', 'mistral', 'phi3', 'qwen2', 'gemma2'],
  local:     ['local'],
};

let isLoading = false;

// ── load current config ──────────────────────────────────────
async function loadConfig() {
  try {
    const r = await fetch('/config');
    const d = await r.json();
    document.getElementById('model-label').textContent = `${d.provider} / ${d.model}`;
  } catch(e) {
    document.getElementById('model-label').textContent = 'offline';
  }
}

// ── send message ─────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('input');
  const text  = input.value.trim();
  if (!text || isLoading) return;

  // Clear welcome
  const msgs = document.getElementById('messages');
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
    if (r.ok) {
      appendMsg('agent', d.response, d.domain);
    } else {
      appendMsg('agent', `Error: ${d.detail}`, 'UNKNOWN');
    }
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

// ── messages ─────────────────────────────────────────────────
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
    tag.textContent = domain;
    bubble.appendChild(tag);
  }

  const content = document.createElement('div');
  content.textContent = text;
  bubble.appendChild(content);

  div.appendChild(avatar);
  div.appendChild(bubble);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
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

// ── reset ────────────────────────────────────────────────────
async function resetChat() {
  await fetch('/reset', {method: 'POST'});
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div class="welcome">
      <h1>DB Assistant Framework</h1>
      <p>Conversation reset. Ask about parallel query costs or schedule serializability.</p>
    </div>`;
}

// ── model modal ──────────────────────────────────────────────
function openModal() { document.getElementById('modal').classList.add('open'); }
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
    document.getElementById('model-label').textContent = `${provider} / ${model}`;
  } else {
    const d = await r.json();
    alert(`Error: ${d.detail}`);
  }
}

// close modal on overlay click
document.getElementById('modal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// init
updateModelList();
loadConfig();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
