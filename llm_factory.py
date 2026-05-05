"""
llm_factory.py
===============
Universal LLM selector supporting:
  - openai    — OpenAI API (gpt-4o, gpt-4o-mini, ...)
  - anthropic — Anthropic Claude (claude-opus-4-5, claude-sonnet-4-5, ...)
  - ollama    — Ollama local server (llama3, mistral, phi3, ...)
  - local     — llama.cpp server  (curl-style /completion endpoint)
                host/port read from config.json

Config file: config.json (sits next to this file)
  Stores preferred local server host, port, and defaults.

Environment variables:
  OPENAI_API_KEY    — for openai
  ANTHROPIC_API_KEY — for anthropic
  (none needed for ollama / local)
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path


# ══════════════════════════════════════════════════════════════
#  CONFIG FILE
# ══════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    """Load config.json. Returns empty dict if file missing."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Persist config.json."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved to {CONFIG_PATH}")


# ══════════════════════════════════════════════════════════════
#  LLM CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    provider:    str   = "openai"
    model:       str   = "gpt-4o"
    temperature: float = 0.0
    api_key:     str   = None    # overrides env var if set
    base_url:    str   = None    # full base URL (built from host+port for local/ollama)
    n_predict:   int   = 2048    # llama.cpp specific: max tokens to generate


# ══════════════════════════════════════════════════════════════
#  LLAMA.CPP WRAPPER
# ══════════════════════════════════════════════════════════════

class LlamaCppChat:
    """
    Minimal LangChain-compatible wrapper around llama.cpp /completion endpoint.

    Mirrors the curl call:
      curl http://HOST:PORT/completion -d '{"prompt": "...", "n_predict": N}'

    Supports .invoke() and .bind_tools() (tools are injected into the prompt
    as a JSON description since llama.cpp doesn't have native tool-calling).
    """

    def __init__(self, base_url: str, n_predict: int = 2048, temperature: float = 0.0):
        self.base_url    = base_url.rstrip("/")
        self.n_predict   = n_predict
        self.temperature = temperature
        self._tools      = []

    def bind_tools(self, tools):
        """Return a copy with tools bound (injected into prompt as descriptions)."""
        clone          = LlamaCppChat(self.base_url, self.n_predict, self.temperature)
        clone._tools   = tools
        return clone

    def _build_prompt(self, messages) -> str:
        """
        Convert LangChain messages to a single prompt string.
        Appends tool descriptions if tools are bound.
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        parts = []

        # Tool descriptions injected into system context
        if self._tools:
            tool_desc = "\n".join(
                f"- {t.name}: {t.description}" for t in self._tools
            )
            parts.append(f"[Available tools]\n{tool_desc}\n")

        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"[System]\n{msg.content}")
            elif isinstance(msg, HumanMessage):
                parts.append(f"[User]\n{msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"[Assistant]\n{msg.content}")
            elif isinstance(msg, ToolMessage):
                parts.append(f"[Tool result: {msg.name}]\n{msg.content}")
            else:
                parts.append(str(msg.content))

        parts.append("[Assistant]")
        return "\n\n".join(parts)

    def invoke(self, messages):
        """Send request to llama.cpp /completion and return AIMessage."""
        import urllib.request
        from langchain_core.messages import AIMessage

        prompt  = self._build_prompt(messages)
        payload = json.dumps({
            "prompt":      prompt,
            "n_predict":   self.n_predict,
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data    = json.loads(resp.read())
                content = data.get("content", "").strip()
        except Exception as e:
            content = f"[LlamaCpp error: {e}]"

        return AIMessage(content=content)


# ══════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════

def build_llm(config: LLMConfig):
    """
    Build and return a LangChain-compatible chat model from LLMConfig.
    """
    provider = config.provider.lower().strip()

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
            **({"base_url": config.base_url} if config.base_url else {}),
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Run: pip install langchain-anthropic")
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Run: pip install langchain-ollama")
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    elif provider == "local":
        # llama.cpp native /completion endpoint
        if not config.base_url:
            raise ValueError("base_url is required for local provider.")
        return LlamaCppChat(
            base_url=config.base_url,
            n_predict=config.n_predict,
            temperature=config.temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose: openai, anthropic, ollama, local."
        )


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE SELECTOR
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"

PROVIDERS = ["openai", "anthropic", "ollama", "local"]

# Models per provider — edit here to add/remove options
PROVIDER_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
    "anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
    "ollama": [
        "llama3",
        "llama3.1",
        "mistral",
        "phi3",
        "qwen2",
        "gemma2",
    ],
    # local has no model list — uses config.json settings directly
}


def _local_label(cfg: dict) -> str:
    """Build display label for local option, e.g. 'local[127.0.0.1:9001]'."""
    saved = cfg.get("local_server", {})
    host  = saved.get("host", "127.0.0.1")
    port  = saved.get("port", 9001)
    return f"local[{host}:{port}]"


def _ollama_label(cfg: dict) -> str:
    """Build display label for ollama option, e.g. 'ollama[127.0.0.1:11434]'."""
    saved = cfg.get("ollama", {})
    host  = saved.get("host", "127.0.0.1")
    port  = saved.get("port", 11434)
    return f"ollama[{host}:{port}]"


def select_llm_interactive() -> LLMConfig:
    """
    Interactive terminal prompt to select provider + model.

    - openai / anthropic: shows model list, asks to pick one.
    - ollama:             shows host:port from config, asks model name.
    - local:              reads ALL settings from config.json silently —
                          no questions asked, just confirm and go.

    Returns a ready LLMConfig.
    """
    cfg = load_config()

    print(f"\n{BOLD}{CYAN}=== LLM Configuration ==={RESET}")

    # Build display labels
    labels = {
        "openai":    "openai",
        "anthropic": "anthropic",
        "ollama":    _ollama_label(cfg),
        "local":     _local_label(cfg),
    }

    default_provider = cfg.get("default_provider", "openai")
    print(f"\n{BOLD}Select provider:{RESET}")
    for i, p in enumerate(PROVIDERS, 1):
        tag   = f" {DIM}(default){RESET}" if p == default_provider else ""
        print(f"  {BOLD}{CYAN}{i}{RESET}. {labels[p]}{tag}")

    while True:
        try:
            raw = input(f"\n{BOLD}Provider [1-{len(PROVIDERS)}]: {RESET}").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(PROVIDERS):
                provider = PROVIDERS[idx]
                break
        except (ValueError, IndexError):
            pass
        print(f"  {YELLOW}Enter a number 1-{len(PROVIDERS)}{RESET}")

    # ── Local: read everything from config, ask nothing ──────
    if provider == "local":
        saved     = cfg.get("local_server", {})
        host      = saved.get("host", "127.0.0.1")
        port      = saved.get("port", 9001)
        n_predict = saved.get("n_predict", 2048)
        base_url  = f"http://{host}:{port}"
        model     = "local"
        print(f"\n{BOLD}{GREEN}Using:{RESET} {BOLD}{_local_label(cfg)}{RESET}  "
              f"{DIM}(n_predict={n_predict}, edit config.json to change){RESET}")
        return LLMConfig(provider=provider, model=model, base_url=base_url, n_predict=n_predict)

    # ── Ollama: host from config, ask model ──────────────────
    if provider == "ollama":
        saved     = cfg.get("ollama", {})
        host      = saved.get("host", "127.0.0.1")
        port      = saved.get("port", 11434)
        base_url  = f"http://{host}:{port}"
        suggested = PROVIDER_MODELS["ollama"]

        print(f"\n{BOLD}Select model:{RESET}")
        for i, m in enumerate(suggested, 1):
            print(f"  {BOLD}{CYAN}{i}{RESET}. {m}")
        print(f"  {BOLD}{CYAN}c{RESET}. Custom")

        while True:
            raw = input(f"{BOLD}Model [1-{len(suggested)}/c]: {RESET}").strip().lower()
            if raw == "c":
                model = input(f"{BOLD}Enter model name: {RESET}").strip() or "custom"
                break
            try:
                model = suggested[int(raw) - 1]
                break
            except (ValueError, IndexError):
                print(f"  {YELLOW}Invalid, try again{RESET}")

        print(f"\n{BOLD}{GREEN}Using:{RESET} {BOLD}{_ollama_label(cfg)}{RESET} / {BOLD}{model}{RESET}")
        return LLMConfig(provider=provider, model=model, base_url=base_url)

    # ── OpenAI / Anthropic: show model list ──────────────────
    suggested = PROVIDER_MODELS.get(provider, [])

    print(f"\n{BOLD}Select model:{RESET}")
    for i, m in enumerate(suggested, 1):
        print(f"  {BOLD}{CYAN}{i}{RESET}. {m}")
    print(f"  {BOLD}{CYAN}c{RESET}. Custom")

    while True:
        raw = input(f"{BOLD}Model [1-{len(suggested)}/c]: {RESET}").strip().lower()
        if raw == "c":
            model = input(f"{BOLD}Enter model name: {RESET}").strip()
            if model:
                break
        else:
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(suggested):
                    model = suggested[idx]
                    break
            except (ValueError, IndexError):
                pass
        print(f"  {YELLOW}Invalid, try again{RESET}")

    print(f"\n{BOLD}{GREEN}Using:{RESET} {BOLD}{provider}{RESET} / {BOLD}{model}{RESET}")
    return LLMConfig(provider=provider, model=model)
