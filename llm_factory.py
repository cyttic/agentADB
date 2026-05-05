"""
llm_factory.py
===============
Universal LLM selector supporting:
  - OpenAI      (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, ...)
  - Anthropic   (claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5, ...)
  - Local/Ollama (llama3, mistral, phi3, ...)

Usage:
    from llm_factory import build_llm, LLMConfig

    # OpenAI
    llm = build_llm(LLMConfig(provider="openai", model="gpt-4o"))

    # Anthropic
    llm = build_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-5"))

    # Local via Ollama
    llm = build_llm(LLMConfig(provider="ollama", model="llama3"))

Environment variables required per provider:
  - OpenAI:    OPENAI_API_KEY
  - Anthropic: ANTHROPIC_API_KEY
  - Ollama:    none (runs locally, default base_url=http://localhost:11434)
"""

import os
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    provider:    str   = "openai"       # "openai" | "anthropic" | "ollama"
    model:       str   = "gpt-4o"
    temperature: float = 0.0
    # Optional overrides
    api_key:     str | None = None      # overrides env var if set
    base_url:    str | None = None      # for Ollama or custom OpenAI-compatible endpoints


# ══════════════════════════════════════════════════════════════
#  PROVIDER NOTES (shown in selector UI)
# ══════════════════════════════════════════════════════════════

PROVIDER_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
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
}


# ══════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════

def build_llm(config: LLMConfig):
    """
    Build and return a LangChain-compatible chat model from config.
    Raises ImportError if the required package is not installed.
    Raises ValueError for unknown providers.
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

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: openai, anthropic, ollama."
        )


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE SELECTOR  (terminal UI)
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def select_llm_interactive() -> LLMConfig:
    """
    Interactive terminal prompt to select provider + model.
    Returns a ready LLMConfig.
    """
    print(f"\n{BOLD}{CYAN}═══ LLM Configuration ═══{RESET}")

    # ── Choose provider ──────────────────────────────────────
    providers = list(PROVIDER_MODELS.keys())
    print(f"\n{BOLD}Select provider:{RESET}")
    for i, p in enumerate(providers, 1):
        print(f"  {BOLD}{CYAN}{i}{RESET}. {p}")

    while True:
        try:
            choice = input(f"\n{BOLD}Provider [1-{len(providers)}]: {RESET}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(providers):
                provider = providers[idx]
                break
        except (ValueError, IndexError):
            pass
        print(f"  {YELLOW}Please enter a number between 1 and {len(providers)}{RESET}")

    # ── Choose model ─────────────────────────────────────────
    suggested = PROVIDER_MODELS[provider]
    print(f"\n{BOLD}Select model for {CYAN}{provider}{RESET}{BOLD}:{RESET}")
    for i, m in enumerate(suggested, 1):
        print(f"  {BOLD}{CYAN}{i}{RESET}. {m}")
    print(f"  {BOLD}{CYAN}c{RESET}. Custom (type your own)")

    while True:
        choice = input(f"\n{BOLD}Model [1-{len(suggested)}/c]: {RESET}").strip().lower()
        if choice == "c":
            model = input(f"{BOLD}Enter model name: {RESET}").strip()
            if model:
                break
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(suggested):
                    model = suggested[idx]
                    break
            except (ValueError, IndexError):
                pass
        print(f"  {YELLOW}Invalid choice, try again{RESET}")

    # ── Optional: custom base_url (for Ollama or custom endpoints) ──
    base_url = None
    if provider == "ollama":
        default_url = "http://localhost:11434"
        raw = input(f"\n{BOLD}Ollama base URL [{DIM}{default_url}{RESET}{BOLD}]: {RESET}").strip()
        base_url = raw if raw else default_url

    # ── Confirm ──────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}✓ Using:{RESET} {BOLD}{provider}{RESET} / {BOLD}{model}{RESET}")
    if base_url:
        print(f"  Base URL: {DIM}{base_url}{RESET}")

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
    )
