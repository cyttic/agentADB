"""
main.py
========
Entry point — model selector + interactive loop.

Usage:
    export OPENAI_API_KEY=sk-...        # for OpenAI
    export ANTHROPIC_API_KEY=sk-ant-... # for Anthropic
    python main.py
"""

from llm_factory  import select_llm_interactive
from orchestrator import Orchestrator

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
DIM    = "\033[2m"
YELLOW = "\033[33m"


def run():
    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}   Parallel DB & Serializability Assistant     {RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════{RESET}")
    print(f"{DIM}  Three agents · One orchestrator · LangGraph  {RESET}")

    # ── Step 1: choose LLM ───────────────────────────────────
    # NOTE: model quality matters — weaker models (e.g. gpt-3.5, phi3)
    # may produce less structured output. gpt-4o / claude-sonnet recommended.
    llm_config = select_llm_interactive()

    # ── Step 2: boot orchestrator ────────────────────────────
    print(f"\n{DIM}Initialising agents...{RESET}", end="", flush=True)
    orchestrator = Orchestrator(llm_config=llm_config)
    print(f"\r{GREEN}✓ Agents ready.{RESET}              \n")

    print(f"{DIM}Type 'exit' to quit · 'model' to switch LLM{RESET}\n")
    print("Example queries:")
    print("  • Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?")
    print("  • Дана таблица Flights, 10000 блоков, round-robin, 10 прocs. σ_fid=777(Flights)")
    print("  • Flowers(name,petal,size,color) 10^4 блоков, Sales(name,cname,amount,price) 10^6 блоков, 10 серверов. Выполнить Flowers join Sales.")
    print()

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        if not user_input:
            continue

        # ── Switch model mid-session ─────────────────────────
        if user_input.lower() == "model":
            llm_config   = select_llm_interactive()
            print(f"\n{DIM}Reinitialising agents...{RESET}", end="", flush=True)
            orchestrator = Orchestrator(llm_config=llm_config)
            print(f"\r{GREEN}✓ Agents restarted with new model.{RESET}   \n")
            continue

        response = orchestrator.handle(user_input)
        print(f"\n{BOLD}{CYAN}Agent:{RESET} {response}\n")


if __name__ == "__main__":
    run()
