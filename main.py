"""
main.py
========
Entry point — interactive loop for the multi-agent system.

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py
"""

from orchestrator import Orchestrator

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
DIM    = "\033[2m"


def run():
    print(f"\n{BOLD}{CYAN}═══ Parallel DB & Serializability Assistant ═══{RESET}")
    print(f"{DIM}Two agents, one orchestrator — powered by LangGraph + GPT-4o{RESET}")
    print(f"{DIM}Type 'exit' to quit{RESET}\n")
    print("Example queries:")
    print("  • Is r1(A) w2(A) r2(B) w1(B) conflict-serializable?")
    print("  • Check view-serializability: r2(B) w2(A) r1(A) r3(A) w1(B) w2(B) w3(B)")
    print("  • 10 processors, block size 2000, Customers 10^6 rows — query cost?")
    print()

    orchestrator = Orchestrator()

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

        response = orchestrator.handle(user_input)
        print(f"\n{BOLD}{CYAN}Agent:{RESET} {response}\n")


if __name__ == "__main__":
    run()
