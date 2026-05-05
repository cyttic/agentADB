"""
evals/run_evals.py
===================
Evaluation runner for the DB Assistant multi-agent system.

- Loads tasks from evals/test_cases.json
- Runs each task through the agent in single-turn mode (no history)
- Uses an LLM judge to compare agent answer vs expected answer
- Prints a full report to terminal

Default model: read from config.json (default_provider + default_model).
Override via CLI:
    python evals/run_evals.py
    python evals/run_evals.py --provider openai --model gpt-4o
    python evals/run_evals.py --difficulty low        # filter by difficulty
    python evals/run_evals.py --id 1 --id 3           # run specific tasks only
"""

import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# ── make sure project root is on the path ───────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llm_factory  import build_llm, LLMConfig, load_config
from orchestrator import Orchestrator


# ══════════════════════════════════════════════════════════════
#  ANSI
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
DIM    = "\033[2m"
MAGENTA= "\033[35m"


# ══════════════════════════════════════════════════════════════
#  JUDGE PROMPT
# ══════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are an expert database systems evaluator.

Your job: compare a student agent's answer to the expected answer for a database task.
The wording may differ — judge by MEANING and CORRECTNESS, not exact match.

Focus on:
  - Is the algorithm choice correct? (alg2 vs alg3, round-robin vs hash, etc.)
  - Are the Elapsed (E) and Total (T) cost formulas correct in structure?
  - Are key values correct? (block counts, multipliers, t_d / t_s terms)

Be lenient on:
  - Different notation (E vs Elapsed, * vs x, 10^3 vs 1000)
  - Extra explanation text in the agent answer
  - Language differences (Russian vs English)

Be strict on:
  - Wrong algorithm choice
  - Wrong formula structure (e.g. missing t_s when it should be there)
  - Wrong numeric values

Respond ONLY with a JSON object, no extra text:
{
  "verdict": "PASS" or "FAIL",
  "reason": "one sentence explaining the verdict"
}
"""


# ══════════════════════════════════════════════════════════════
#  SINGLE-TURN AGENT CALL
# ══════════════════════════════════════════════════════════════

def run_single_turn(orchestrator: Orchestrator, task_text: str) -> str:
    """
    Run one task through the orchestrator with no history.
    Each call is fully independent — fresh state every time.
    Returns the agent's text response.
    """
    # Temporarily clear histories so there's no bleed between tasks
    orig_serial  = orchestrator.serial_history
    orig_query   = orchestrator.query_history
    orig_context = orchestrator.query_db_context

    orchestrator.serial_history   = []
    orchestrator.query_history    = []
    orchestrator.query_db_context = {}

    try:
        response = orchestrator.handle(task_text)
    finally:
        # Restore original histories
        orchestrator.serial_history   = orig_serial
        orchestrator.query_history    = orig_query
        orchestrator.query_db_context = orig_context

    return response


# ══════════════════════════════════════════════════════════════
#  LLM JUDGE
# ══════════════════════════════════════════════════════════════

def judge(llm, task_text: str, expected: str, agent_answer: str) -> dict:
    """
    Ask the LLM judge to compare agent_answer vs expected.
    Returns {"verdict": "PASS"|"FAIL", "reason": "..."}
    """
    user_msg = f"""Task:
{task_text}

Expected answer:
{expected}

Agent answer:
{agent_answer}
"""
    response = llm.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=user_msg),
    ])

    raw = response.content.strip()

    # Strip markdown fences if model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        if "verdict" not in result:
            raise ValueError("missing verdict")
        result["verdict"] = result["verdict"].upper().strip()
        return result
    except Exception:
        # Fallback: scan for PASS/FAIL keyword
        upper = raw.upper()
        verdict = "PASS" if "PASS" in upper else "FAIL"
        return {"verdict": verdict, "reason": raw[:200]}


# ══════════════════════════════════════════════════════════════
#  REPORT HELPERS
# ══════════════════════════════════════════════════════════════

def _wrap(text: str, width: int = 80, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, width=width, subsequent_indent=indent)
        for line in text.splitlines()
    )


def print_header(total: int, provider: str, model: str):
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  DB Assistant — Eval Runner{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"  Model   : {BOLD}{provider} / {model}{RESET}")
    print(f"  Tasks   : {BOLD}{total}{RESET}")
    print(f"  Started : {DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")


def print_task_result(task: dict, agent_answer: str, judgment: dict, idx: int, total: int):
    verdict  = judgment.get("verdict", "FAIL")
    reason   = judgment.get("reason", "")
    is_pass  = verdict == "PASS"

    verdict_str = f"{GREEN}{BOLD}PASS ✓{RESET}" if is_pass else f"{RED}{BOLD}FAIL ✗{RESET}"
    diff_color  = {"low": GREEN, "medium": YELLOW, "high": RED}.get(
        task.get("difficulty", ""), DIM
    )

    print(f"{BOLD}Task {task['id']}{RESET}  [{idx}/{total}]  "
          f"difficulty: {diff_color}{task.get('difficulty','?')}{RESET}  "
          f"→  {verdict_str}")
    print(f"  {DIM}Task   :{RESET} {_wrap(task['task'][:120] + ('...' if len(task['task']) > 120 else ''), indent='           ')}")
    print(f"  {DIM}Expected:{RESET} {YELLOW}{task['answer']}{RESET}")
    print(f"  {DIM}Agent   :{RESET} {_wrap(agent_answer[:300] + ('...' if len(agent_answer) > 300 else ''), indent='           ')}")
    print(f"  {DIM}Judge   :{RESET} {reason}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")


def print_summary(results: list):
    total    = len(results)
    passed   = sum(1 for r in results if r["verdict"] == "PASS")
    failed   = total - passed
    score    = (passed / total * 100) if total else 0

    by_diff = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"pass": 0, "fail": 0})
        if r["verdict"] == "PASS":
            by_diff[d]["pass"] += 1
        else:
            by_diff[d]["fail"] += 1

    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}  EVAL SUMMARY{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")
    print(f"  Total  : {BOLD}{total}{RESET}")
    print(f"  Passed : {GREEN}{BOLD}{passed}{RESET}")
    print(f"  Failed : {RED}{BOLD}{failed}{RESET}")

    color = GREEN if score >= 80 else (YELLOW if score >= 50 else RED)
    print(f"  Score  : {color}{BOLD}{score:.1f}%{RESET}")

    if by_diff:
        print(f"\n  By difficulty:")
        for diff, counts in sorted(by_diff.items()):
            t = counts["pass"] + counts["fail"]
            pct = counts["pass"] / t * 100 if t else 0
            bar_color = {"low": GREEN, "medium": YELLOW, "high": RED}.get(diff, DIM)
            print(f"    {bar_color}{diff:8}{RESET}  {counts['pass']}/{t}  ({pct:.0f}%)")

    # Failed task IDs
    failed_ids = [str(r["id"]) for r in results if r["verdict"] == "FAIL"]
    if failed_ids:
        print(f"\n  Failed task IDs: {RED}{', '.join(failed_ids)}{RESET}")

    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}\n")


# ══════════════════════════════════════════════════════════════
#  DEFAULT MODEL FROM CONFIG
# ══════════════════════════════════════════════════════════════

def default_llm_config() -> LLMConfig:
    """
    Read default provider + model from config.json.
    Falls back to gpt-5.4-nano / openai if not set.
    """
    cfg      = load_config()
    provider = cfg.get("default_provider", "openai")
    model    = cfg.get("default_model",    "gpt-5.4-nano")

    base_url  = None
    n_predict = 2048

    if provider == "local":
        saved     = cfg.get("local_server", {})
        host      = saved.get("host", "127.0.0.1")
        port      = saved.get("port", 9001)
        base_url  = f"http://{host}:{port}"
        n_predict = saved.get("n_predict", 2048)
    elif provider == "ollama":
        saved    = cfg.get("ollama", {})
        host     = saved.get("host", "127.0.0.1")
        port     = saved.get("port", 11434)
        base_url = f"http://{host}:{port}"

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        n_predict=n_predict,
    )


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DB Assistant Eval Runner")
    parser.add_argument("--provider",    default=None, help="Override LLM provider")
    parser.add_argument("--model",       default=None, help="Override LLM model")
    parser.add_argument("--difficulty",  default=None, help="Filter by difficulty: low/medium/high")
    parser.add_argument("--id",          type=int, action="append", dest="ids",
                        help="Run specific task IDs only (repeatable: --id 1 --id 3)")
    parser.add_argument("--test-file",   default=str(ROOT / "evals" / "test_cases.json"),
                        help="Path to test_cases.json")
    args = parser.parse_args()

    # ── Load test cases ──────────────────────────────────────
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"{RED}Error: test file not found: {test_file}{RESET}")
        sys.exit(1)

    with open(test_file) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])

    # Apply filters
    if args.ids:
        tasks = [t for t in tasks if t["id"] in args.ids]
    if args.difficulty:
        tasks = [t for t in tasks if t.get("difficulty") == args.difficulty]

    if not tasks:
        print(f"{YELLOW}No tasks matched the filters.{RESET}")
        sys.exit(0)

    # ── Build LLM config ─────────────────────────────────────
    llm_cfg = default_llm_config()
    if args.provider:
        llm_cfg.provider = args.provider
    if args.model:
        llm_cfg.model = args.model

    # ── Boot orchestrator + judge LLM ────────────────────────
    print(f"{DIM}Booting agents...{RESET}", end="", flush=True)
    orchestrator = Orchestrator(llm_config=llm_cfg)
    judge_llm    = build_llm(llm_cfg)   # same model judges the answers
    print(f"\r{GREEN}✓ Ready.{RESET}           \n")

    print_header(len(tasks), llm_cfg.provider, llm_cfg.model)

    # ── Run tasks ────────────────────────────────────────────
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"{DIM}Running task {task['id']}...{RESET}", end="\r", flush=True)

        # Single-turn agent call
        try:
            agent_answer = run_single_turn(orchestrator, task["task"])
        except Exception as e:
            agent_answer = f"[ERROR: {e}]"

        # LLM judge
        try:
            judgment = judge(judge_llm, task["task"], task["answer"], agent_answer)
        except Exception as e:
            judgment = {"verdict": "FAIL", "reason": f"Judge error: {e}"}

        print_task_result(task, agent_answer, judgment, i, len(tasks))

        results.append({
            "id":           task["id"],
            "difficulty":   task.get("difficulty", "unknown"),
            "verdict":      judgment.get("verdict", "FAIL"),
            "reason":       judgment.get("reason", ""),
            "agent_answer": agent_answer,
            "expected":     task["answer"],
        })

    # ── Summary ───────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
