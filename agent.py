"""
My Deep Agent: a personal assistant built with LangChain Deep Agents + Gemini.

The four layers in this file:
  MODEL      Google Gemini (free tier)          -> make_model()
  SDK/API    langchain-google-genai (calls Google's Gemini API for you)
  HARNESS    LangChain Deep Agents              -> create_deep_agent(...)
  YOUR AGENT 2 custom tools + a system prompt + a notes folder

Compared with the hand-written version, notice what's MISSING here:
there is no agent loop. Deep Agents runs the loop for you.

Run it:   .venv/bin/python agent.py
(Your GOOGLE_API_KEY is read from the .env file next to this script.)
"""

import ast
import os
import operator
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(Path(__file__).parent / ".env")  # puts GOOGLE_API_KEY into the environment
WORKSPACE = Path(__file__).parent / "workspace"  # the only folder the agent can read/write


# ---------------------------------------------------------------------------
# LAYER 1 - MODEL: Gemini, via LangChain's Google connector.
# Swapping the model (Claude, GPT, Ollama...) only means changing this function.
# (It used to be ChatAnthropic(model="claude-opus-5-5") - nothing else changed.)
# ---------------------------------------------------------------------------

def make_model():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        max_output_tokens=16000,
        max_retries=0,  # retries are handled (visibly) by the middleware in build_agent()
    )


# ---------------------------------------------------------------------------
# YOUR TOOLS: plain Python functions. Deep Agents reads the function name,
# the type hints, and the docstring, and turns them into the tool description
# the model sees. (In the hand-written version you wrote that JSON by hand.)
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression like '(19.99 * 3) * 1.08'.
    Use this for any math instead of doing it in your head."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("Only numbers and + - * / ** % are allowed")
    return str(walk(ast.parse(expression, mode="eval").body))

# Notice: no save_note / read_notes tools. Deep Agents comes with built-in
# file tools (ls, read_file, write_file, edit_file, glob, grep), so the agent
# keeps notes as real files in the workspace folder.


SYSTEM_PROMPT = """You are a friendly personal assistant. Keep answers short.

When the user asks you to remember something, save it to /notes.md (append to
the file if it already exists, don't overwrite it). When they ask about their
notes, read /notes.md."""


# ---------------------------------------------------------------------------
# LAYER 3 - HARNESS: one call builds the whole agent.
# ---------------------------------------------------------------------------

def is_daily_limit(error: Exception) -> bool:
    return "PerDay" in str(error)


def should_retry(error: Exception) -> bool:
    """Retry the per-minute rate limit (it clears in seconds), but not the daily
    limit (it clears tomorrow - waiting would just look like the agent froze)."""
    if isinstance(error, GoogleRateLimitError) and not is_daily_limit(error):
        print("   ⏳ Gemini's per-minute limit reached - waiting, then retrying...")
        return True
    return False


def build_agent():
    WORKSPACE.mkdir(exist_ok=True)
    return create_deep_agent(
        model=make_model(),
        tools=[get_current_time, calculate],
        system_prompt=SYSTEM_PROMPT,
        # Real files on disk, locked inside ./workspace (paths like /notes.md map there).
        backend=FilesystemBackend(root_dir=WORKSPACE, virtual_mode=True),
        # Remembers the conversation between your messages (lost when you quit).
        checkpointer=InMemorySaver(),
        # Gemini's free tier allows only a few requests per minute, and one question
        # can take several calls. When we hit the per-minute limit, wait and try
        # again (20s, 30s, 45s, 60s) instead of crashing.
        middleware=[
            ModelRetryMiddleware(
                retry_on=should_retry,
                max_retries=4,
                initial_delay=20,
                backoff_factor=1.5,
                max_delay=60,
                on_failure=lambda e: "Sorry, I hit Gemini's free-tier rate limit. Please wait a minute and ask again.",
            )
        ],
    )


def chat(agent, user_input: str, thread_id: str = "main") -> str:
    """Send one message and print each step the agent takes along the way:
    every LLM response, every tool it asks for, and every tool result."""
    config = {"configurable": {"thread_id": thread_id}}
    final = ""
    llm_calls = 0
    for update in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="updates",
    ):
        for step in update.values():
            for msg in (step or {}).get("messages", []) if isinstance(step, dict) else []:
                if isinstance(msg, AIMessage):
                    # One AIMessage = one response from the LLM.
                    llm_calls += 1
                    usage = msg.usage_metadata or {}
                    print(f"   🧠 LLM response #{llm_calls} "
                          f"({usage.get('input_tokens', '?')} tokens in, {usage.get('output_tokens', '?')} out)")
                    if msg.text:
                        label = "final answer" if not msg.tool_calls else "says"
                        print(f"      💬 {label}: {msg.text.strip()}")
                    for call in msg.tool_calls:
                        print(f"      🔧 wants tool: {call['name']}({call['args']})")
                    if not msg.tool_calls and msg.text:
                        final = msg.text
                elif isinstance(msg, ToolMessage):
                    preview = str(msg.content).strip().replace("\n", " ")
                    print(f"      ↳ {preview[:100]}{'…' if len(preview) > 100 else ''}")
    return final


def main():
    agent = build_agent()
    print("Deep Agent ready. Type a message (or 'quit').\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        try:
            print(f"Agent: {chat(agent, user_input)}\n")
        except GoogleRateLimitError as e:
            if not is_daily_limit(e):
                raise
            print("Agent: Sorry, this Gemini model's free daily limit is used up. "
                  "Try again tomorrow, or set a different GEMINI_MODEL in .env.\n")


if __name__ == "__main__":
    main()
