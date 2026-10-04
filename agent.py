"""
My Deep Agent: a personal assistant built with LangChain Deep Agents + Claude.

The four layers in this file:
  MODEL      Claude Opus 5.5                    -> make_model()
  SDK/API    langchain-anthropic (calls Anthropic's API for you)
  HARNESS    LangChain Deep Agents              -> create_deep_agent(...)
  YOUR AGENT 2 custom tools + a system prompt + a notes folder

Compared with the hand-written version, notice what's MISSING here:
there is no agent loop. Deep Agents runs the loop for you.

Run it:   .venv/bin/python agent.py
"""

import ast
import operator
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

WORKSPACE = Path(__file__).parent / "workspace"  # the only folder the agent can read/write


# ---------------------------------------------------------------------------
# LAYER 1 - MODEL: Claude, via LangChain's Anthropic connector.
# Swapping the model (GPT, Gemini, Ollama...) only means changing this function.
# ---------------------------------------------------------------------------

def make_model():
    return ChatAnthropic(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},  # how hard Claude thinks: low / medium / high
    )


# ---------------------------------------------------------------------------
# YOUR TOOLS: plain Python functions. Deep Agents reads the function name,
# the type hints, and the docstring, and turns them into the tool description
# Claude sees. (In the hand-written version you wrote that JSON by hand.)
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
    )


def chat(agent, user_input: str, thread_id: str = "main") -> str:
    """Send one message and print each step the agent takes along the way."""
    config = {"configurable": {"thread_id": thread_id}}
    final = ""
    for update in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="updates",
    ):
        for step in update.values():
            for msg in (step or {}).get("messages", []) if isinstance(step, dict) else []:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        print(f"   🔧 {call['name']}({call['args']})")
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
        if user_input:
            print(f"Agent: {chat(agent, user_input)}\n")


if __name__ == "__main__":
    main()
