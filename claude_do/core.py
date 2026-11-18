"""Hand Claude a file; Claude does what the file says."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import anthropic

MODEL = "claude-sonnet-5-5"
MAX_OUTPUT = 8_000

SYSTEM = (
    "You are an agent. The user message is a task file. Do exactly what it asks, "
    "using the tools provided inside the working directory. Work step by step, "
    "verify your results, then finish with a short summary of what you did."
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict[str, str]
    run: Callable[..., str]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in self.properties.items()},
                "required": list(self.properties),
            },
        }


class Workspace:
    """A sandboxed directory plus the tools that may touch it."""

    def __init__(self, root: Path, approve: Callable[[str], bool]):
        self.root = root.resolve()
        self.approve = approve

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError(f"{relative!r} is outside the working directory")
        return path

    def read_file(self, path: str) -> str:
        return self._path(path).read_text()

    def write_file(self, path: str, content: str) -> str:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def list_files(self, path: str = ".") -> str:
        base = self._path(path)
        return "\n".join(sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()))

    def run_command(self, command: str) -> str:
        if not self.approve(command):
            return "the user declined to run this command"
        done = subprocess.run(
            command, shell=True, cwd=self.root, capture_output=True, text=True, timeout=60
        )
        return f"exit {done.returncode}\n{done.stdout}{done.stderr}"[:MAX_OUTPUT]

    def tools(self) -> dict[str, Tool]:
        catalog = [
            Tool("read_file", "Read a text file.", {"path": "relative file path"}, self.read_file),
            Tool(
                "write_file",
                "Create or overwrite a text file.",
                {"path": "relative file path", "content": "full file content"},
                self.write_file,
            ),
            Tool("list_files", "List files under a directory.", {"path": "relative directory"}, self.list_files),
            Tool("run_command", "Run a shell command.", {"command": "the command"}, self.run_command),
        ]
        return {tool.name: tool for tool in catalog}


def _call(tool: Tool | None, arguments: dict[str, Any]) -> tuple[str, bool]:
    if tool is None:
        return "unknown tool", True
    try:
        return tool.run(**arguments), False
    except Exception as error:  # surfaced to Claude so it can recover
        return f"{type(error).__name__}: {error}", True


def run_task(
    task: str,
    workspace: Workspace,
    *,
    model: str = MODEL,
    max_turns: int = 25,
    on_event: Callable[[str], None] = print,
) -> str:
    client = anthropic.Anthropic()
    tools = workspace.tools()
    specs = [tool.spec() for tool in tools.values()]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

    for _ in range(max_turns):
        reply = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM, tools=specs, messages=messages
        )
        messages.append({"role": "assistant", "content": reply.content})
        text = "".join(block.text for block in reply.content if block.type == "text")

        if reply.stop_reason != "tool_use":
            return text

        results = []
        for block in reply.content:
            if block.type != "tool_use":
                continue
            on_event(f"-> {block.name} {block.input}")
            output, failed = _call(tools.get(block.name), block.input)
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": failed}
            )
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"task not finished after {max_turns} turns")
