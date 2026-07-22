#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENV_BLOCK_RE = re.compile(r"^\s*<environment_context>.*?</environment_context>\s*$", re.S)


@dataclass
class Message:
    timestamp: str
    role: str
    phase: str | None
    text: str


@dataclass
class Session:
    session_id: str
    started_at: str
    cwd: str
    cli_version: str | None
    jsonl_path: Path
    messages: list[Message]

    @property
    def first_user_message(self) -> str:
        for message in self.messages:
            if message.role == "user":
                return one_line(message.text)
        return "(no user message found)"


def one_line(text: str, limit: int = 120) -> str:
    flattened = " ".join(text.strip().split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1] + "…"


def iter_session_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("rollout-*.jsonl"))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text_blocks(content: list[dict]) -> str:
    parts: list[str] = []
    for item in content:
        kind = item.get("type")
        if kind in {"input_text", "output_text"}:
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def should_skip_user_text(text: str) -> bool:
    return not text or ENV_BLOCK_RE.match(text) is not None


def parse_session(path: Path, target_cwd: str) -> Session | None:
    session_id = None
    started_at = None
    cwd = None
    cli_version = None
    messages: list[Message] = []

    for item in read_jsonl(path):
        item_type = item.get("type")
        payload = item.get("payload", {})

        if item_type == "session_meta":
            cwd = payload.get("cwd")
            if cwd != target_cwd:
                return None
            session_id = payload.get("id")
            started_at = payload.get("timestamp")
            cli_version = payload.get("cli_version")
            continue

        if item_type != "response_item":
            continue

        if payload.get("type") != "message":
            continue

        role = payload.get("role")
        if role not in {"user", "assistant"}:
            continue

        text = extract_text_blocks(payload.get("content", []))
        if role == "user" and should_skip_user_text(text):
            continue
        if not text:
            continue

        messages.append(
            Message(
                timestamp=item.get("timestamp", ""),
                role=role,
                phase=payload.get("phase"),
                text=text,
            )
        )

    if cwd != target_cwd or not session_id or not started_at:
        return None

    return Session(
        session_id=session_id,
        started_at=started_at,
        cwd=cwd,
        cli_version=cli_version,
        jsonl_path=path,
        messages=messages,
    )


def session_slug(session: Session) -> str:
    stamp = session.started_at.replace(":", "-")
    return f"{stamp}_{session.session_id}"


def write_session_markdown(session: Session, output_dir: Path) -> Path:
    output_path = output_dir / f"{session_slug(session)}.md"
    lines = [
        f"# Codex Session {session.session_id}",
        "",
        f"- Started: `{session.started_at}`",
        f"- CWD: `{session.cwd}`",
        f"- CLI version: `{session.cli_version or 'unknown'}`",
        f"- Source JSONL: `{session.jsonl_path}`",
        f"- Message count: `{len(session.messages)}`",
        "",
    ]

    for index, message in enumerate(session.messages, start=1):
        phase_suffix = f" [{message.phase}]" if message.phase else ""
        lines.extend(
            [
                f"## {index}. {message.role.upper()}{phase_suffix}",
                "",
                f"_Timestamp: `{message.timestamp}`_",
                "",
                message.text,
                "",
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_index(sessions: list[Session], output_dir: Path, session_files: dict[str, Path]) -> Path:
    index_path = output_dir / "INDEX.md"
    lines = [
        "# Recovered Codex Sessions",
        "",
        f"Recovered `{len(sessions)}` sessions for cwd `{sessions[0].cwd if sessions else ''}`.",
        "",
    ]

    for session in sessions:
        session_file = session_files[session.session_id]
        lines.extend(
            [
                f"## {session.started_at}",
                "",
                f"- Session ID: `{session.session_id}`",
                f"- Recovered file: `{session_file.name}`",
                f"- Source JSONL: `{session.jsonl_path}`",
                f"- First user message: {session.first_user_message}",
                f"- Message count: `{len(session.messages)}`",
                "",
            ]
        )

    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover local Codex sessions for a project cwd.")
    parser.add_argument("--sessions-root", required=True, type=Path)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[Session] = []
    for path in iter_session_files(args.sessions_root):
        session = parse_session(path, args.cwd)
        if session is not None:
            sessions.append(session)

    sessions.sort(key=lambda session: session.started_at)

    session_files: dict[str, Path] = {}
    for session in sessions:
        session_files[session.session_id] = write_session_markdown(session, args.output_dir)

    write_index(sessions, args.output_dir, session_files)
    print(f"Recovered {len(sessions)} sessions into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
