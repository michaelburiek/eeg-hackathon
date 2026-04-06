from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Verbose / debug mode
# ---------------------------------------------------------------------------

_verbose: bool = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose command logging."""
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def print_cmd(cmd: list[str] | str) -> None:
    """Print the shell command about to be executed (only in verbose mode)."""
    if not _verbose:
        return
    if isinstance(cmd, list):
        import shlex

        cmd = " ".join(shlex.quote(c) for c in cmd)
    console.print(f"[dim]  $ {cmd}[/dim]")


def print_verbose(message: str) -> None:
    """Print a verbose/debug message (only in verbose mode)."""
    if _verbose:
        console.print(f"[dim]VERBOSE[/dim] {message}")


# ---------------------------------------------------------------------------
# Standard status helpers
# ---------------------------------------------------------------------------


def print_success(message: str) -> None:
    console.print(f"[bold green] ✓ [/bold green] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red] ✗ [/bold red] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold blue] → [/bold blue] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow] ⚠ [/bold yellow] {message}")


def print_rule(title: str = "", *, style: str = "dim") -> None:
    """Print a styled horizontal rule, optionally with a centred title."""
    console.print(Rule(title, style=style))


def supports_live_rendering() -> bool:
    term = (os.environ.get("TERM") or "").lower()
    return console.is_terminal and console.is_interactive and term not in {"", "dumb"}


# ---------------------------------------------------------------------------
# Spinner context manager
# ---------------------------------------------------------------------------


@contextmanager
def spinner(message: str) -> Iterator[None]:
    """Display a Rich spinner while the body of the ``with`` block runs.

    In verbose mode the spinner is suppressed so that command output is not
    interleaved with the live display.
    """
    if _verbose or not supports_live_rendering():
        console.print(f"[dim]{message}[/dim]")
        yield
        return

    with console.status(f"[dim]{message}[/dim]", spinner="dots"):
        yield


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def render_pipe_table(title: str, raw_output: str, *, highlight_user: str | None = None) -> None:
    lines = [line for line in raw_output.splitlines() if line.strip()]
    if not lines:
        print_info("No rows returned.")
        return

    header = lines[0].split("|")
    table = Table(
        title=title,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=False,
        padding=(0, 1),
    )
    for column in header:
        table.add_column(column, overflow="fold")

    for line in lines[1:]:
        cells = line.split("|")
        # Pad short rows so Rich never receives fewer cells than columns.
        if len(cells) < len(header):
            cells.extend([""] * (len(header) - len(cells)))
        row_style = None
        if highlight_user and len(cells) > 1 and cells[1] == highlight_user:
            if "RUNNING" in cells:
                row_style = "bold green"
            elif "PENDING" in cells:
                row_style = "bold yellow"
            else:
                row_style = "bold"
        table.add_row(*cells[: len(header)], style=row_style)

    console.print(table)


def state_style(state: str) -> str:
    upper = state.upper()
    if upper in {"RUNNING", "COMPLETED"}:
        return "bold green"
    if upper in {"PENDING", "CONFIGURING"}:
        return "bold yellow"
    if upper in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
        return "bold red"
    return "bold cyan"


def render_state_badge(state: str) -> Text:
    return Text(state.upper(), style=state_style(state))
