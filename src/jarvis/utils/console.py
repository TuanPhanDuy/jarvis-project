from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.status import Status
from rich.table import Table

console = Console()


def print_welcome() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]J.A.R.V.I.S[/bold cyan]\n"
            "[dim]Just A Rather Very Intelligent System[/dim]\n\n"
            "[white]AI Research Agent — powered by Ollama[/white]\n"
            "[dim]Type your research question. Type [bold]exit[/bold] to quit.[/dim]",
            border_style="cyan",
        )
    )


def print_response(text: str) -> None:
    console.print()
    console.print(
        Panel(
            Markdown(text),
            title="[bold cyan]JARVIS[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def get_user_input() -> str:
    return Prompt.ask("[bold green]You[/bold green]")


def thinking_status(message: str = "JARVIS is thinking...") -> Status:
    return console.status(f"[cyan]{message}[/cyan]", spinner="dots")


def stream_start() -> None:
    """Print the JARVIS response header before streaming text begins."""
    console.print()
    console.rule("[bold cyan]JARVIS[/bold cyan]", style="cyan")


def stream_chunk(text: str) -> None:
    """Print a single streaming chunk without a trailing newline."""
    console.print(text, end="", highlight=False)


def stream_end() -> None:
    """Print the closing rule after streaming finishes."""
    console.print()
    console.rule(style="cyan")
    console.print()


def listening_status() -> Status:
    """Spinner shown while recording microphone input."""
    return console.status("[bold green]Listening...[/bold green]", spinner="arc")


def print_voice_input(text: str) -> None:
    """Echo the transcribed speech so the user can confirm what was heard."""
    console.print(f"[bold green]You (voice):[/bold green] {text}")


def print_voice_welcome() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]J.A.R.V.I.S — Voice Mode[/bold cyan]\n"
            "[dim]Just A Rather Very Intelligent System[/dim]\n\n"
            "[white]Speak your question. JARVIS will reply aloud.[/white]\n"
            "[dim]Say [bold]exit[/bold] or [bold]quit[/bold] to stop.[/dim]",
            border_style="green",
        )
    )


def print_usage_summary(usage: dict) -> None:
    """Render a session token usage and cost summary table."""
    table = Table(title="Session Usage", border_style="dim", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Prompt tokens", f"{usage.get('prompt_tokens', 0):,}")
    table.add_row("Completion tokens", f"{usage.get('completion_tokens', 0):,}")
    table.add_row("Total tokens", f"{usage.get('total_tokens', 0):,}")
    table.add_row(
        "[bold]Cost[/bold]",
        "[bold green]$0.00 (local model)[/bold green]",
    )

    console.print()
    console.print(table)
    console.print()
