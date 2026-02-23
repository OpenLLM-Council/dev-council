import json
import os
import sys
import urllib.request
import urllib.error
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

console = Console()
CONFIG_DIR = os.path.expanduser("~/.dev-council")
os.makedirs(CONFIG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(CONFIG_DIR, "llm_config.json")


def onboard_ollama():
    """Interactive CLI to onboard Ollama models."""
    console.rule("[bold cyan]Ollama Model Onboarding[/bold cyan]")

    base_url = Prompt.ask(
        "[bold yellow]Enter Ollama Base URL[/bold yellow]",
        default="http://localhost:11434",
    ).rstrip("/")

    # Fetch available models
    with console.status(
        f"[bold green]Fetching models from {base_url}/api/tags...", spinner="dots"
    ):
        try:
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            console.print(
                f"[bold red]Failed to connect to Ollama at {base_url}: {e.reason}[/bold red]"
            )
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]Error fetching models: {e}[/bold red]")
            sys.exit(1)

    models = data.get("models", [])
    if not models:
        console.print(
            "[bold yellow]No models found at the specified Ollama instance.[/bold yellow]"
        )
        sys.exit(0)

    model_names = [model["name"] for model in models]

    console.print(Panel("[bold]Available Models:[/bold]", border_style="green"))
    for idx, name in enumerate(model_names):
        console.print(f"  [bold cyan]{idx + 1}.[/bold cyan] {name}")

    console.print("")
    selections_str = Prompt.ask(
        "[bold yellow]Enter the numbers of the models you want to use (comma-separated)[/bold yellow]"
    )

    selected_models = []
    try:
        indices = [int(i.strip()) - 1 for i in selections_str.split(",") if i.strip()]
        for idx in set(indices):
            if 0 <= idx < len(model_names):
                selected_models.append(model_names[idx])
            else:
                console.print(
                    f"[bold red]Warning: Index {idx + 1} is out of range. Skipping.[/bold red]"
                )
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter numbers separated by commas.[/bold red]"
        )
        sys.exit(1)

    if not selected_models:
        console.print("[bold yellow]No valid models selected. Exiting.[/bold yellow]")
        sys.exit(0)

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    config["ollama_base_url"] = base_url
    config.setdefault("ollama_models", [])

    for model in selected_models:
        if model not in config["ollama_models"]:
            config["ollama_models"].append(model)

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    console.print(f"\n[bold green]✓ Configuration saved to {CONFIG_FILE}[/bold green]")
    console.print("[bold]Selected Models:[/bold]")
    for model in selected_models:
        console.print(f"  - {model}")
