from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_console = Console()


_DEFAULT_AGENT_NAME = "ClawSwarm"
_DEFAULT_AGENT_DESCRIPTION = (
    "A hierarchical swarm of agents that can " "handle complex tasks"
)
_DEFAULT_WORKER_MODEL = "gpt-5.4"
_DEFAULT_GATEWAY_HOST = "[::]"
_DEFAULT_GATEWAY_PORT = 50051
_DEFAULT_GATEWAY_TLS = False
_DEFAULT_API_PORT = 8080
_DEFAULT_API_KEY = ""


@dataclass
class WorkerAgentConfig:
    """Configuration for a single worker agent."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    max_tokens: int = 0
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerAgentConfig":
        """Create a WorkerAgentConfig from a dictionary."""
        if not data:
            return cls()
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            system_prompt=str(data.get("system_prompt", "")),
            model=str(data.get("model", "")),
            max_tokens=int(data.get("max_tokens", 0) or 0),
            temperature=float(data.get("temperature", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, omitting empty values."""
        result: dict[str, Any] = {}
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.system_prompt:
            result["system_prompt"] = self.system_prompt
        if self.model:
            result["model"] = self.model
        if self.max_tokens > 0:
            result["max_tokens"] = self.max_tokens
        if self.temperature > 0.0:
            result["temperature"] = self.temperature
        return result


@dataclass
class DirectorConfig:
    """Configuration for the director agent."""

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    max_tokens: int = 0
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectorConfig":
        """Create a DirectorConfig from a dictionary."""
        if not data:
            return cls()
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            system_prompt=str(data.get("system_prompt", "")),
            model=str(data.get("model", "")),
            max_tokens=int(data.get("max_tokens", 0) or 0),
            temperature=float(data.get("temperature", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, omitting empty values."""
        result: dict[str, Any] = {}
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.system_prompt:
            result["system_prompt"] = self.system_prompt
        if self.model:
            result["model"] = self.model
        if self.max_tokens > 0:
            result["max_tokens"] = self.max_tokens
        if self.temperature > 0.0:
            result["temperature"] = self.temperature
        return result


@dataclass
class AgentConfig:
    """Configuration for all agents (director + workers)."""

    director: DirectorConfig = None  # type: ignore[assignment]
    workers: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.director is None:
            self.director = DirectorConfig()
        if self.workers is None:
            self.workers = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create an AgentConfig from a dictionary."""
        if not data:
            return cls()
        director_data = data.get("director", {}) or {}
        workers_data = data.get("workers", {}) or {}
        workers: dict = {}
        for key, value in workers_data.items():
            if isinstance(value, dict):
                workers[key] = WorkerAgentConfig.from_dict(value)
            else:
                workers[key] = WorkerAgentConfig()
        return cls(
            director=DirectorConfig.from_dict(director_data),
            workers=workers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "director": self.director.to_dict(),
        }
        if self.workers:
            result["workers"] = {
                k: v.to_dict() for k, v in self.workers.items()
            }
        return result


@dataclass
class ClawConfig:
    """Central configuration for ClawSwarm."""

    agent_name: str = _DEFAULT_AGENT_NAME
    agent_description: str = _DEFAULT_AGENT_DESCRIPTION
    worker_model_name: str = _DEFAULT_WORKER_MODEL
    # Gateway
    gateway_host: str = _DEFAULT_GATEWAY_HOST
    gateway_port: int = _DEFAULT_GATEWAY_PORT
    gateway_tls: bool = _DEFAULT_GATEWAY_TLS
    # HTTP API
    api_port: int = _DEFAULT_API_PORT
    api_key: str = _DEFAULT_API_KEY
    # Runtime
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "ClawConfig":
        agent = data.get("agent", {}) or {}
        worker = data.get("worker", {}) or {}
        gateway = data.get("gateway", {}) or {}
        api = data.get("api", {}) or {}
        runtime = data.get("runtime", {}) or {}

        return cls(
            agent_name=agent.get("name", _DEFAULT_AGENT_NAME),
            agent_description=agent.get(
                "description", _DEFAULT_AGENT_DESCRIPTION
            ),
            worker_model_name=worker.get(
                "model_name", _DEFAULT_WORKER_MODEL
            ),
            gateway_host=gateway.get("host", _DEFAULT_GATEWAY_HOST),
            gateway_port=int(
                gateway.get("port", _DEFAULT_GATEWAY_PORT)
            ),
            gateway_tls=bool(
                gateway.get("tls", _DEFAULT_GATEWAY_TLS)
            ),
            api_port=int(api.get("port", _DEFAULT_API_PORT)),
            api_key=str(api.get("key", _DEFAULT_API_KEY)),
            verbose=bool(runtime.get("verbose", False)),
        )

    def to_dict(self) -> dict:
        return {
            "agent": {
                "name": self.agent_name,
                "description": self.agent_description,
            },
            "worker": {"model_name": self.worker_model_name},
            "gateway": {
                "host": self.gateway_host,
                "port": self.gateway_port,
                "tls": bool(self.gateway_tls),
            },
            "api": {
                "port": self.api_port,
                "key": self.api_key,
            },
            "runtime": {"verbose": bool(self.verbose)},
        }


def _find_project_root(start: str | None = None) -> Path:
    """
    Find the project root by walking up from ``start`` (or cwd) until
    we see a ``pyproject.toml`` or ``.git`` directory.
    """
    cwd = Path(start or os.getcwd()).resolve()
    for _ in range(10):
        if (cwd / "pyproject.toml").is_file() or (
            cwd / ".git"
        ).is_dir():
            return cwd
        parent = cwd.parent
        if parent == cwd:
            break
        cwd = parent
    return Path(start or os.getcwd()).resolve()


def config_path(start: str | None = None) -> Path:
    """Return the expected path to ``claw_config.yaml``."""
    root = _find_project_root(start)
    return root / "claw_config.yaml"


def load_config(path: Path | None = None) -> ClawConfig:
    """Load configuration from ``claw_config.yaml`` if present."""
    cfg_path = path or config_path()
    if not cfg_path.is_file():
        return ClawConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ClawConfig.from_dict(data)


def agent_config_path(start: str | None = None) -> Path:
    """Return the expected path to ``claw_swarm_agents.yaml``."""
    root = _find_project_root(start)
    return root / "claw_swarm_agents.yaml"


def _validate_agent_config(cfg: AgentConfig) -> list[str]:
    """
    Validate agent config and return list of warnings for invalid/missing fields.

    Returns a list of warning messages (empty if config is valid).
    """
    warnings: list[str] = []
    valid_worker_keys = {"search", "response", "developer", "token_launch"}
    for key in cfg.workers:
        if key not in valid_worker_keys:
            warnings.append(
                f"Unknown worker key '{key}' in agents config. "
                f"Expected one of: {', '.join(sorted(valid_worker_keys))}"
            )
    return warnings


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """
    Load agent configuration from ``claw_swarm_agents.yaml`` if present.

    The file format is::

        director:
          name: "CustomName"
          description: "Custom description"
          system_prompt: "Custom prompt..."
          model: "claude-sonnet-4-6"
          max_tokens: 8096
          temperature: 0.7

        workers:
          search:
            name: "CustomSearch"
            description: "Custom search description"
            system_prompt: "Custom search prompt..."
          response:
            name: "CustomResponse"
          developer:
            name: "CustomDev"
          token_launch:
            name: "CustomTokenLaunch"

    All fields are optional. If omitted, defaults from the existing agent
    definitions are used.

    Args:
        path: Explicit path to the agent config file. If None, searches
            in order: AGENT_CONFIG_PATH env var > project root.

    Returns:
        AgentConfig with all agent settings (unspecified fields are empty strings/0).

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if path:
        cfg_path = Path(path)
    else:
        env_path = os.environ.get("AGENT_CONFIG_PATH")
        if env_path:
            cfg_path = Path(env_path)
        else:
            cfg_path = agent_config_path()

    if not cfg_path.is_file():
        return AgentConfig()

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = AgentConfig.from_dict(data)

    warnings = _validate_agent_config(cfg)
    for w in warnings:
        _console.print(f"[yellow]Warning: {w}[/yellow]")

    return cfg


def apply_config_to_env(cfg: ClawConfig) -> None:
    """
    Map config values into environment variables used by the rest of
    the codebase.

    Existing env vars take precedence; config only fills in defaults.
    """
    os.environ.setdefault("WORKER_MODEL_NAME", cfg.worker_model_name)
    os.environ.setdefault("CLAWSWARM_AGENT_NAME", cfg.agent_name)
    os.environ.setdefault(
        "CLAWSWARM_AGENT_DESCRIPTION", cfg.agent_description
    )
    os.environ.setdefault("GATEWAY_HOST", cfg.gateway_host)
    os.environ.setdefault("GATEWAY_PORT", str(cfg.gateway_port))
    os.environ.setdefault(
        "GATEWAY_TLS", "1" if cfg.gateway_tls else "0"
    )
    os.environ.setdefault("API_PORT", str(cfg.api_port))
    if cfg.api_key:
        os.environ.setdefault("API_KEY", cfg.api_key)
    os.environ["CLAWSWARM_VERBOSE"] = "1" if cfg.verbose else "0"


def _prompt(label: str, default: str) -> str:
    """Ask a text question with a styled arrow prompt."""
    return Prompt.ask(
        f"  [bold cyan]❯[/bold cyan] [bold]{label}[/bold]",
        default=default,
        console=_console,
    )


def _prompt_int(label: str, default: int) -> int:
    """Ask an integer question with validation."""
    while True:
        raw = Prompt.ask(
            f"  [bold cyan]❯[/bold cyan] [bold]{label}[/bold]",
            default=str(default),
            console=_console,
        )
        try:
            return int(raw)
        except ValueError:
            _console.print(
                f"  [bold red]✗[/bold red] Please enter a valid"
                f" integer (got [yellow]{raw!r}[/yellow])"
            )


def _confirm(label: str, default: bool = False) -> bool:
    """Ask a yes/no question with a styled arrow prompt."""
    return Confirm.ask(
        f"  [bold cyan]❯[/bold cyan] [bold]{label}[/bold]",
        default=default,
        console=_console,
    )


def onboarding_interactive(*, force: bool = False) -> ClawConfig:
    """
    Run onboarding to create ``claw_config.yaml`` in the project root.

    If the file already exists, it is not overwritten unless
    ``force`` is True.
    """
    cfg_path = config_path()
    if cfg_path.is_file() and not force:
        cfg = load_config(cfg_path)
        apply_config_to_env(cfg)
        _console.print()
        _console.print(
            Panel(
                Text.from_markup(
                    "[bold yellow]Config already exists[/bold yellow]"
                    f" at [cyan]{cfg_path}[/cyan]\n\n"
                    "Re-run with [bold]--force[/bold] to overwrite."
                ),
                title="[bold yellow]ClawSwarm[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        _console.print()
        return cfg

    # ── Welcome banner ───────────────────────────────────────────────
    _console.print()
    _console.print(
        Panel(
            Text.from_markup(
                "[bold white]Welcome to ClawSwarm[/bold white]\n\n"
                "Let's configure your [cyan]claw_config.yaml[/cyan].\n"
                "This file controls the hierarchical agent swarm\n"
                "that powers your Telegram, Discord, and WhatsApp bots.",
                justify="center",
            ),
            title="[bold cyan]🦀  Setup Wizard[/bold cyan]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # ── Agent ────────────────────────────────────────────────────────
    _console.print()
    _console.print(Rule("[dim]Agent[/dim]", style="cyan"))
    _console.print()
    agent_name = _prompt("Agent name", _DEFAULT_AGENT_NAME)
    agent_description = _prompt(
        "Agent description", _DEFAULT_AGENT_DESCRIPTION
    )

    # ── Worker ───────────────────────────────────────────────────────
    _console.print()
    _console.print(Rule("[dim]Worker[/dim]", style="cyan"))
    _console.print()
    worker_model_name = _prompt(
        "Worker model name", _DEFAULT_WORKER_MODEL
    )

    # ── Gateway ──────────────────────────────────────────────────────
    _console.print()
    _console.print(Rule("[dim]Gateway (gRPC)[/dim]", style="cyan"))
    _console.print()
    gateway_host = _prompt("Bind host", _DEFAULT_GATEWAY_HOST)
    gateway_port = _prompt_int("Bind port", _DEFAULT_GATEWAY_PORT)
    gateway_tls = _confirm("Enable TLS?", _DEFAULT_GATEWAY_TLS)

    # ── HTTP API ─────────────────────────────────────────────────────
    _console.print()
    _console.print(Rule("[dim]HTTP API[/dim]", style="cyan"))
    _console.print()
    api_port = _prompt_int("API port", _DEFAULT_API_PORT)
    api_key = _prompt(
        "API key [dim](leave blank for open access)[/dim]",
        _DEFAULT_API_KEY,
    )

    # ── Runtime ──────────────────────────────────────────────────────
    _console.print()
    _console.print(Rule("[dim]Runtime[/dim]", style="cyan"))
    _console.print()
    verbose = _confirm("Enable verbose logging?", False)

    cfg = ClawConfig(
        agent_name=agent_name,
        agent_description=agent_description,
        worker_model_name=worker_model_name,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
        gateway_tls=gateway_tls,
        api_port=api_port,
        api_key=api_key,
        verbose=verbose,
    )

    # ── Summary table ────────────────────────────────────────────────
    _console.print()
    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column(style="bold dim", justify="right")
    table.add_column(style="cyan")
    table.add_row("agent name", agent_name)
    table.add_row("description", agent_description)
    table.add_row("worker model", worker_model_name)
    table.add_row("gateway host", gateway_host)
    table.add_row("gateway port", str(gateway_port))
    table.add_row(
        "gateway tls",
        "[green]yes[/green]" if gateway_tls else "no",
    )
    table.add_row("api port", str(api_port))
    table.add_row(
        "api key",
        "[dim](open)[/dim]" if not api_key else api_key[:8] + "...",
    )
    table.add_row(
        "verbose",
        "[green]yes[/green]" if verbose else "no",
    )
    _console.print(
        Panel(
            table,
            title="[bold]Configuration Summary[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    _console.print()

    # ── Write file ───────────────────────────────────────────────────
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg.to_dict(),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    _console.print(
        Panel(
            Text.from_markup(
                "[bold green]✓[/bold green]  Config written to"
                f" [cyan]{cfg_path}[/cyan]"
            ),
            border_style="green",
            padding=(0, 2),
        )
    )
    _console.print()

    apply_config_to_env(cfg)
    return cfg


def ensure_config_interactive() -> ClawConfig:
    """
    Ensure that ``claw_config.yaml`` exists in the project root.

    If the file is missing, run a small interactive wizard to collect
    the core settings and write the file. Always applies the resulting
    config to the current environment.
    """
    cfg_path = config_path()
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_env(cfg)
        return cfg

    _console.print()
    _console.print(
        Panel(
            Text.from_markup(
                "[bold yellow]No claw_config.yaml found.[/bold yellow]\n\n"
                "Let's create one with your preferred settings."
            ),
            title="[bold cyan]🦀  ClawSwarm[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    return onboarding_interactive(force=True)
